"""Fits a nonlinear GP to a dataset and returns analysis results in an interactible form."""

import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from GPy.util.initialization import initialize_latent
from GPy.core.parameterization.variational import VariationalPosterior
from GPy.core import GP
from GPy.kern import Kern, RBF
from GPy.likelihoods import Likelihood, Gaussian
from GPy.models import GPLVM, BayesianGPLVM
from GPy.models.mrd import MRD

from mri_analysis.datatypes import (
    DATA_FEATURES,
    GPType,
    DataProcessingType,
    InducedInitializationType,
    LatentInitializationType,
    CovarianceOutput,
    LatentOutput,
    PredictionOutput,
    SensitivityOutput,
)
from mri_analysis.constants import MODELS_PATH, RESULTS_PATH

from loguru import logger

from mri_analysis.utils import get_time_identifier


def get_gp_from_type(
    gp_type: GPType,
    data: np.ndarray,
    X: np.ndarray,
    n_components: int,
    n_inducing: int,
    kernel: Kern,
    likelihood: Likelihood,
    name: str,
    Z: np.ndarray = None,
) -> GP:
    match gp_type:
        case "GP":
            gp = GPLVM(
                data,
                n_components,
                X=X,
                kernel=kernel,
                name=name,
            )
        case "Bayesian":
            gp = BayesianGPLVM(
                data,
                n_components,
                X=X,
                Z=Z,
                kernel=kernel,
                likelihood=likelihood,
                num_inducing=n_inducing,
                name=name,
            )
        case _:
            logger.warning(
                f"Unrecognized GP type: {gp_type}. Using GPLVM instead."
            )
            gp = GPLVM(data, n_components, X=X, kernel=kernel, name=name)
    return gp


class GPAnalysis:
    data: pd.DataFrame = None
    gps: List[GP] = None

    def __init__(
        self,
        kernels: Union[Kern, List[Kern]],
        gp_type: GPType = "GP",
        data_processing: DataProcessingType = "none",
        latent_initialization: LatentInitializationType = "pca",
        induced_initialization: InducedInitializationType = "permute",
        multi_dimensional: bool = False,
        use_mrd: bool = True,
        remove_components: int = None,
        n_inducing: int = 25,
        n_restarts: int = 10,
        fixed_optimization: bool = False,
        burst_optimization: bool = False,
        n_optimization_iters: int = 5e4,
        expand_args: Dict[str, Union[int, float]] = {},
        name: str = None,
    ):
        self.kernels = kernels if type(kernels) == list else kernels
        self.gp_type = gp_type
        self.data_processing = data_processing
        self.latent_initialization = latent_initialization
        self.induced_initialization = induced_initialization
        self.multi_dimensional = multi_dimensional
        self.name = name
        # gp parameters
        self.n_inducing = n_inducing
        self.n_restarts = n_restarts
        self.fixed_optimization = fixed_optimization
        self.burst_optimization = burst_optimization
        self.n_optimization_iters = n_optimization_iters
        self.expand_args = {
            "dim": 500,
            "noise": 1e-6,
            "var": 1.0,
        }
        for key, value in expand_args.items():
            self.expand_args[key] = value
        self.use_mrd = use_mrd
        self.remove_components = remove_components

    def fit(
        self,
        data: pd.DataFrame,
        n_components: int,
        features: List[str] = DATA_FEATURES,
        optimize: bool = True,
        optimization_method: str = "bfgs",
        flat_data: bool = False,
        tol: float = None,
    ) -> None:
        """Fits the model to some data. Required before calling any other methods."""
        # set name based on parameters
        if self.name is None:
            self.name = f"{self.gp_type}_{'flat' if flat_data else 'md'}_{[kern.name for kern in self.kernels]}_{n_components}"
        else:
            self.name = f"{self.name}_{'flat' if flat_data else 'md'}"
        self.data = data.copy()
        ## setup data
        if flat_data:
            self.features = ["Value"]
        else:
            self.features = features
        if isinstance(self.data.columns, pd.MultiIndex):  # subject data
            self.labels = self.data.droplevel(
                level=list(range(1, self.data.columns.nlevels)), axis="columns"
            ).reset_index()[
                list(set(self.data.index.names) - set(self.features))
            ]
        else:
            non_number_columns = set(
                self.data.select_dtypes(exclude="number").columns
            ) - set(self.features)
            self.labels = self.data[list(non_number_columns)].reset_index(
                drop=True
            )
        assert len(self.features) >= 1, "No numerical features found in data."
        # remove linear PCA components
        if self.remove_components is not None:
            pca = PCA(n_components=len(self.features))
            trans_y = pca.fit_transform(self.data[self.features].to_numpy())
            logger.info(f"PCA variances: {pca.explained_variance_}")
            modified_components = pca.components_.copy()
            modified_components[: self.remove_components] = 0
            new_data = trans_y @ modified_components + pca.mean_
            new_data -= new_data.mean()
            new_data /= new_data.std()
            self.data.loc[
                :,
                (
                    self.features
                    if not isinstance(self.data.columns, pd.MultiIndex)
                    else (self.features, slice(None))
                ),
            ] = new_data
        self.Y = self.data[self.features].to_numpy()
        # setup YList
        Y_list = [self.data[feature].to_numpy() for feature in self.features]
        # fix 1-D Y_list
        Y_list = [Y.reshape(-1, 1) if Y.ndim < 2 else Y for Y in Y_list]
        if self.multi_dimensional:
            # single element YList
            Y_list = [np.hstack(Y_list)]
            self.features = ["All"]
        # expand dimensions if enabled
        match self.data_processing:
            case "expand":
                # multiply each array by high-dimensional random noise
                Y_list = [
                    Y.dot(
                        self.expand_args["var"]
                        * np.random.randn(Y.shape[1], self.expand_args["dim"])
                    )
                    + self.expand_args["noise"]
                    * np.random.randn(Y.shape[0], self.expand_args["dim"])
                    for Y in Y_list
                ]
                Y_list = [Y - Y.mean(0) for Y in Y_list]
                Y_list = [Y / Y.std(0) for Y in Y_list]
            case _:
                logger.warning(
                    "Data processing type not recognized. Using none."
                )
        Y = np.hstack(Y_list)
        Y -= Y.mean()
        Y /= Y.std()
        # initialize latent space
        match self.latent_initialization:
            case "pca":
                X, _ = initialize_latent("PCA", n_components, Y)
            case "random":
                X = np.random.randn(Y.shape[0], n_components)
            case "rbf_random":
                kern = RBF(n_components)
                t = np.c_[
                    [
                        np.linspace(-1, 5, Y.shape[0])
                        for _ in range(n_components)
                    ]
                ]
                X = np.random.multivariate_normal(
                    np.arange(n_components),
                    kern.K(t),
                    Y.shape[0],
                )
            case "cluster_random":
                centers = np.random.uniform(-5, 5, len(Y_list))
                components = np.array_split(range(n_components), len(Y_list))
                X = np.hstack(
                    [
                        np.random.normal(
                            centers[i],
                            1,
                            (Y.shape[0], len(components[i])),
                        )
                        for i in range(len(Y_list))
                    ]
                )
            case "data":
                X = Y.copy().dot(np.random.randn(Y.shape[1], n_components))
            case _:
                logger.warning(
                    f"Unrecognized latent initialization type: {self.latent_initialization}. Using PCA."
                )
                X, _ = initialize_latent("PCA", n_components, Y)
        X -= X.mean()
        X /= X.std()
        # initialize inducing points
        match self.induced_initialization:
            case "permute":
                Z = np.random.permutation(X.copy())[: self.n_inducing]
            case "random":
                Z = np.random.randn(self.n_inducing, n_components) * X.var()
            case _:
                logger.warning(
                    f"Unrecognized induced initialization type: {self.induced_initialization}. Using permute."
                )
                Z = np.random.permutation(X.copy())[: self.n_inducing]
        ## setup MRD model
        if self.use_mrd:
            logger.info("Using MRD model...")
            print([Y.shape for Y in Y_list])
            self.model = MRD(
                Y_list,
                n_components,
                X=X,
                Z=Z,
                num_inducing=self.n_inducing,
                kernel=self.kernels,
                Ynames=self.features,
                name=self.name,
            )
        ## setup single GP model
        else:
            logger.info("Using GP-LVM model...")
            # set noise as 1% of variance in data
            likelihood_variance = np.var(Y) * 0.01
            likelihood = Gaussian(variance=likelihood_variance)
            self.model = get_gp_from_type(
                self.gp_type,
                Y,
                X,
                n_components,
                self.n_inducing,
                (
                    self.kernels
                    if isinstance(self.kernels, Kern)
                    else self.kernels[0]
                ),
                likelihood,
                self.name,
                Z=Z,
            )

        if optimize:
            if self.fixed_optimization:
                self.fix_model()
                self.optimize_model(
                    optimization_method=optimization_method,
                    tol=tol,
                    n_iters=250,
                )
                self.unfix_model()
                self.optimize_model(
                    optimization_method=optimization_method, tol=tol
                )
            else:
                self.optimize_model(
                    optimization_method=optimization_method, tol=tol
                )

    def fix_model(self):
        self.model.kern.fix()

    def unfix_model(self):
        self.model.kern.constrain_positive()

    def optimize_model(
        self,
        optimization_method="bfgs",
        tol: float = None,
        n_iters: int = None,
    ):
        if self.burst_optimization:
            for _ in range(self.n_restarts):
                (
                    self.model.optimize(
                        optimization_method,
                        messages=1,
                        max_iters=100 if n_iters is None else n_iters,
                        gtol=tol,
                        bfgs_factor=10.0,
                    )
                    if tol is not None
                    else self.model.optimize(
                        optimization_method,
                        messages=1,
                        max_iters=100 if n_iters is None else n_iters,
                        bfgs_factor=10.0,
                    )
                )
        else:
            (
                self.model.optimize(
                    optimization_method,
                    messages=1,
                    max_iters=(
                        self.n_optimization_iters
                        if n_iters is None
                        else n_iters
                    ),
                    bfgs_factor=10.0,
                )
                if tol is not None
                else self.model.optimize(
                    optimization_method,
                    messages=1,
                    max_iters=(
                        self.n_optimization_iters
                        if n_iters is None
                        else n_iters
                    ),
                    bfgs_factor=10.0,
                )
            )

    def get_covariance(
        self,
    ) -> CovarianceOutput:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        if hasattr(self.model.kern, "linear"):
            result_dict["Linear"] = {}
            if self.use_mrd:
                for i, feature in enumerate(self.features):
                    X = self.model.bgplvms[i].X.mean
                    result_dict["Linear"][feature] = self.model.bgplvms[
                        i
                    ].kern.linear.K(X)
            else:
                if isinstance(self.model.X, VariationalPosterior):
                    X = self.model.X.mean
                else:
                    X = self.model.X
                result_dict["Linear"]["All"] = self.model.kern.linear.K(X)
        if hasattr(self.model.kern, "rbf"):
            result_dict["RBF"] = {}
            if self.use_mrd:
                for i, feature in enumerate(self.features):
                    X = self.model.bgplvms[i].X.mean
                    result_dict["RBF"][feature] = self.model.bgplvms[
                        i
                    ].kern.rbf.K(X)
            else:
                if isinstance(self.model.X, VariationalPosterior):
                    X = self.model.X.mean
                else:
                    X = self.model.X
                result_dict["RBF"]["All"] = self.model.kern.rbf.K(X)
        return result_dict

    def get_sensitivity(self) -> SensitivityOutput:
        """Gets the input sensitivity of each latent component."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        if hasattr(self.model.kern, "linear"):
            result_dict["Linear"] = {}
            if self.use_mrd:
                for i, feature in enumerate(self.features):
                    result_dict["Linear"][feature] = self.model.bgplvms[
                        i
                    ].kern.linear.input_sensitivity(summarize=True)
            else:
                result_dict["Linear"]["All"] = (
                    self.model.kern.linear.input_sensitivity(summarize=True)
                )
        if hasattr(self.model.kern, "rbf"):
            result_dict["RBF"] = {}
            if self.use_mrd:
                for i, feature in enumerate(self.features):
                    result_dict["RBF"][feature] = self.model.bgplvms[
                        i
                    ].kern.rbf.input_sensitivity(summarize=True)
            else:
                result_dict["RBF"]["All"] = (
                    self.model.kern.rbf.input_sensitivity(summarize=True)
                )
        return result_dict

    def get_latents(self) -> LatentOutput:
        """Gets the latent space representation of each data point."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."

        if isinstance(self.model.X, VariationalPosterior):
            latents = self.model.X.mean
        else:
            latents = self.model.X
        result_dict = {}
        for i in range(latents.shape[1]):
            result_dict[f"Component_{i}"] = latents[:, i]
        return result_dict

    def get_predictions(self) -> PredictionOutput:
        """Gets the posterior predictions of each feature for each latent dimension."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        if isinstance(self.model.X, VariationalPosterior):
            latents = np.array(self.model.X.mean)
        else:
            latents = np.array(self.model.X)
        average_latents = latents.mean(axis=0)
        # setup tiled latents
        tiled_average_latents = np.tile(average_latents, (20, 1))
        new_latent_data = np.linspace(-5, 5, 20).T
        # predict
        index_type = "Region" if "Region" in self.labels.columns else "Subject"
        results_dict = {}
        for i in range(latents.shape[1]):
            results_dict[i] = {}
            if self.use_mrd:
                ##  if MRD is of feature, predict for each feature independently
                if self.features == DATA_FEATURES:
                    # average latent
                    results_dict[i]["Average"] = {}
                    new_latent = tiled_average_latents.copy()
                    new_latent[:, i] = new_latent_data
                    for j, feat in enumerate(self.features):
                        mean, _ = self.model.predict(new_latent, Yindex=j)
                        results_dict[i]["Average"][feat] = mean
                    # latent per region/subject
                    results_dict[i][index_type] = {}
                    for j, data_point in enumerate(
                        self.labels[index_type].unique()
                    ):
                        results_dict[i][index_type][data_point] = {}
                        new_latent = np.tile(latents[j], (20, 1))
                        new_latent[:, i] = new_latent_data
                        for k, feat in enumerate(self.features):
                            mean, _ = self.model.predict(new_latent, Yindex=k)
                            results_dict[i][index_type][data_point][
                                feat
                            ] = mean
                ##  if MRD is of label/gender, predict for all features together
                else:
                    # average latent
                    results_dict[i]["Average"] = {}
                    new_latent = tiled_average_latents.copy()
                    new_latent[:, i] = new_latent_data
                    for j, feat in enumerate(DATA_FEATURES):
                        results_dict[i]["Average"][feat] = {}
                        for k, label in enumerate(self.features):
                            mean, _ = self.model.predict(new_latent, Yindex=k)
                            feature_means = np.split(
                                mean, len(DATA_FEATURES), axis=1
                            )
                            results_dict[i]["Average"][feat][label] = (
                                feature_means[j]
                            )
                    # latent per region/subject
                    results_dict[i][index_type] = {}
                    for j, data_point in enumerate(
                        self.labels[index_type].unique()
                    ):
                        results_dict[i][index_type][data_point] = {}
                        new_latent = np.tile(latents[j], (20, 1))
                        new_latent[:, i] = new_latent_data
                        for k, feat in enumerate(DATA_FEATURES):
                            results_dict[i][index_type][data_point][feat] = {}
                            for l, label in enumerate(self.features):
                                mean, _ = self.model.predict(
                                    new_latent, Yindex=l
                                )
                                feature_means = np.split(
                                    mean, len(DATA_FEATURES), axis=1
                                )
                                results_dict[i][index_type][data_point][feat][
                                    label
                                ] = feature_means[k]
            else:
                # average latent
                results_dict[i]["Average"] = {}
                new_latent = tiled_average_latents.copy()
                new_latent[:, i] = new_latent_data
                mean, _ = self.model.predict(new_latent)
                feature_means = np.split(mean, len(self.features), axis=1)
                for j, feat in enumerate(self.features):
                    results_dict[i]["Average"][feat] = feature_means[j]
                # latent per region/subject
                results_dict[i][index_type] = {}
                for j, data_point in enumerate(
                    self.labels[index_type].unique()
                ):
                    results_dict[i][index_type][data_point] = {}
                    new_latent = np.tile(latents[j], (20, 1))
                    new_latent[:, i] = new_latent_data
                    mean, _ = self.model.predict(new_latent)
                    feature_means = np.split(mean, len(self.features), axis=1)
                    for k, feat in enumerate(self.features):
                        results_dict[i][index_type][data_point][feat] = (
                            feature_means[k]
                        )
        return results_dict

    def print_model_weights(self) -> None:
        logger.info("Model weights:")
        logger.info(self.model)
        if hasattr(self.model.kern, "rbf"):
            logger.info("RBF lengthscale:")
            logger.info(self.model.kern.rbf.lengthscale)
        if hasattr(self.model.kern, "linear"):
            logger.info("Linear variance:")
            logger.info(self.model.kern.linear.variances)

    def plot_scales(self) -> None:
        import matplotlib.pyplot as plt

        self.model.kern.plot_ARD()
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{self.get_name()}_ard_{get_time_identifier()}.png"
        )
        plt.close()

    def plot_data(self) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(self.Y.shape[1], 1, figsize=(20, 20))
        for i, ax in enumerate(axes):
            ax.plot(self.Y[:, i])
        fig.savefig(
            f"{RESULTS_PATH}/nonlinear/{self.get_name()}_data_{get_time_identifier()}.png"
        )
        plt.close()

    def plot_latent(self) -> None:
        import matplotlib.pyplot as plt

        self.model.plot_latent()
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{self.get_name()}_latent_{get_time_identifier()}.png"
        )
        plt.close()

    def get_name(self) -> str:
        return f"{self.name}_npca-{self.remove_components}_dp-{self.data_processing}_li-{self.latent_initialization}_ii-{self.induced_initialization}-{self.n_inducing}_md-{self.multi_dimensional}_mrd-{self.use_mrd}_opt-{self.n_optimization_iters}-{self.n_restarts}-f{self.fixed_optimization}-b{self.burst_optimization}"

    def save_model(self):
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        save_location = f"{MODELS_PATH}/{self.get_name()}.npy"
        logger.info(f"Saving GP model to {save_location}...")
        np.save(save_location, self.model.param_array)

    def load_model_weights(self):
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        load_location = f"{MODELS_PATH}/{self.get_name()}.npy"
        assert os.path.isfile(
            load_location
        ), f"Model weights not found at {load_location}."
        # load numpy weights from file
        self.model.update_model(False)
        self.model.initialize_parameter()
        self.model[:] = np.load(load_location)
        self.model.update_model(True)
