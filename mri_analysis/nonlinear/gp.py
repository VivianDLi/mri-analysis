"""Fits a nonlinear GP to a dataset and returns analysis results in an interactible form."""

import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from GPy.util.initialization import initialize_latent
from GPy.core.parameterization.variational import VariationalPosterior
from GPy.core import GP
from GPy.kern import Kern
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
    SensitivityOutput,
)
from mri_analysis.constants import MODELS_PATH

from loguru import logger


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
        name: str = None,
        n_inducing: int = 25,
        n_restarts: int = 10,
        burst_optimization: bool = False,
        n_optimization_iters: int = 5e4,
        use_mrd: bool = True,
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
        self.burst_optimization = burst_optimization
        self.n_optimization_iters = n_optimization_iters
        self.use_mrd = use_mrd

    def fit(
        self,
        data: pd.DataFrame,
        n_components: int,
        features: List[str] = DATA_FEATURES,
        optimize: bool = True,
        flat_data: bool = False,
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
                level=1, axis="columns"
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
                    Y.dot(np.random.randn(Y.shape[1], 100))
                    + 1e-6 * np.random.randn(Y.shape[0], 100)
                    for Y in Y_list
                ]
                Y_list = [Y - Y.mean(0) for Y in Y_list]
                Y_list = [Y / Y.std(0) for Y in Y_list]
            case _:
                logger.warning(
                    "Data processing type not recognized. Using none."
                )
        # initialize latent space
        Y = np.hstack(Y_list)
        match self.latent_initialization:
            case "pca":
                X, _ = initialize_latent("PCA", n_components, Y)
            case "single":
                pass
            case "random":
                X = np.random.randn(Y.shape[0], n_components)
            case "data":
                X = Y.copy() * np.random.randn(Y.shape[1], n_components)
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
            self.fix_model()
            self.optimize_model(gtol=0.5)
            self.unfix_model()
            self.optimize_model(gtol=0.5)

    def fix_model(self):
        self.model.kern.fix()

    def unfix_model(self):
        self.model.unfix()
        self.model.kern.constrain_positive()

    def optimize_model(self, gtol: float = None):
        if self.burst_optimization:
            for _ in range(self.n_restarts):
                (
                    self.model.optimize(
                        "bfgs",
                        messages=1,
                        max_iters=100,
                        gtol=gtol,
                        clear_after_finish=True,
                    )
                    if gtol is not None
                    else self.model.optimize(
                        "bfgs",
                        messages=1,
                        max_iters=100,
                        clear_after_finish=True,
                    )
                )
        else:
            (
                self.model.optimize(
                    "bfgs",
                    messages=1,
                    max_iters=self.n_optimization_iters,
                    gtol=gtol,
                )
                if gtol is not None
                else self.model.optimize(
                    "bfgs",
                    messages=1,
                    max_iters=self.n_optimization_iters,
                )
            )

    def get_covariance(
        self,
    ) -> Dict[str, CovarianceOutput]:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        if self.use_mrd:
            for i, feature in enumerate(self.features):
                X = self.model.bgplvms[i].X.mean
                result_dict[feature] = self.model.bgplvms[i].kern.K(X)
        else:
            if isinstance(self.model.X, VariationalPosterior):
                X = self.model.X.mean
            else:
                X = self.model.X
            result_dict["All"] = self.model.kern.K(X)
        return result_dict

    def get_sensitivity(self) -> Dict[str, SensitivityOutput]:
        """Gets the input sensitivity of each latent component."""
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        if self.use_mrd:
            for i, feature in enumerate(self.features):
                result_dict[feature] = self.model.bgplvms[
                    i
                ].kern.input_sensitivity(summarize=False)[0, :]
        else:
            result_dict["All"] = self.model.kern.input_sensitivity(
                summarize=False
            )[0, :]
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

    def save_model(self):
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        save_location = f"{MODELS_PATH}/{self.name}.npy"
        logger.info(f"Saving GP model to {save_location}...")
        np.save(save_location, self.model.param_array)

    def load_model_weights(self):
        assert (
            self.data is not None and self.model is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        load_location = f"{MODELS_PATH}/{self.name}.npy"
        assert os.path.isfile(
            load_location
        ), f"Model weights not found at {load_location}."
        # load numpy weights from file
        self.model.update_model(False)
        self.model.initialize_parameter()
        self.model[:] = np.load(load_location)
        self.model.update_model(True)
