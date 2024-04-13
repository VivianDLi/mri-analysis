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

from mri_analysis.datatypes import (
    DATA_FEATURES,
    ExplainedVarianceOutput,
    GPType,
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
        expand_dims: bool = True,
        multi_dimensional: bool = False,
        name: str = None,
        n_inducing: int = 10,
        likelihood_variance: float = None,
    ):
        self.kernels = kernels if type(kernels) == list else [kernels]
        self.gp_type = gp_type
        self.expand_dims = expand_dims
        self.multi_dimensional = multi_dimensional
        self.name = name
        # gp parameters
        self.n_inducing = n_inducing
        self.likelihood_variance = likelihood_variance

    def fit(
        self,
        data: pd.DataFrame,
        n_components: int,
        optimize: bool = True,
        flat_data: bool = False,
    ) -> None:
        """Fits the model to some data. Required before calling any other methods."""
        from paramz import ObsAr  # local paramz import for setting up model

        # set name based on parameters
        if self.name is None:
            self.name = f"{self.gp_type}_{'flat' if flat_data else 'md'}_{[kern.name for kern in self.kernels]}_{n_components}"
        else:
            self.name = f"{self.name}_{'flat' if flat_data else 'md'}"
        self.data = data
        ## setup data
        # filter out numerical features only
        numerical_columns = self.data.select_dtypes(include="number").columns
        non_number_columns = set(data.columns) - set(numerical_columns)
        if flat_data:
            self.features = ["All"]
        else:
            self.features = list(set(numerical_columns) & set(DATA_FEATURES))
        self.labels = data[list(non_number_columns)]
        assert len(self.features) >= 1, "No numerical features found in data."
        numerical_data = self.data[self.features].to_numpy()
        # setup YList
        if self.multi_dimensional:
            # single element YList
            Y_list = [numerical_data]
            self.features = ["All"]
        else:
            # YList per feature
            Y_list = [
                numerical_data[:, [i]] for i in range(numerical_data.shape[1])
            ]
        # expand dimensions if enabled
        if self.expand_dims:
            # multiply each array by high-dimensional random noise
            Y_list = [
                np.matmul(Y, np.random.randn(Y.shape[1], 100))
                + 1e-6 * np.random.randn(Y.shape[0], 100)
                for Y in Y_list
            ]
        # set YList observers
        Y_list = [ObsAr(Y) for Y in Y_list]
        Y = Y_list[-1]

        ## setup GP model
        # initialize latent space
        X, _ = initialize_latent("PCA", n_components, np.hstack(Y_list))
        Z = np.random.permutation(X.copy())[: self.n_inducing]
        # initialize kernel list (copy if only one kernel)
        while len(self.kernels) < len(numerical_columns):
            self.kernels.append(self.kernels[0].copy())
        assert len(self.kernels) == len(
            numerical_columns
        ), f"Number of given kernels ({len(self.kernels)}) must match the number of numerical features ({len(numerical_columns)})."
        # initialize likelihoods
        if self.likelihood_variance is None:
            # set noise as 1% of variance in data
            likelihood_variances = np.var(numerical_data, axis=0) * 0.01
            self.likelihoods = [
                Gaussian(variance=var) for var in likelihood_variances
            ]
        else:
            self.likelihoods = [
                Gaussian(variance=self.likelihood_variance)
                for _ in len(self.features)
            ]
        # initialize base GP (to optimize)
        self.gp = get_gp_from_type(
            self.gp_type,
            Y,
            X,
            n_components,
            self.n_inducing,
            None,
            Gaussian(),
            self.name,
            Z=Z,
        )
        self._log_marginal_likelihood = 0
        self.gp.unlink_parameter(self.gp.likelihood)
        self.gp.unlink_parameter(self.gp.kern)
        # initialize other GPs
        self.gps = []
        for i, Y in enumerate(Y_list):
            new_gp = get_gp_from_type(
                self.gp_type,
                Y,
                X,
                n_components,
                self.n_inducing,
                self.kernels[i],
                self.likelihoods[i],
                self.name + f"_{self.features[i]}",
                Z=Z,
            )
            new_gp.unlink_parameter(new_gp.X)
            del new_gp.X
            new_gp.X = self.gp.X
            if hasattr(new_gp, "Z"):
                new_gp.unlink_parameter(new_gp.Z)
                del new_gp.Z
                new_gp.Z = self.gp.Z
                self.gp.link_parameter(new_gp, i + 2)
            else:
                self.gp.link_parameter(new_gp, i + 1)
            self.gps.append(new_gp)
        # update base gp
        b = self.gps[0]
        self.gp.posterior = b.posterior
        self.gp.kern = b.kern
        self.gp.likelihood = b.likelihood

        if optimize:
            self.gp.optimize("bfgs", messages=1, max_iters=5e4, gtol=0.1)

    def get_covariance(
        self,
    ) -> Dict[str, CovarianceOutput]:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        for i, feature in enumerate(self.features):
            if isinstance(self.gps[i].X, VariationalPosterior):
                X = self.gps[i].X.mean
            else:
                X = self.gps[i].X
            result_dict[feature] = self.gps[i].kern.K(X)
        return result_dict

    def get_sensitivity(self) -> Dict[str, SensitivityOutput]:
        """Gets the input sensitivity of each latent component."""
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        for i, feature in enumerate(self.features):
            result_dict[feature] = self.gps[i].kern.input_sensitivity(
                summarize=False
            )
        return result_dict

    def get_latents(self) -> LatentOutput:
        """Gets the latent space representation of each data point."""
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."

        if isinstance(self.gp.X, VariationalPosterior):
            latents = self.gp.X.mean
        else:
            latents = self.gp.X
        result_dict = {}
        for i in range(latents.shape[1]):
            result_dict[f"Component_{i}"] = latents[:, i]
        return result_dict

    def save_model(self):
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        for i in range(len(self.gps)):
            save_location = f"{MODELS_PATH}/{self.name}_{i}.npy"
            logger.info(f"Saving GP model to {save_location}...")
            np.save(save_location, self.gps[i].param_array)

    def load_model_weights(self):
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        for i in range(len(self.gps)):
            load_location = f"{MODELS_PATH}/{self.name}_{i}.npy"
            assert os.path.isfile(
                load_location
            ), f"Model weights not found at {load_location}."
            # load numpy weights from file
            self.gps[i].update_model(False)
            self.gps[i].initialize_parameter()
            self.gps[i][:] = np.load(load_location)
            self.gps[i].update_model(True)
