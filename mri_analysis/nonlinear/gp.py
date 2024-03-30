"""Fits a nonlinear GP to a dataset and returns analysis results in an interactible form."""

import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from GPy.util.initialization import initialize_latent
from GPy.core import GP
from GPy.kern import Kern
from GPy.likelihoods import Likelihood, Gaussian
from GPy.models import GPLVM, SparseGPLVM, BayesianGPLVM

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
        case "Sparse":
            gp = SparseGPLVM(
                data,
                n_components,
                X=X,
                kernel=kernel,
                num_inducing=n_inducing,
                name=name,
            )
        case "Bayesian":
            gp = BayesianGPLVM(
                data,
                n_components,
                X=X,
                Z=Z,
                kernel=kernel,
                num_inducing=n_inducing,
                name=name,
            )
        case _:
            logger.warning(
                f"Unrecognized GP type: {gp_type}. Using GPLVM instead."
            )
            gp = GPLVM(data, n_components, X=X, kernel=kernel, name=name)
    gp.likelihood = likelihood
    return gp


class GPAnalysis:
    data: pd.DataFrame = None
    gps: List[GP] = None

    def __init__(
        self,
        kernels: Union[Kern, List[Kern]],
        gp_type: GPType = "GP",
        expand_dims: bool = False,
        name: str = None,
        n_inducing: int = 10,
        likelihood_variance: float = 0.1,
    ):
        self.kernels = kernels if type(kernels) == list else [kernels]
        self.gp_type = gp_type
        self.expand_dims = expand_dims
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
        # set name based on parameters
        if self.name is None:
            self.name = f"{self.gp_type}_{'flat' if flat_data else 'md'}_{[kern.name for kern in self.kernels]}_{n_components}"
        else:
            self.name = f"{self.name}_{'flat' if flat_data else 'md'}"
        self.data = data
        # filter out numerical features only
        numerical_columns = self.data.select_dtypes(include="number").columns
        non_number_columns = set(data.columns) - set(numerical_columns)
        if flat_data:
            self.features = ["Value"]
        else:
            self.features = list(set(numerical_columns) & set(DATA_FEATURES))
        self.labels = data[list(non_number_columns)]
        assert len(self.features) >= 1, "No numerical features found in data."
        # expand dimensions if enabled
        numerical_data = self.data[self.features].to_numpy()
        if self.expand_dims:
            numerical_data = np.matmul(
                numerical_data,
                np.random.normal(0, 1e-6, size=(numerical_data.shape[1], 100)),
            )
        # initialize latent space
        X, _ = initialize_latent("PCA", n_components, numerical_data)
        # initialize kernel list (copy if only one kernel)
        while len(self.kernels) < len(numerical_columns):
            self.kernels.append(self.kernels[0].copy())
        assert len(self.kernels) == len(
            numerical_columns
        ), f"Number of given kernels ({len(self.kernels)}) must match the number of numerical features ({len(numerical_columns)})."
        # initialize likelihoods
        # set noise as 1% of variance in data
        likelihood_variances = np.var(numerical_data, axis=0) * 0.01
        self.likelihoods = [
            Gaussian(variance=var) for var in likelihood_variances
        ]
        # initialize base GP (to optimize)
        self.gps = []
        self.gps.append(
            get_gp_from_type(
                self.gp_type,
                numerical_data[:, [0]],
                X,
                n_components,
                self.n_inducing,
                self.kernels[0],
                self.likelihoods[0],
                self.name + f"_{self.features[0]}",
            )
        )
        # check for inducing points
        Z = None
        if hasattr(self.gps[0], "Z"):
            Z = self.gps[0].Z
        # initialize other GPs if multi-dimensional
        if len(numerical_columns) > 1:
            for i, feature in enumerate(self.features[1:]):
                self.gps.append(
                    get_gp_from_type(
                        self.gp_type,
                        numerical_data[:, [i]],
                        X,
                        n_components,
                        self.n_inducing,
                        self.kernels[i],
                        self.likelihoods[i],
                        self.name + f"_{feature}",
                        Z=Z,
                    )
                )
        if optimize:
            self.gps[0].optimize(messages=1, max_iters=5e4)

    def get_covariance(
        self,
    ) -> Dict[str, CovarianceOutput]:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None and self.gps is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        result_dict = {}
        for i, feature in enumerate(self.features):
            result_dict[feature] = self.gps[i].kern.K(self.gps[i].X)
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

        latents = self.gps[0].X
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
