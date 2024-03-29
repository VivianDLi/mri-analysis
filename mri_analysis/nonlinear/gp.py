"""Fits a nonlinear GP to a dataset and returns analysis results in an interactible form."""

import os
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from GPy.util.initialization import initialize_latent
from GPy.core import GP
from GPy.kern import Kern
from GPy.models import GPLVM, SparseGPLVM, BayesianGPLVM

from mri_analysis.datatypes import (
    GPType,
    CovarianceOutput,
    LatentOutput,
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
    name: str,
    Z: np.ndarray = None,
) -> GP:
    match gp_type:
        case "GP":
            return GPLVM(data, n_components, X=X, kernel=kernel, name=name)
        case "Sparse":
            return SparseGPLVM(
                data,
                n_components,
                X=X,
                kernel=kernel,
                num_inducing=n_inducing,
                name=name,
            )
        case "Bayesian":
            return BayesianGPLVM(
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
            return GPLVM


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
        self.kernels = kernels
        self.gp_type = gp_type
        self.expand_dims = expand_dims
        self.name = name
        # gp parameters
        self.n_inducing = n_inducing
        self.likelihood_variance = likelihood_variance

    def fit(
        self, data: pd.DataFrame, n_components: int, optimize: bool = True
    ) -> None:
        """Fits the model to some data. Required before calling any other methods."""
        # set name based on parameters
        if self.name is None:
            self.name = f"{self.gp_type}_{self.kernel.name}_{n_components}"
        self.data = data
        # filter out numerical features only
        numerical_columns = self.data.select_dtypes(include="number").columns
        non_number_columns = set(data.columns) - set(numerical_columns)
        self.features = numerical_columns
        self.labels = data[non_number_columns]
        assert len(self.features) >= 1, "No numerical features found in data."
        # expand dimensions if enabled
        numerical_data = self.data[numerical_columns].to_numpy()
        if self.expand_dims:
            numerical_data = np.matmul(
                numerical_data,
                np.random.normal(0, 1e-6, size=(numerical_data.shape[1], 100)),
            )
        # initialize latent space
        X = initialize_latent("PCA", n_components, numerical_data)
        # initialize kernel list (copy if only one kernel)
        if isinstance(self.kernels, Kern):
            self.kernels = [
                self.kernels.copy() for _ in range(len(numerical_columns))
            ]
        assert len(self.kernels) == len(
            numerical_columns
        ), f"Number of given kernels ({len(self.kernels)}) must match the number of numerical features ({len(numerical_columns)})."
        # initialize base GP (to optimize)
        self.gps.append(
            get_gp_from_type(
                self.gp_type,
                numerical_data[:, 0],
                X,
                n_components,
                self.n_inducing,
                self.kernels[0],
                self.name + f"_{self.features[0]}",
            )
        )
        # check for inducing points
        Z = None
        if hasattr(self.gps[0], "Z"):
            Z = self.gps[0].Z
        # initialize other GPs if multi-dimensional
        if len(numerical_columns) > 1:
            for i, feature in self.features[1:]:
                self.gps.append(
                    get_gp_from_type(
                        self.gp_type,
                        numerical_data[:, i],
                        X,
                        n_components,
                        self.n_inducing,
                        self.kernels[i],
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
        result_dict
        return result_dict

    def get_latents(self) -> LatentOutput:
        """Gets the latent space representation of each data point."""
        assert (
            self.data is not None and self.gp is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."

        latents = self.gp.X
        result_dict = {}
        for i in range(latents.shape[1]):
            result_dict[f"Component_{i}"] = latents[:, i]
        return result_dict

    def save_model(self):
        assert (
            self.data is not None and self.gp is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        save_location = f"{MODELS_PATH}/{self.name}.npy"
        logger.info(f"Saving GP model to {save_location}...")
        np.save(save_location, self.gp.param_array)

    def load_model_weights(self):
        assert (
            self.data is not None and self.gp is not None
        ), f"GP needs to be fitted with data beforehand by calling <fit>."
        load_location = f"{MODELS_PATH}/{self.name}.npy"
        assert os.path.isfile(
            load_location
        ), f"Model weights not found at {load_location}."
        # load numpy weights from file
        self.gp.update_model(False)
        self.gp.initialize_parameter()
        self.gp[:] = np.load(load_location)
        self.gp.update_model(True)
