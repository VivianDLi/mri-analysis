"""Computes PCA for a dataset and returns it in an interactible form."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from mri_analysis.datatypes import (
    DATA_FEATURES,
    ComponentOutput,
    CovarianceOutput,
    ExplainedVarianceOutput,
    LatentOutput,
)


class ComponentAnalysis:
    data: pd.DataFrame = None
    pca: PCA = None

    def fit(self, data: pd.DataFrame, n_components: int) -> None:
        """Fits the model to some data. Required before calling any other methods."""
        self.data = data
        # filter out numerical features only
        numerical_columns = self.data.select_dtypes(include="number").columns
        self.features = set(numerical_columns) & set(DATA_FEATURES)
        self.pca = PCA(n_components=n_components).fit(
            self.data[list(self.features)]
        )

    def get_covariance(self) -> CovarianceOutput:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None and self.pca is not None
        ), f"PCA needs to be fitted with data beforehand by calling <fit>."

        return self.pca.get_covariance()

    def get_explained_variance(self) -> ExplainedVarianceOutput:
        """Gets the explained variance of each component and the total explained variance for the model."""
        assert (
            self.data is not None and self.pca is not None
        ), f"PCA needs to be fitted with data beforehand by calling <fit>."

        return {
            "variances": self.pca.explained_variance_,
            "total_variance": np.sum(self.pca.explained_variance_),
        }

    def get_components(self) -> ComponentOutput:
        """Gets the direction of each component in the original feature space."""
        assert (
            self.data is not None and self.pca is not None
        ), f"PCA needs to be fitted with data beforehand by calling <fit>."

        components = self.pca.components_
        result_dict = {}
        for i in range(components.shape[0]):
            result_dict[f"Component_{i}"] = components[i, :]
        return result_dict

    def get_latents(self) -> LatentOutput:
        """Gets the latent space representation of each data point."""
        assert (
            self.data is not None and self.pca is not None
        ), f"PCA needs to be fitted with data beforehand by calling <fit>."

        latents = self.pca.transform(self.data[list(self.features)])
        result_dict = {}
        for i in range(latents.shape[1]):
            result_dict[f"Component_{i}"] = latents[:, i]
        return result_dict
