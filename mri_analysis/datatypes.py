"""Python module for defining custom data types."""

## Literals

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, NewType, Tuple, TypedDict

import numpy as np
import pandas as pd
from scipy.stats import zscore, median_abs_deviation

from loguru import logger


## Dataset Types
DATA_COLUMNS = Literal[
    "Subject",
    "Region",
    "CT",
    "SD",
    "MD",
    "ICVF",
    "Label",
    "Age",
    "Brain Volume",
]
DATA_FEATURES = ["CT", "SD", "MD", "ICVF"]
NormalizeType = Literal["zscore", "median"]


@dataclass
class Dataset:
    data: pd.DataFrame
    average_feature: str = "Region"
    sort_order: List[str] = field(
        default_factory=lambda: ["Feature", "Labels", "Region", "Subject"]
    )
    normalize_type: NormalizeType = "zscore"

    def get_data(
        self,
        subset: int = None,
        average: bool = True,
        normalize: bool = True,
        remove_outliers: bool = False,
        flatten: bool = False,
    ) -> pd.DataFrame:
        assert (
            "Region" in self.data.columns
        ), "Region column not found in dataset."
        data = self.data.copy()
        sort_order = self.sort_order.copy()
        if subset is not None:
            logger.info(f"Subsetting data to {subset} random regions...")
            region_list = data["Region"].unique()
            data = data[
                data["Region"].isin(
                    np.random.choice(region_list, size=subset, replace=False)
                )
            ]
        if remove_outliers:
            logger.info(f"Removing outliers from dataset...")
            data = self._remove_outliers(data)
        if normalize:
            logger.info(f"Normalizing data...")
            data = self._normalize(data)
        if average:
            logger.info(f"Averaging data across {self.average_feature}...")
            non_numeric_columns = set(
                data.select_dtypes(exclude="number").columns
            ) - set([self.average_feature])
            data = self._average(data)
            # remove missing columns from sort_order
            to_remove = [
                col
                for col in sort_order
                if col in non_numeric_columns and col != "Feature"
            ]
            logger.debug(
                f"Removing non-numeric columns: {to_remove} from sort order."
            )
            for col in to_remove:
                sort_order.remove(col)
        if flatten:
            logger.info(f"Flattening data...")
            data = pd.melt(
                data,
                id_vars=["Region", "Labels", "Subject"],
                value_vars=DATA_FEATURES,
                var_name="Feature",
                value_name="Value",
            )
        elif "Feature" in sort_order:
            sort_order.remove("Feature")
        logger.info(f"Sorting data by {sort_order}...")
        data = data.sort_values(by=sort_order)
        return data

    def _average(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns the same dataframe averaging all data across a certain feature"""
        assert (
            self.average_feature in data.columns
        ), f"Average feature: {self.average_feature} not in dataset columns: {data.columns}."
        number_columns = data.select_dtypes(include="number").columns
        non_number_columns = list(data.select_dtypes(exclude="number").columns)
        # remove average feature from non_number_columns
        if self.average_feature in non_number_columns:
            non_number_columns.remove(self.average_feature)
        average_df = (
            data.groupby(self.average_feature)[number_columns]
            .mean()
            .reset_index()
        )
        label_df = (
            data.groupby(self.average_feature)[non_number_columns]
            .first()
            .reset_index()
        )
        df = average_df.merge(label_df, on=self.average_feature, how="left")

        logger.info(
            f"Averaging columns: {number_columns} across {self.average_feature}..."
        )
        logger.debug(f"Resulting dataframe as columns: {average_df.columns}.")
        return df

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns the same dataframe normalizing feature column values (CT, SD, MD, ICVF) with their z-scores."""
        assert all(
            feat in data.columns for feat in DATA_FEATURES
        ), f"Features {DATA_FEATURES} not found in dataset columns: {data.columns}."
        match self.normalize_type:
            case "zscore":
                logger.info("Normalizing data using z-scores...")
                # Uses a normal z-score using the mean and variance
                for feature in DATA_FEATURES:
                    data[feature] = zscore(data[feature])
            case "median":
                logger.info(
                    "Normalizing data using median absolute deviation (MAD)..."
                )
                # Uses the modified z-score from (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) using medians
                for feature in DATA_FEATURES:
                    data[feature] = (
                        0.6745
                        * (data[feature] - np.median(data[feature]))
                        / median_abs_deviation(data[feature])
                    )
            case _:
                logger.warning(
                    f"Unrecognized normalization algorithm: {self.normalize_type}. Using z-score instead."
                )
                for feature in DATA_FEATURES:
                    data[feature] = zscore(data[feature])
        return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds an additional two outlier columns to mark population-wide outliers per feature"""
        assert all(
            feat in data.columns for feat in DATA_FEATURES
        ), f"Features {DATA_FEATURES} not found in dataset columns: {data.columns}."
        # find outliers for the population-wide feature distribution
        data = self.normalize(data)
        data = data[
            data["CT"].abs() > 3
            and data["SD"].abs() > 3
            and data["MD"].abs() > 3
            and data["ICVF"].abs() > 3
        ]
        return data

    def set_average_feature(self, average_feature: str) -> None:
        if average_feature not in self.data.columns:
            logger.warning(
                f"Feature {average_feature} is not in the dataset. Ignoring new average feature."
            )
            return
        self.average_feature = average_feature

    def set_sort_order(self, sort_order: List[str]) -> None:
        if not set(sort_order) - set(["Feature"]) <= set(self.data.columns):
            logger.warning(
                f"Features {set(sort_order) - set(self.data.columns)} are not in the dataset. Ignoring new sort order."
            )
            return
        self.sort_order = sort_order


## Latent Types
CovarianceOutput = NewType("CovarianceOutput", np.ndarray)
ComponentOutput = NewType("ComponentOutput", Dict[str, np.array])
LatentOutput = NewType("LatentOutput", Dict[str, np.array])
SensitivityOutput = NewType("SensitivityOutput", np.array)


class ExplainedVarianceOutput(TypedDict):
    variances: np.array
    total_variance: float


class RegressionOutput(TypedDict):
    feat_1: str
    feat_2: str
    slope: float
    intercept: float
    r_squared: float
    region_slopes: List[float]
    region_intercepts: List[float]
    region_r_squared: List[float]
    region_names: List[str]


GPType = Literal["Base", "Bayesian"]


## Plotting Types
PlotConfig = NewType("PlotConfig", Dict[str, Any])


DistributionPlotType = Literal[
    "feature_histogram",
    "regression_histogram",
    "feature_scatter",
    "feature_regression",
    "feature_strip",
]

ClusterPlotType = Literal["cluster_scatter", "cluster_map"]

LinearPlotType = Literal[
    "pca_covariance", "pca_variance", "pca_eigenvectors", "pca_latents"
]

NonlinearPlotType = Literal["gp_covariance", "gp_sensitivity", "gp_latents"]

BrainPlotType = Literal["brain_feature", "brain_regression"]
