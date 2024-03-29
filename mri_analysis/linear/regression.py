"""Computes linear regression for a dataset and returns it in an interactible form."""

from typing import List
import pandas as pd
import scipy

from mri_analysis.datatypes import RegressionOutput


class RegressionAnalysis:
    data: pd.DataFrame = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fits the model to some data. Required before calling any other methods."""
        self.data = data

    def get_regressions(self) -> List[RegressionOutput]:
        """Gets the estimated covariance matrix for the model."""
        assert (
            self.data is not None
        ), f"PCA needs to be fitted with data beforehand by calling <fit>."
        results = []
        numerical_columns = self.data.select_dtypes(include="number").columns
        for feat_1 in numerical_columns:
            for feat_2 in set(numerical_columns) - set([feat_1]):
                # calculate overall regressions
                slope, intercept, r, _, _ = scipy.stats.linregress(
                    x=self.data[feat_1], y=self.data[feat_2]
                )
                # calculate per-region regressions
                slopes = []
                intercepts = []
                r_squares = []
                names = []
                for name, group in self.data.groupby("Region"):
                    group_slope, group_intercept, group_r, _, _ = (
                        scipy.stats.linregress(
                            x=group[feat_1], y=group[feat_2]
                        )
                    )
                    slopes.append(group_slope)
                    intercepts.append(group_intercept)
                    r_squares.append(group_r**2)
                    names.append(name)
                results.append(
                    RegressionOutput(
                        feature_1=feat_1,
                        feature_2=feat_2,
                        slope=slope,
                        intercept=intercept,
                        r_squared=r**2,
                        region_slopes=slopes,
                        region_intercepts=intercepts,
                        region_r_squared=r_squares,
                        region_names=names,
                    )
                )
        return results
