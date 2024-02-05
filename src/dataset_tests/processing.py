import pandas as pd
import scipy
from scipy.stats import zscore, median_abs_deviation
import numpy as np


def get_outliers(df, use_modified: bool = False, outlier_cutoff: float = 3):
    """Adds an additional two outlier columns to mark population-wide outliers per feature"""

    feature_cols = ["CT", "SD", "MD", "ICVF"]
    assert all([col in df.columns for col in feature_cols])
    # find outliers for the population-wide feature distribution
    norm_df = normalize_data(df, use_modified)
    df["Population CT Outliers"] = norm_df["CT_norm"].abs() > outlier_cutoff
    df["Population SD Outliers"] = norm_df["SD_norm"].abs() > outlier_cutoff
    df["Population MD Outliers"] = norm_df["MD_norm"].abs() > outlier_cutoff
    df["Population ICVF Outliers"] = (
        norm_df["ICVF_norm"].abs() > outlier_cutoff
    )

    return df


def get_regressions(df, combinations, group_name: str = "Subject"):
    """Calculates regression slopes per subject for every possible feature pair"""
    grouped = df.groupby(group_name)
    new_dict = {}
    names = []
    for f1, f2 in combinations:
        names.append(f"{f1}_{f2}_slope")
        names.append(f"{f1}_{f2}_intercept")
        names.append(f"{f1}_{f2}_r^2")
    names.append("Age")
    names.append("Brain Volume")
    for name, group in grouped:
        values = []
        for f1, f2 in combinations:
            # calculate slope, intercept, and r^2 of regression equation
            slope, intercept, r, _, _ = scipy.stats.linregress(
                x=group[f1], y=group[f2]
            )
            values.append(slope)
            values.append(intercept)
            values.append(r**2)
        values.append(group["Age"].to_numpy()[0])
        values.append(group["Brain Volume"].to_numpy()[0])
        new_dict[name] = values
    return pd.DataFrame.from_dict(
        new_dict, orient="index", columns=names
    ).reset_index(names="Subject")


def average_across(df, group: str):
    """Returns a new df averaging all columns across subjects"""
    number_columns = df.select_dtypes(include="number").columns
    return df.groupby(group)[number_columns].mean()


def normalize_data(df, use_modified: bool = False):
    """Adds feature column values (CT_norm, SD_norm, MD_norm, ICVF_norm) with their z-scores."""
    if use_modified:
        # Uses the modified z-score from (https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) using medians
        df["CT_norm"] = (
            0.6745
            * (df["CT"] - np.median(df["CT"]))
            / median_abs_deviation(df["CT"])
        )
        df["SD_norm"] = (
            0.6745
            * (df["SD"] - np.median(df["SD"]))
            / median_abs_deviation(df["SD"])
        )
        df["MD_norm"] = (
            0.6745
            * (df["MD"] - np.median(df["MD"]))
            / median_abs_deviation(df["MD"])
        )
        df["ICVF_norm"] = (
            0.6745
            * (df["ICVF"] - np.median(df["ICVF"]))
            / median_abs_deviation(df["ICVF"])
        )
    else:
        # Uses a normal z-score using the mean and variance
        df["CT_norm"] = zscore(df["CT"])
        df["SD_norm"] = zscore(df["SD"])
        df["MD_norm"] = zscore(df["MD"])
        df["ICVF_norm"] = zscore(df["ICVF"])
    return df
