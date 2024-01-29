from scipy.stats import zscore, median_abs_deviation
import numpy as np


def get_outliers(df, use_modified: bool = False, outlier_cutoff: float = 3):
    """Adds an additional two outlier columns to mark subject-wide and population-wide outliers per feature"""

    def map_subject_outliers(x):
        norm_x = normalize(x, use_modified)
        x["Subject CT Outliers"] = norm_x["CT_norm"].abs() > outlier_cutoff
        x["Subject SD Outliers"] = norm_x["SD_norm"].abs() > outlier_cutoff
        x["Subject MD Outliers"] = norm_x["MD_norm"].abs() > outlier_cutoff
        x["Subject ICVF Outliers"] = norm_x["ICVF_norm"].abs() > outlier_cutoff

    feature_cols = ["CT", "SD", "MD", "ICVF"]
    assert all([col in df.columns for col in feature_cols])
    # find outliers for the population-wide feature distribution
    norm_df = normalize(df, use_modified)
    df["Population CT Outliers"] = norm_df["CT_norm"].abs() > outlier_cutoff
    df["Population SD Outliers"] = norm_df["SD_norm"].abs() > outlier_cutoff
    df["Population MD Outliers"] = norm_df["MD_norm"].abs() > outlier_cutoff
    df["Population ICVF Outliers"] = (
        norm_df["ICVF_norm"].abs() > outlier_cutoff
    )
    # find outliers for the subject-wide feature distribution
    df.groupby(by=["Subject"], group_keys=False).apply(map_subject_outliers)
    return df


def normalize(df, use_modified: bool = False):
    """Adds feature column values (CT, SD, MD, ICVF) with their z-scores."""
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
