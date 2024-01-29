from sklearn.cluster import AgglomerativeClustering, HDBSCAN, BisectingKMeans
import pandas as pd

import os
from datetime import datetime


def agglomerative_clustering(
    df,
    feature_names,
    remove_outliers: bool = True,
    average: bool = False,
    n_clusters=5,
    metric="euclidean",
    linkage="ward",
    connectivity="none",
):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    match connectivity:
        case _:
            connectivity = None
    assert len(feature_names) == 4
    if remove_outliers:
        assert (
            "Population CT Outliers" in df.columns
            and "Population SD Outliers" in df.columns
            and "Population MD Outliers" in df.columns
            and "Population ICVF Outliers" in df.columns
        )
        df = df[~df["Population CT Outliers"]]
        df = df[~df["Population SD Outliers"]]
        df = df[~df["Population MD Outliers"]]
        df = df[~df["Population ICVF Outliers"]]
    if average:
        df = df.groupby(by=["Region"])[feature_names].mean()

    clusters = AgglomerativeClustering(
        n_clusters, metric=metric, linkage=linkage, connectivity=connectivity
    ).fit_predict(df[feature_names].to_numpy())
    df["Cluster"] = clusters
    return df


# Other clustering methods:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
