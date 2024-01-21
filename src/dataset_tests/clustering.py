from sklearn.cluster import AgglomerativeClustering, HDBSCAN, BisectingKMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from datetime import datetime
from constants import DATASET_FOLDER


def agglomerative_clustering(
    file, metric="euclidean", linkage="ward", connectivity="none"
):
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    file_name = f"{os.path.basename(file)}-agglomerative-{metric}-{linkage}-{connectivity}-{datetime.now().strftime('%m-%d')}"
    match connectivity:
        case _:
            connectivity = None
    df = pd.read_csv(file, header=0)[:180]  # only take the left hemisphere
    feature_names = df.columns[1:]
    assert len(feature_names) == 4

    n_clusters = (
        5  # vision, hearing, sensory/motor, task positive, task negative
    )
    clusters = AgglomerativeClustering(
        n_clusters,
        metric=metric,
        linkage=linkage,
        connectivity=connectivity,
        memory=f"./src/dataset_tests/memory/{file_name}",
    ).fit_predict(df[feature_names].to_numpy())
    df["cluster"] = clusters

    # graph all six possible combinations of 4-features
    fig, axes = plt.subplots(2, 3, figsize=(12, 15))
    for i in range(3):
        for j in range(i + 1, 4):
            ax = (
                axes[i][j - 1]
                if i == 0
                else axes[i][j - 2]
                if i == 1
                else axes[1][2]
            )
            sns.scatterplot(
                df,
                x=feature_names[i],
                y=feature_names[j],
                hue="cluster",
                ax=ax,
            )
    fig.suptitle(file_name)
    fig.savefig(f"./src/dataset_tests/results/{file_name}.png")
    plt.close()


# Other clustering methods:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
