from sklearn.cluster import AgglomerativeClustering, HDBSCAN, BisectingKMeans
from sklearn.decomposition import PCA


def agglomerative_clustering(
    df,
    feature_names,
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

    clusters = AgglomerativeClustering(
        n_clusters, metric=metric, linkage=linkage, connectivity=connectivity
    ).fit_predict(df[feature_names].to_numpy())
    df["Cluster"] = clusters
    return df


# Other clustering methods:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans


def principle_component_analysis(df, column_names):
    components = PCA(n_components=2).fit_transform(df[column_names].to_numpy())
    df["Component1"] = components[:, 0]
    df["Component2"] = components[:, 1]
    return df
