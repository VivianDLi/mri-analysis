from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import scipy

import os
import glob
from datetime import datetime

from mri_analysis.visualizations.clustering import agglomerative_clustering
from mri_analysis.nonlinear.latent_methods import (
    principle_component_analysis,
    load_gp_model,
    create_named_gp,
    get_gp_covariance,
)
from mri_analysis.data.processing import *
from constants import DATA_PATH, METADATA_PATH, RESULTS_PATH, BRAINPLOT_PATH


def get_dataset():
    """Loads the dataset as a DataFrame with 7 columns: [brain] Region, CT, SD, MD, ICVF, Subject, Label"""

    def load_csv(file):
        subject_name = os.path.basename(file)[:-4]
        with open("./src/dataset_tests/HCP_von_Economo_labels.txt", "r") as f:
            labels = [line.rstrip() for line in f]
        df = pd.read_csv(file, header=0)
        df.rename(columns={df.columns[0]: "Region"}, inplace=True)
        df["Subject"] = subject_name
        df["Label"] = labels
        df["Label"] = df["Label"].astype(int)
        return df

    df = pd.concat(
        map(load_csv, glob.glob(f"{DATA_PATH}/*.csv")), ignore_index=True
    )
    # merge in demographics information
    demographics = pd.read_csv(
        f"{METADATA_PATH}/demographics.csv", header=0
    ).rename(
        columns={
            "ID": "Subject",
            "Age at visit to assessment centre (Imaging)": "Age",
            "Estimated Total Intracranial Volume": "Brain Volume",
        }
    )
    df = df.merge(demographics, how="left", on="Subject")
    return df


def get_gp_dataset(
    subset: int = 10,
    normalize: bool = True,
    average: bool = False,
    md: bool = False,
):
    """Loads the dataset as a DataFrame with 4 sorted columns: [brain] Region, Subject, Feature, Value"""
    df = get_dataset()
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if md:
        df = df.set_index(["Region", "Subject"]).sort_index()
    else:
        df = pd.melt(
            df,
            id_vars=["Region", "Subject"],
            value_vars=feature_names,
            var_name="Feature",
            value_name="Value",
        )
        df = df.set_index(["Region", "Feature", "Subject"]).sort_index()
    if subset is not None:
        indices = np.random.choice(
            len(df.index.get_level_values(0).unique()),
            size=subset,
            replace=False,
        )
        df = df.loc[
            df.index.get_level_values(0).unique()[indices].to_list(), :, :
        ]
    if average:
        df = average_across(df, "Region" if md else ["Region", "Feature"])
    return df


## Plot Linear Relationships


def plot_histograms(
    df, x_column: str, hue_column: str = None, normalize: bool = True
):
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if hue_column is None:
        df = get_outliers(df)
    if x_column == "feature":
        # plot distribution of population-wide features
        fig, axes = plt.subplots(2, 2, figsize=(12, 15))
        for i, feature in enumerate(feature_names):
            if hue_column is None or "Outliers" in hue_column:
                hue_column = f"Population {feature} Outliers"
            sns.histplot(
                df,
                x=feature_names[i],
                hue=hue_column,
                bins=100,
                ax=axes[i % 2][i // 2],
            )
    elif x_column == "regression":
        inputs = []
        for f1 in feature_names:
            for f2 in feature_names:
                combi = set([f1, f2])
                if f1 == f2 or combi in inputs:
                    break
                inputs.append(combi)
        df = get_regressions(df, inputs)
        if df[hue_column].dtype.kind in "iufc":
            # discretize numerical hue column
            df[hue_column] = pd.cut(df[hue_column], bins=10)
        else:
            hue_column = None
        # graph all six possible combinations of pairs
        fig, axes = plt.subplots(len(inputs), 3, figsize=(20, 20))
        for i, col in enumerate(
            df.columns[1:-2]
        ):  # ignore end Age and Brain Volume columns and beginning Subject column
            p = sns.histplot(
                df,
                x=col,
                hue=hue_column,
                palette="mako",
                bins=100,
                ax=axes[i // 3][i % 3],
            )
            if i == 2:
                sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
            else:
                p.get_legend().remove()
    else:
        if df[hue_column].dtype.kind in "iufc":
            # discretize numerical hue column
            df[hue_column] = pd.cut(df[hue_column], bins=10)
        # graph histograms
        ax = sns.histplot(
            df,
            x=x_column,
            hue=hue_column,
            palette="mako",
            bins=100,
            legend=False,
        )
        fig = ax.get_figure()
    fig.savefig(
        f"{RESULTS_PATH}/histograms/x-{x_column}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    return True


def plot_strip_plots(
    df,
    x_column: str = "Region",
    y_column: str = "feature",
    hue_column: str = "Subject",
    normalize: bool = True,
):
    # plot distribution of population-wide features
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if y_column == "feature":
        fig, axes = plt.subplots(4, 1, figsize=(40, 20), sharex=True)
        for i, feature in enumerate(feature_names):
            sns.stripplot(
                df,
                x=x_column,
                y=feature,
                hue=hue_column,
                palette="mako",
                legend=False,
                ax=axes[i],
            )
    else:
        ax = sns.stripplot(
            df, x=x_column, y=y_column, hue=hue_column, legend=False
        )
        fig = ax.get_figure()
    labels = [
        "".join(item.get_text().split("_")[1:3]) for item in plt.xticks()[1]
    ]
    plt.xticks(ticks=plt.xticks()[0], labels=labels, rotation=80)
    fig.savefig(
        f"{RESULTS_PATH}/strips/x-{x_column}-y-{y_column}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    return True


def plot_violin_plots(df, normalize: bool = True):
    # plot distribution of population-wide features
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    df = df.melt(
        id_vars=["Subject", "Region"],
        value_vars=feature_names,
        var_name="Feature",
        value_name="Value",
    )
    sns.violinplot(
        df,
        x="Feature",
        y="Value",
        inner="point",
    )
    plt.savefig(
        f"{RESULTS_PATH}/violins/x-feature-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    return True


def plot_regression_plots(
    df,
    remove_outliers: bool = False,
    average: bool = True,
    normalize: bool = True,
):
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if remove_outliers:
        df = get_outliers(df)
        df = df[~df["Population CT Outliers"]]
        df = df[~df["Population SD Outliers"]]
        df = df[~df["Population MD Outliers"]]
        df = df[~df["Population ICVF Outliers"]]
    if average:
        df = average_across(df, "Region")
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    for i in range(3):
        for j in range(i + 1, 4):
            ax = (
                axes[i][j - 1]
                if i == 0
                else axes[i][j - 2] if i == 1 else axes[1][2]
            )
            sns.regplot(
                df,
                x=feature_names[i],
                y=feature_names[j],
                robust=remove_outliers,
                ax=ax,
            )
            ax.set_ylim(-1.5, 3)
            # calculate slope and intercept of regression equation
            _, _, r, _, _ = scipy.stats.linregress(
                x=df[feature_names[i]], y=df[feature_names[j]]
            )
            # add regression equation to plot
            ax.set_title("r^2 = " + str(round(r**2, 3)))
    fig.savefig(
        f"{RESULTS_PATH}/regressions/x-features-y-features-out-{remove_outliers}-avg-{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


def plot_scatter_plots(
    df,
    x_column: str | List[str] = "feature",
    y_column: str | List[str] = "feature",
    hue_column: str | List[str] = "feature",
    remove_outliers: bool = False,
    average: bool = True,
    normalize: bool = True,
):
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if remove_outliers:
        df = get_outliers(df)
        df = df[~df["Population CT Outliers"]]
        df = df[~df["Population SD Outliers"]]
        df = df[~df["Population MD Outliers"]]
        df = df[~df["Population ICVF Outliers"]]
    if average:
        df = average_across(df, "Region")
    # setup variables
    if x_column == "feature":
        x_column = feature_names
    if y_column == "feature":
        y_column = feature_names
    if hue_column == "feature":
        hue_column = feature_names
    # get all possible combinations of inputs
    inputs = []
    for x in x_column:
        for y in y_column:
            combi = set([x, y])
            if x == y or combi in inputs:
                break
            inputs.append(combi)
    # graph all possible combinations of 3-way combinations
    fig, axes = plt.subplots(
        len(inputs),
        len(hue_column),
        figsize=(10 * len(hue_column), 5 * len(inputs)),
    )
    for i, tup in enumerate(inputs):
        x, y = tup
        for j, hue in enumerate(hue_column):
            p = sns.scatterplot(
                df,
                x=x,
                y=y,
                hue=hue,
                palette="mako",
                ax=axes[i][j] if len(hue_column) > 1 else axes[i],
            )
            sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(
        f"{RESULTS_PATH}/scatters/x-{x_column}-y-{y_column}-hue-{hue_column}-out-{remove_outliers}-avg-{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
    return True


## Plot 3D Features


def plot_3D_features(df, normalize: bool = True):
    # import BrainPlotting
    import sys

    sys.path.insert(1, BRAINPLOT_PATH)
    from PlotBrains import plot_brain

    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    inputs = []
    for f1 in feature_names:
        for f2 in feature_names:
            combi = set([f1, f2])
            if f1 == f2 or combi in inputs:
                break
            inputs.append(combi)
    df = get_regressions(df, inputs, "Region")
    # plot_brain requires numpy array of 360 values (1 per brain region)
    for col in df.columns[1:-2]:
        data = df[col].to_numpy()
        plot_brain(
            data,
            parc="HCP",
            cbar=True,
            cbartitle=col,
            cmap="viridis",
            outfile=f"{RESULTS_PATH}/brainplots/{col}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png",
        )
    return True


## Clustering Results


def plot_clusters(
    df,
    hue_column: str = "Cluster",
    remove_outliers: bool = False,
    average: bool = True,
    normalize: bool = True,
):
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    if remove_outliers:
        df = get_outliers(df)
        df = df[~df["Population CT Outliers"]]
        df = df[~df["Population SD Outliers"]]
        df = df[~df["Population MD Outliers"]]
        df = df[~df["Population ICVF Outliers"]]
    if average:
        df = average_across(df, "Region")
    df = agglomerative_clustering(df, feature_names)
    # graph all six possible combinations of 4-features
    fig, axes = plt.subplots(2, 3, figsize=(12, 15))
    for i in range(3):
        for j in range(i + 1, 4):
            ax = (
                axes[i][j - 1]
                if i == 0
                else axes[i][j - 2] if i == 1 else axes[1][2]
            )
            sns.scatterplot(
                df,
                x=feature_names[i],
                y=feature_names[j],
                hue=hue_column,
                palette="mako",
                ax=ax,
            )
    fig.savefig(
        f"{RESULTS_PATH}/clusters/hue-{hue_column}-out{remove_outliers}-avg{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


def plot_cluster_map(df, normalize: bool = True):
    if normalize:
        feature_names = [
            "CT_norm",
            "SD_norm",
            "MD_norm",
            "ICVF_norm",
            "Subject",
            "Region",
        ]
    else:
        feature_names = ["CT", "SD", "MD", "ICVF", "Subject", "Region"]
    df = df[feature_names]
    df = df.sort_values(by=["Region", "Subject"], ignore_index=True)
    color_indices = df[["Region", "Subject"]].copy()
    subset = len(color_indices["Region"].unique())
    df = df.drop(["Region", "Subject"], axis=1)
    region_cmap = sns.color_palette(
        "hls", n_colors=len(color_indices["Region"].unique())
    )
    region_lut = dict(
        zip(
            color_indices["Region"].unique(),
            np.concatenate([region_cmap, region_cmap]),
        )
    )
    subject_lut = dict(
        zip(
            color_indices["Subject"].unique(),
            sns.color_palette(
                "hls", n_colors=len(color_indices["Subject"].unique())
            ),
        )
    )
    color_indices["Region"] = color_indices["Region"].map(region_lut)
    color_indices["Subject"] = color_indices["Subject"].map(subject_lut)
    sns.clustermap(
        df,
        row_colors=color_indices,
        row_cluster=False,
        col_cluster=False,
        z_score=1 if not normalize else None,
        cmap="icefire",
    )
    plt.savefig(
        f"{RESULTS_PATH}/clusters/clustermap-subset-{subset}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


## PCA Results


def plot_pcas(
    df, feature_columns: List[str], hue_column: str, normalize: bool = True
):
    if normalize:
        df = normalize_data(df)
    df = principle_component_analysis(df, feature_columns)
    if df[hue_column].dtype.kind in "iufc":
        # discretize numerical hue column
        df[hue_column] = pd.cut(df[hue_column], bins=10)
    p = sns.scatterplot(
        df,
        x="Component1",
        y="Component2",
        hue=hue_column,
        palette="mako",
    )
    sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{RESULTS_PATH}/pcas/features-{feature_columns}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


## graph explained variance over number of components
def plot_pca_variance(
    df, additional_columns=[], normalize: bool = True, average: bool = False
):
    if normalize:
        feature_names = [
            "CT_norm",
            "SD_norm",
            "MD_norm",
            "ICVF_norm",
            *additional_columns,
        ]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF", *additional_columns]
    if average:
        df = average_across(df, "Region")
    variances = []
    exp_var = 0
    num_components = 1
    while exp_var < 1:
        pca = principle_component_analysis(df, feature_names, num_components)
        exp_var = np.sum(pca.explained_variance_ratio_)
        variances.append(exp_var)
        num_components += 1

    sns.lineplot(x=range(len(variances)), y=variances)
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Percentage of Explained Variance")
    plt.title("PCA Explained Variance")
    plt.savefig(
        f"{RESULTS_PATH}/pcas/explained-variance-add-{additional_columns}-norm-{normalize}-avg-{average}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


## graph covariance matrix
def plot_pca_covariance(
    df,
    n_components,
    additional_columns=[],
    group_name: str = None,
    normalize: bool = True,
):
    if normalize:
        feature_names = [
            "CT_norm",
            "SD_norm",
            "MD_norm",
            "ICVF_norm",
            *additional_columns,
        ]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF", *additional_columns]
    if group_name is not None:
        df = average_across(df, group_name)
    pca = principle_component_analysis(df, feature_names, n_components)
    cov = pca.get_covariance()
    p = sns.heatmap(
        data=cov,
        cmap=sns.color_palette("mako", as_cmap=True),
        square=True,
        annot=True,
        fmt=".2f",
    )
    p.set_xticks(p.get_xticks(), feature_names)
    p.set_yticks(p.get_yticks(), feature_names)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"PCA Covariance averaged for each {group_name}")
    plt.savefig(
        f"{RESULTS_PATH}/pcas/covariance-add-{additional_columns}-group-{group_name}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


## graph eigenvector components (pairwise)
def plot_pca_eigenvectors(
    df,
    n_components,
    additional_columns=[],
    removed_columns=[],
    group_name: str = None,
    normalize: bool = True,
):
    if normalize:
        feature_names = [
            "CT_norm",
            "SD_norm",
            "MD_norm",
            "ICVF_norm",
            *additional_columns,
        ]
        df = normalize_data(df)
    else:
        feature_names = ["CT", "SD", "MD", "ICVF", *additional_columns]
    for col in removed_columns:
        feature_names.remove(col)
    if group_name is not None:
        df = average_across(df, group_name)
    pca = principle_component_analysis(df, feature_names, n_components)
    components = pca.components_
    component_names = [f"Component {i}" for i in range(n_components)]
    variances = pca.explained_variance_ratio_
    cmap = sns.color_palette(n_colors=n_components, as_cmap=True)
    fig, axes = plt.subplots(
        len(feature_names),
        len(feature_names),
        sharey="row",
        figsize=(4 * len(feature_names), 6 * len(feature_names)),
    )
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            feature1 = components[:, i]
            name1 = feature_names[i]
            feature2 = components[:, j]
            name2 = feature_names[j]
            if i == j:
                axes[i][j].bar(component_names, feature1, color=cmap)
            else:
                for k in range(len(feature1)):
                    axes[i][j].arrow(
                        0,
                        0,
                        feature2[k],
                        feature1[k],
                        width=0.05 * variances[k],
                        color=cmap[k],
                    )
            axes[i][j].set_ylabel(name1)
            axes[i][j].set_xlabel(name2)
    a = inset_axes(
        axes[0][len(feature_names) - 1],
        width="30%",  # width = 30% of parent_bbox
        height="20%",  # height : 1 inch
        loc=1,
    )
    sns.barplot(x=range(n_components), y=variances, ax=a)
    a.set_title("sensitivity")
    leg = fig.legend(component_names)
    for i in range(n_components):
        leg.legend_handles[i].set_color(cmap[i])
    fig.suptitle(f"PCA Eigenvectors for Groups: {group_name}")
    fig.savefig(
        f"{RESULTS_PATH}/pcas/eigenvectors-add-{additional_columns}-rem-{removed_columns}-group-{group_name}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


# GPLVM Results


## graph correlation matrices (Use sns.clustermap)
def plot_gp_correlation(
    df,
    n_components: int = 2,
    use_file: bool = True,
    normalize: bool = False,
    cluster: bool = False,
    sort: bool = False,
    subset: int = 10,
    sort_by: List[str] = ["Region", "Subject", "Feature"],
    model_name: str = "model",
):
    # import BrainPlotting
    import sys

    sys.path.insert(1, BRAINPLOT_PATH)
    from PlotBrains import plot_brain

    if use_file:
        model = create_named_gp(
            df,
            model_name,
            n_components=n_components,
            sort_by=sort_by,
            optimize=False,
        )
        model = load_gp_model(model, f"./src/gp_models/{model_name}.npy")
    else:
        model = create_named_gp(
            df,
            model_name,
            n_components=n_components,
            sort_by=sort_by,
            optimize=True,
        )
    correlation, labels = get_gp_covariance(df, model, sort_by=sort_by)
    if subset is not None:
        subset_length = df.loc[
            df.index.get_level_values("Region").unique()[:subset].to_list(),
            :,
            :,
        ].shape[0]
        correlation = correlation.iloc[:subset_length, :subset_length]
        labels = labels[:subset_length]
    index_names = pd.DataFrame(labels, columns=sort_by)
    color_indices = pd.DataFrame(labels, columns=sort_by)
    region_lut = dict(
        zip(
            color_indices["Region"].unique(),
            (
                sns.color_palette(
                    "hls", n_colors=len(index_names["Region"].unique())
                )
                if len(color_indices["Region"].unique()) <= 20
                else sns.color_palette(
                    "mako", n_colors=len(index_names["Region"].unique())
                )
            ),
        )
    )

    color_indices["Region"] = color_indices["Region"].map(region_lut)

    colors = [color_indices["Region"]]
    if "Feature" in sort_by:
        feature_lut = dict(
            zip(
                color_indices["Feature"].unique(),
                sns.color_palette(
                    "hls", n_colors=len(color_indices["Feature"].unique())
                ),
            )
        )
        color_indices["Feature"] = color_indices["Feature"].map(feature_lut)
        colors.append(color_indices["Feature"])
    if "Subject" in sort_by:
        subject_lut = dict(
            zip(
                color_indices["Subject"].unique(),
                sns.color_palette(
                    "mako", n_colors=len(color_indices["Subject"].unique())
                ),
            )
        )
        color_indices["Subject"] = color_indices["Subject"].map(subject_lut)
        colors.append(color_indices["Subject"])
    x = correlation.to_numpy()
    if normalize:
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    if sort:
        sorted_indices = np.flip(np.argsort(x, axis=1), axis=1)
        x = np.take_along_axis(x, sorted_indices, axis=1)
    sns.clustermap(
        x,
        row_colors=colors,
        col_colors=(None if sort else colors),
        row_cluster=False,
        col_cluster=cluster,
        cmap="mako",
    )
    plt.savefig(
        f"{RESULTS_PATH}/gps/correlation-{model_name}-clst{cluster}-sort{sort}-subset{subset}-{sort_by}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
    plot_brain(
        index_names["Region"].unique(),
        parc="HCP",
        cbar=True,
        cbartitle="Region Indices",
        cmap=(
            sns.color_palette(
                "hls",
                n_colors=len(index_names["Region"].unique()),
                as_cmap=True,
            )
            if len(index_names["Region"].unique()) <= 20
            else "mako"
        ),
        outfile=f"{RESULTS_PATH}/brainplots/correlation-{model_name}-clst{cluster}-sort{sort}-subset{subset}-{sort_by}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png",
        categorical=True,
    )
    plt.close()


## graph gplvm fit
def plot_gp_latents(
    df, sort_by, n_components: int = 2, model_name: str = "model"
):
    model = create_named_gp(
        df, model_name, n_components, sort_by=sort_by, optimize=False
    )
    labels = np.array([i[1] for i in df.index.values])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    model.kern.plot_ARD(ax=axes[0])
    model.plot_latent(labels=labels, marker="<^>v", legend=False, ax=axes[1])
    model.plot_magnification(
        labels=labels, marker="<^>v", legend=True, ax=axes[2]
    )
    fig.suptitle(f"GP-LVM Latents for all data")
    fig.savefig(
        f"{RESULTS_PATH}/gps/lvm-latents-{model_name}-ncomp-{n_components}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


## graph bgplvm fit
def plot_bgp_latents(df, n_components: int = 5):
    model = create_named_gp(df, "bayesian_model", n_components)
    labels = np.array([i[1] for i in df.index.values])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    model.kern.plot_ARD(ax=axes[0])
    model.plot_latent(labels=labels, marker="<^>v", legend=False, ax=axes[1])
    model.plot_magnification(
        labels=labels, marker="<^>v", legend=True, ax=axes[2]
    )
    fig.suptitle(f"Bayesian GP-LVM Latents for all data")
    fig.savefig(
        f"{RESULTS_PATH}/gps/blvm-latents-feature-ncomp-{n_components}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
