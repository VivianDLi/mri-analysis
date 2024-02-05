import math
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

import os
import glob
from datetime import datetime

from .clustering import agglomerative_clustering, principle_component_analysis
from .processing import *
from .BrainPlottingShare import plot_brain
from constants import DATASET_FOLDER, RESULTS_FOLDER


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
        map(load_csv, glob.glob(f"{DATASET_FOLDER}/*.csv")), ignore_index=True
    )
    # merge in demographics information
    demographics = pd.read_csv(
        "./src/dataset_tests/demographics_and_etiv.csv", header=0
    ).rename(
        columns={
            "ID": "Subject",
            "Age at visit to assessment centre (Imaging)": "Age",
            "Estimated Total Intracranial Volume": "Brain Volume",
        }
    )
    df = df.merge(demographics, how="left", on="Subject")
    return df


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
        f"{RESULTS_FOLDER}/histograms/x-{x_column}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
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
        f"{RESULTS_FOLDER}/strips/x-{x_column}-y-{y_column}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
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
        f"{RESULTS_FOLDER}/violins/x-feature-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
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
        f"{RESULTS_FOLDER}/regressions/x-features-y-features-out-{remove_outliers}-avg-{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
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
        f"{RESULTS_FOLDER}/scatters/x-{x_column}-y-{y_column}-hue-{hue_column}-out-{remove_outliers}-avg-{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
    return True


def plot_3D_features(df, normalize: bool = True):
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
            outfile=f"{RESULTS_FOLDER}/brainplots/{col}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png",
        )
    return True


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
        f"{RESULTS_FOLDER}/pcas/features-{feature_columns}-hue-{hue_column}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()


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
        df = average_across(df, "Subject")
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
        f"{RESULTS_FOLDER}/clusters/hue-{hue_column}-out{remove_outliers}-avg{average}-norm-{normalize}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
