import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import glob
from datetime import datetime

from constants import DATASET_FOLDER, RESULTS_FOLDER


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


def get_dataset():
    """Loads the dataset as a DataFrame with 7 columns: [brain] Region, CT, SD, MD, ICVF, Subject, Label"""
    df = pd.concat(
        map(load_csv, glob.glob(f"{DATASET_FOLDER}/*.csv")), ignore_index=True
    )
    return df


def plot_histograms(df, category_name: str = None, normalize: bool = True):
    # plot distribution of population-wide features
    fig, axes = plt.subplots(2, 2, figsize=(12, 15))
    feature_names = ["CT", "SD", "MD", "ICVF"]
    for i, feature in enumerate(feature_names):
        if category_name is None:
            category_name = f"Population {feature} Outliers"
        sns.histplot(
            df, x=feature, hue=category_name, bins=100, ax=axes[i % 2][i // 2]
        )
    fig.savefig(
        f"{RESULTS_FOLDER}/{'normalized' if normalize else 'unnormalized'}_feature_histograms/{category_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    # plot average histogram of each subject for each feature (low opacity overlayed plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 15))
    for i, feature in enumerate(feature_names):
        if category_name is None:
            category_name = f"Subject {feature} Outliers"
        sns.histplot(
            df,
            x=feature,
            hue=category_name,
            bins=100,
            legend=False,
            ax=axes[i % 2][i // 2],
        )
    fig.savefig(
        f"{RESULTS_FOLDER}/{'normalized' if normalize else 'unnormalized'}_feature_histograms_per_subject/{category_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    return True


def plot_violin_plots(df, normalize: bool = True):
    # plot distribution of population-wide features
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
    df = df.melt(
        id_vars=["Subject", "Region"],
        value_vars=feature_names,
        var_name="Feature",
        value_name="Normalized Value" if normalize else "Value",
    )
    sns.violinplot(
        df,
        x="Feature",
        y="Normalized Value" if normalize else "Value",
        inner="point",
    )
    plt.savefig(
        f"{RESULTS_FOLDER}/{'normalized' if normalize else 'unnormalized'}_feature_violins/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    plt.close()
    return True


def plot_clusters(
    df,
    category_name: str = "Cluster",
    remove_outliers: bool = True,
    average: bool = False,
    normalize: bool = True,
):
    if normalize:
        feature_names = ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
    else:
        feature_names = ["CT", "SD", "MD", "ICVF"]
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
                hue=category_name,
                ax=ax,
            )
    fig.savefig(
        f"{RESULTS_FOLDER}/{'normalized' if normalize else 'unnormalized'}_clusters/{category_name}-{remove_outliers}-{average}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    plt.close()
