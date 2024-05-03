"""Loads data from the dataset folder and returns it in an interactible form."""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from GPy.kern import RBF, White, Linear

from mri_analysis.constants import DATA_PATH, METADATA_PATH
from mri_analysis.datatypes import Dataset

from loguru import logger


def _load_subject_data(file_path: str) -> pd.DataFrame:
    subject_name = os.path.basename(file_path)[
        :-4
    ]  # extract subject name from file name
    df = pd.read_csv(file_path, header=0)
    df.rename(columns={df.columns[0]: "Region"}, inplace=True)
    df["Subject"] = subject_name
    return df


def load_dataset() -> Dataset:
    """Loads a dataset consisting of .csv files in the DATA_PATH folder. Also includes metadata if available."""
    # load data features: subject, region, CT, SD, MD, ICVF
    dataset_files = glob.glob(f"{DATA_PATH}/*.csv")
    assert (
        len(dataset_files) > 0
    ), f"No dataset files found in data path: {DATA_PATH}"
    data = pd.concat(map(_load_subject_data, dataset_files), ignore_index=True)

    ## load metadata features
    # load von Economo labels
    if os.path.isfile(f"{METADATA_PATH}/labels.csv"):
        logger.info(f"Found von Economo labels. Adding to existing data.")
        labels_df = pd.read_csv(
            f"{METADATA_PATH}/labels.csv", header=0, dtype=str
        )
        labels_df["Labels"] = "L_" + labels_df["Labels"]
        # combine data with labels
        data = data.merge(labels_df, how="left", on="Region")
    # load subject demographics
    if os.path.isfile(f"{METADATA_PATH}/demographics.csv"):
        logger.info(f"Found subject demographics. Adding to existing data.")
        demographics_df = pd.read_csv(
            f"{METADATA_PATH}/demographics.csv", header=0
        )
        demographics_df.rename(
            columns={
                "ID": "Subject",
                "Age at visit to assessment centre (Imaging)": "Age",
                "Estimated Total Intracranial Volume": "Brain Volume",
            },
            inplace=True,
        )
        # combine data with demographics
        data = data.merge(demographics_df, how="left", on="Subject")

    return Dataset(data)


def load_synthetic_dataset(
    n_shared: int,
    n_indv: int,
    classes: List[str],
    n_subjects: int,
    n_regions: int = 360,
    noise: float = 0.01,
) -> Tuple[Dataset, int]:
    """Loads a synthetic dataset for testing purposes.

    Args:
        n_shared (int): number of shared latent dimensions
        n_indv (int): number of individual latent dimensions per class
        n_classes (int): number of unique classes for MRD
        n_samples (int): number of data samples per class
    """
    data = {}
    data["Region"] = np.tile(
        np.arange(n_regions).astype(str), n_subjects
    ).tolist()
    data["Subject"] = np.repeat(
        np.arange(n_subjects).astype(str), n_regions
    ).tolist()
    n_components = n_shared + len(classes) * n_indv
    shared_length_scale = [np.random.uniform(1, 6) for _ in range(n_shared)]
    for i in range(len(classes)):
        indv_length_scale = [np.random.uniform(1, 6) for _ in range(n_indv)]
        kernel = RBF(
            n_shared + n_indv,
            lengthscale=shared_length_scale + indv_length_scale,
            ARD=True,
            name=f"RBF_{i}",
        ) + White(
            n_indv,
            variance=noise,
            name=f"White_{i}",
        )
        t = np.c_[
            [np.linspace(-1, 5, n_regions) for _ in range(n_shared + n_indv)]
        ].T
        K = kernel.K(t)
        region_means = np.zeros(n_regions)
        data[classes[i]] = (
            np.random.multivariate_normal(region_means, K, size=(n_subjects,))
            .flatten()
            .tolist()
        )
        print(np.mean(data[classes[i]]), np.std(data[classes[i]]))
    data_df = pd.DataFrame(data)
    return (
        Dataset(data_df),
        n_components,
    )


def load_synthetic_dataframe(
    n_shared: int,
    n_indv: int,
    classes: Dict[str, int],
    n_samples: int = 360,
    white_noise: float = 0.01,
    hd_noise: float = 0.3,
) -> pd.DataFrame:
    n_components = n_shared + len(classes) * n_indv
    kernel = RBF(
        n_components,
        lengthscale=np.random.uniform(1, 6, n_components),
        ARD=True,
        name="RBF",
    )
    for i in range(len(classes)):
        kernel += White(
            n_indv,
            active_dims=range(
                n_shared + i * n_indv, n_shared + (i + 1) * n_indv
            ),
            variance=white_noise,
            name=f"White_{i}",
        )
    t = np.c_[[np.linspace(-1, 5, n_samples) for _ in range(n_components)]].T
    K = kernel.K(t)
    region_means = np.zeros(n_samples)
    data = np.random.multivariate_normal(
        region_means, K, size=(n_components,)
    ).T  # n_samples x n_components
    class_arrays = []
    for i, n_dim in enumerate(classes.values()):
        shared_data = data[:, 0:n_shared]
        indv_data = data[
            :, n_shared + i * n_indv : n_shared + (i + 1) * n_indv
        ]
        total_data = np.hstack([shared_data, indv_data])
        # high-dimensional expansion
        total_data = total_data.dot(
            np.random.randn(total_data.shape[1], n_dim)
        ) + hd_noise * np.random.randn(total_data.shape[0], n_dim)
        total_data -= total_data.mean(0)
        total_data /= total_data.std(0)
        class_arrays.append(total_data)
    row_index = pd.Index(np.arange(n_samples).astype(str), name="Region")
    column_index = pd.MultiIndex.from_tuples(
        [
            (class_name, i)
            for class_name, n_dim in classes.items()
            for i in range(n_dim)
        ],
        names=["Class", "Dimension"],
    )
    data_df = pd.DataFrame(
        np.hstack(class_arrays), columns=column_index, index=row_index
    )
    return data_df
