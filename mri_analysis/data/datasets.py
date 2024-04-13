"""Loads data from the dataset folder and returns it in an interactible form."""

import os
import glob
import pandas as pd

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
