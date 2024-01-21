import glob

from constants import DATASET_FOLDER
from dataset_tests import agglomerative_clustering

for file in glob.glob(f"{DATASET_FOLDER}/*.csv"):
    agglomerative_clustering(file)
