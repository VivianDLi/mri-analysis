from constants import DATASET_FOLDER
from dataset_tests import *

df = get_dataset()
outlier_df = get_outliers(df)
# plot_histograms(outlier_df, normalize=True)
# plot_histograms(outlier_df, category_name="Subject", normalize=True)
# plot_histograms(outlier_df, category_name="Label", normalize=True)
# plot_violin_plots(outlier_df)
clustered_df = agglomerative_clustering(
    outlier_df, ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]
)
clustered_average_df = agglomerative_clustering(
    outlier_df, ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"], average=True
)
clustered_average_df_outliers = agglomerative_clustering(
    outlier_df,
    ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"],
    remove_outliers=False,
    average=True,
)
plot_clusters(clustered_df, True, False)
plot_clusters(clustered_df, "Label", True, False)
plot_clusters(clustered_average_df, True, True)
plot_clusters(clustered_average_df_outliers, False, True)
