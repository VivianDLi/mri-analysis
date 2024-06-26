from typing import Dict, List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

from mri_analysis.constants import RESULTS_PATH
from mri_analysis.datatypes import DATA_FEATURES, ClusterPlotType, PlotConfig

from loguru import logger

from mri_analysis.utils import get_time_identifier


class ClusterPlotter:
    plots: Dict[ClusterPlotType, PlotConfig] = None

    def __init__(self, plots: List[ClusterPlotType]):
        self.plots = {plot: {} for plot in plots}

    def set_plot_configs(
        self, configs: Dict[ClusterPlotType, PlotConfig]
    ) -> None:
        for plot, config in configs.items():
            if plot not in self.plots:
                logger.warning(
                    f"Plot {plot} not found in available plots when setting configs."
                )
                continue
            self.plots[plot] = config

    def create_plots(
        self, data: pd.DataFrame = None, flat_data: pd.DataFrame = None
    ) -> None:
        logger.debug(f"Creating cluster plots {self.plots}...")
        for plot, config in self.plots.items():
            # check for correct dataset
            if "scatter" in plot and data is None:
                logger.warning(
                    f"Scatter cluster plot {plot} requires a DataFrame to be passed as the <data> argument."
                )
                continue
            if "map" in plot and flat_data is None:
                logger.warning(
                    f"Map cluster plot {plot} requires a flattened DataFrame to be passed as the <flat_data> argument."
                )
                continue
            # plot
            match plot:
                case "cluster_scatter":
                    self._plot_cluster_scatter(data, **config)
                case "cluster_map":
                    self._plot_cluster_map(
                        data=data, flat_data=flat_data, **config
                    )
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_cluster_scatter(self, data: pd.DataFrame, **kwargs) -> None:
        cluster_data = self._cluster(data)
        # get all possible combinations of inputs
        inputs = {}
        for x in DATA_FEATURES:
            for y in set(DATA_FEATURES) - set([x]):
                if x not in inputs:
                    inputs[x] = set()
                # check for re-ordered duplicates
                if y not in inputs:
                    inputs[x].add(y)
            # prevent empty sets
            if len(inputs[x]) == 0:
                del inputs[x]
        # graph all possible combinations of 3-way combinations
        fig = plt.figure(figsize=(15, len(inputs) * 3))
        subfigs = fig.subfigures(nrows=len(inputs), ncols=1)
        for i, x_feat in enumerate(inputs):
            axs = subfigs[i].subplots(nrows=1, ncols=len(inputs[x_feat]))
            for j, y_feat in enumerate(inputs[x_feat]):
                p = sns.scatterplot(
                    cluster_data,
                    x=x_feat,
                    y=y_feat,
                    hue="Cluster",
                    palette="mako",
                    ax=axs[j] if len(inputs[x_feat]) > 1 else axs,
                    **kwargs,
                )
                p.set_xlabel(x_feat)
                p.set_ylabel(y_feat)
        plt.suptitle("Population-Wide 2-feature Clustered Scatter Plots")
        plt.savefig(
            f"{RESULTS_PATH}/clustering/cluster_scatter_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_cluster_map(
        self,
        data: pd.DataFrame = None,
        flat_data: pd.DataFrame = None,
        **kwargs,
    ) -> None:
        if data is not None:
            if not set(DATA_FEATURES) <= set(data.columns):
                logger.warning(
                    f"{DATA_FEATURES} columns not found in dataset columns: {data.columns}."
                )
                return
            number_columns = data.select_dtypes(include="number").columns
            non_number_columns = set(data.columns) - set(number_columns)
            feature_columns = set(number_columns) & set(DATA_FEATURES)
            color_indices = data[list(non_number_columns)].copy()
            for feat in non_number_columns:
                color = "hls" if data[feat].unique().size <= 20 else "mako"
                cmap = sns.color_palette(
                    color, n_colors=len(data[feat].unique())
                )
                lut = dict(zip(data[feat].unique(), cmap))
                color_indices[feat] = color_indices[feat].map(lut)
            sns.clustermap(
                data[list(feature_columns)],
                row_colors=color_indices,
                row_cluster=False,
                col_cluster=False,
                cmap="icefire",
                **kwargs,
            )
            plt.suptitle("Population-Wide Clustered Heatmaps")
            plt.savefig(
                f"{RESULTS_PATH}/clustering/cluster_map_{get_time_identifier()}.png"
            )
            plt.close()
        if flat_data is not None:
            if "Value" not in flat_data.columns:
                logger.warning(
                    f"Value column not found in dataset columns: {flat_data.columns}."
                )
                return
            number_columns = flat_data.select_dtypes(include="number").columns
            non_number_columns = set(flat_data.columns) - set(number_columns)
            color_indices = flat_data[list(non_number_columns)].copy()
            for feat in non_number_columns:
                color = "hls" if data[feat].unique().size <= 20 else "mako"
                cmap = sns.color_palette(
                    color, n_colors=len(flat_data[feat].unique())
                )
                lut = dict(zip(flat_data[feat].unique(), cmap))
                color_indices[feat] = color_indices[feat].map(lut)
            sns.clustermap(
                flat_data[number_columns],
                row_colors=color_indices,
                row_cluster=False,
                col_cluster=False,
                cmap="icefire",
                **kwargs,
            )
            plt.suptitle("Population-Wide Clustered Heatmaps")
            plt.savefig(
                f"{RESULTS_PATH}/clustering/cluster_flat_map_{get_time_identifier()}.png"
            )
            plt.close()

    def _cluster(self, data: pd.DataFrame) -> pd.DataFrame:
        clusters = AgglomerativeClustering(
            5, metric="euclidean", linkage="ward", connectivity=None
        ).fit_predict(data[DATA_FEATURES].to_numpy())
        data["Cluster"] = clusters
        return data
