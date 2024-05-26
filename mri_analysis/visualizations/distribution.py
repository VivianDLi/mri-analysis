"""Plots distribution information about raw data."""

from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mri_analysis.constants import RESULTS_PATH
from mri_analysis.datatypes import (
    DATA_FEATURES,
    DistributionPlotType,
    PlotConfig,
    RegressionOutput,
)
from mri_analysis.utils import get_time_identifier

from loguru import logger


class DistributionPlotter:
    plots: Dict[DistributionPlotType, PlotConfig] = None

    def __init__(self, plots: List[DistributionPlotType]):
        self.plots = {plot: {} for plot in plots}

    def set_plot_configs(
        self, configs: Dict[DistributionPlotType, PlotConfig]
    ) -> None:
        for plot, config in configs.items():
            if plot not in self.plots:
                logger.warning(
                    f"Plot {plot} not found in available plots when setting configs."
                )
                continue
            self.plots[plot] = config

    def create_plots(
        self,
        data: pd.DataFrame = None,
        regression_data: List[RegressionOutput] = None,
    ) -> None:
        logger.debug(f"Creating distribution plots {self.plots}...")
        for plot, config in self.plots.items():
            # check for correct dataset
            if "feature" in plot and data is None:
                logger.warning(
                    f"Feature-based plot {plot} requires a DataFrame to be passed as the <data> argument."
                )
                continue
            if "regression" in plot and regression_data is None:
                logger.warning(
                    f"Regression-based plot {plot} requires a RegressionOutput to be passed as the <regression_data> argument."
                )
                continue
            # plot
            match plot:
                case "feature_histogram":
                    self._plot_feature_histogram(data, **config)
                case "regression_histogram":
                    self._plot_regression_histogram(regression_data, **config)
                case "feature_scatter":
                    self._plot_feature_scatter(data, **config)
                case "feature_regression":
                    self._plot_feature_regression(
                        data, regression_data, **config
                    )
                case "feature_strip":
                    self._plot_feature_strip(data, **config)
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_feature_histogram(self, data: pd.DataFrame, **kwargs) -> None:
        # plot distribution of population-wide features
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for i, feature in enumerate(DATA_FEATURES):
            sns.histplot(
                data, x=feature, bins=100, ax=axes[i % 2][i // 2], **kwargs
            )
        fig.suptitle("Population-Wide Histogram of Feature Distributions")
        fig.savefig(
            f"{RESULTS_PATH}/distributions/feature_histogram_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_regression_histogram(
        self, data: List[RegressionOutput], **kwargs
    ) -> None:
        # distribution of regression features
        fig = plt.figure(figsize=(15, len(data) * 3))
        subfigs = fig.subfigures(nrows=len(data), ncols=1)
        for i, reg_data in enumerate(data):
            subfigs[i].suptitle(
                f"Intercept, Slope, and R^2 for {reg_data['feat_1']} vs. {reg_data['feat_2']}"
            )
            axs = subfigs[i].subplots(nrows=1, ncols=3)
            p_intercept = sns.histplot(
                x=reg_data["region_intercepts"],
                palette="mako",
                bins=100,
                ax=axs[0],
                **kwargs,
            )
            p_slope = sns.histplot(
                x=reg_data["region_slopes"],
                palette="mako",
                bins=100,
                ax=axs[1],
                **kwargs,
            )
            p_r = sns.histplot(
                x=reg_data["region_r_squared"],
                palette="mako",
                bins=100,
                ax=axs[2],
                **kwargs,
            )
        plt.suptitle("Linear Regression Histograms for Region-Wide Data")
        plt.savefig(
            f"{RESULTS_PATH}/distributions/regression_histogram_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_feature_scatter(self, data: pd.DataFrame, **kwargs) -> None:
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
        fig = plt.figure(figsize=(15, len(inputs) * 5))
        subfigs = fig.subfigures(nrows=len(inputs), ncols=1)
        for i, x_feat in enumerate(inputs):
            axs = subfigs[i].subplots(
                nrows=1, ncols=len(inputs[x_feat]) * (len(DATA_FEATURES) - 2)
            )
            for j, y_feat in enumerate(inputs[x_feat]):
                hues = set(DATA_FEATURES) - set([x_feat]) - set([y_feat])
                for k, hue_feat in enumerate(hues):
                    p = sns.scatterplot(
                        data,
                        x=x_feat,
                        y=y_feat,
                        hue=hue_feat,
                        palette="mako",
                        ax=axs[j * len(hues) + k],
                        **kwargs,
                    )
                    p.set_xlabel(x_feat)
                    p.set_ylabel(y_feat)
                    sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
        plt.suptitle("Population-Wide 3-feature Scatter Plots")
        plt.savefig(
            f"{RESULTS_PATH}/distributions/feature_scatter_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_feature_regression(
        self,
        data: pd.DataFrame,
        regression_data: List[RegressionOutput],
        **kwargs,
    ) -> None:
        inputs = {}
        # get input combinations from regression information
        for reg_data in regression_data:
            if reg_data["feat_1"] not in inputs:
                inputs[reg_data["feat_1"]] = {}
            inputs[reg_data["feat_1"]][reg_data["feat_2"]] = reg_data
        # graph all possible combinations of 2-way combinations
        fig = plt.figure(figsize=(15, len(inputs) * 3))
        subfigs = fig.subfigures(nrows=len(inputs), ncols=1)
        for i, x_feat in enumerate(inputs):
            axs = subfigs[i].subplots(nrows=1, ncols=len(inputs[x_feat]))
            for j, y_feat in enumerate(inputs[x_feat]):
                reg_data = inputs[x_feat][y_feat]
                p = sns.regplot(
                    data,
                    x=x_feat,
                    y=y_feat,
                    ax=axs[j] if len(inputs[x_feat]) > 1 else axs,
                    label=f"y={reg_data['slope']}x+{reg_data['intercept']}, {reg_data['r_squared']}",
                    **kwargs,
                )
                p.set_xlabel(x_feat)
                p.set_ylabel(y_feat)
        plt.suptitle("Population-Wide 2-feature Regression Plots")
        plt.savefig(
            f"{RESULTS_PATH}/distributions/feature_regression_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_feature_strip(self, data: pd.DataFrame, **kwargs) -> None:
        fig, axs = plt.subplots(
            len(DATA_FEATURES), 1, figsize=(40, 20), sharex=True
        )
        for i, feat in enumerate(DATA_FEATURES):
            sns.stripplot(
                data,
                x="Region",
                y=feat,
                hue="Subject",
                palette="mako",
                legend=False,
                ax=axs[i],
                **kwargs,
            )
        # rotate x-axis labels (assuming regions)
        labels = [
            "".join(item.get_text().split("_")[1:3])
            for item in plt.xticks()[1]
        ]
        plt.xticks(ticks=plt.xticks()[0], labels=labels, rotation=80)
        fig.suptitle("Population-Wide Strip Plot of Feature Distributions")
        fig.savefig(
            f"{RESULTS_PATH}/distributions/feature_strip_{get_time_identifier()}.png"
        )
        plt.close()
