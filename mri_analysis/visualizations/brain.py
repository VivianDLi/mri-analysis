"""Plots information onto a 3D brain model."""

from typing import Dict, List

from matplotlib import pyplot as plt
import pandas as pd

from mri_analysis.constants import BRAINPLOT_PATH, RESULTS_PATH
from mri_analysis.datatypes import (
    DATA_FEATURES,
    BrainPlotType,
    PlotConfig,
    RegressionOutput,
)
from mri_analysis.utils import get_time_identifier

from loguru import logger


class BrainPlotter:
    plots: Dict[BrainPlotType, PlotConfig] = None

    def __init__(self, plots: List[BrainPlotType]):
        self.plots = {plot: {} for plot in plots}
        # initialize brain plotting
        import sys

        sys.path.insert(1, BRAINPLOT_PATH)
        from PlotBrains import plot_brain

        self.plot_brain = plot_brain

    def set_plot_configs(
        self, configs: Dict[BrainPlotType, PlotConfig]
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
        logger.debug(f"Creating linear plots {self.plots}...")
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
                case "brain_feature":
                    self._plot_brain_feature(data, **config)
                case "brain_regression":
                    self._plot_brain_regression(regression_data, **config)
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_brain_feature(self, data: pd.DataFrame, **kwargs) -> None:
        # plot_brain requires numpy array of 360 values (1 per brain region)
        if len(data) != 360:
            logger.warning(
                f"Dataframe has {len(data)} rows, expected 360 for brain plotting."
            )
            return
        for feat in DATA_FEATURES:
            self.plot_brain(
                data[feat].to_numpy().flatten(),
                parc="HCP",
                cbar=True,
                cbartitle=feat,
                cmap="viridis",
                outfile=f"{RESULTS_PATH}/brain/brain_feature_{feat}_{get_time_identifier()}.png",
            )
        plt.close()

    def _plot_brain_regression(
        self, regression_data: List[RegressionOutput], **kwargs
    ) -> None:
        # plot_brain requires numpy array of 360 values (1 per brain region)
        for reg_data in regression_data:
            if len(reg_data["region_slopes"]) != 360:
                logger.warning(
                    f"Regression data for {reg_data["feat_1"]} x {reg_data["feat_2"]} has {len(reg_data["region_slopes"])} rows, expected 360 for brain plotting."
                )
                return
            self.plot_brain(
                reg_data["region_slopes"],
                parc="HCP",
                cbar=True,
                cbartitle=f"{reg_data["feat_1"]} vs. {reg_data["feat_2"]} Slopes",
                cmap="viridis",
                outfile=f"{RESULTS_PATH}/brain/brain_regression_{reg_data["feat_1"]}x{reg_data["feat_2"]}_slope_{get_time_identifier()}.png",
            )
            self.plot_brain(
                reg_data["region_intercepts"],
                parc="HCP",
                cbar=True,
                cbartitle=f"{reg_data["feat_1"]} vs. {reg_data["feat_2"]} Intercepts",
                cmap="viridis",
                outfile=f"{RESULTS_PATH}/brain/brain_regression_{reg_data["feat_1"]}x{reg_data["feat_2"]}_intercept_{get_time_identifier()}.png",
            )
            self.plot_brain(
                reg_data["region_r_squared"],
                parc="HCP",
                cbar=True,
                cbartitle=f"{reg_data["feat_1"]} vs. {reg_data["feat_2"]} R^2",
                cmap="viridis",
                outfile=f"{RESULTS_PATH}/brain/brain_regression_{reg_data["feat_1"]}x{reg_data["feat_2"]}_r_{get_time_identifier()}.png",
            )
        plt.close()