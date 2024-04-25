"""Plots nonlinear decomposition information about data."""

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mri_analysis.constants import BRAINPLOT_PATH, RESULTS_PATH
from mri_analysis.datatypes import (
    CovarianceOutput,
    LatentOutput,
    NonlinearPlotType,
    PlotConfig,
    SensitivityOutput,
)
from mri_analysis.utils import get_time_identifier

from loguru import logger


class NonlinearPlotter:
    plots: Dict[NonlinearPlotType, PlotConfig] = None

    def __init__(self, plots: List[NonlinearPlotType]):
        self.plots = {plot: {} for plot in plots}

    def set_plot_configs(
        self, configs: Dict[NonlinearPlotType, PlotConfig]
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
        covariance_data: Dict[str, CovarianceOutput] = None,
        covariance_labels: pd.DataFrame = None,
        sensitivity_data: Dict[str, SensitivityOutput] = None,
        latent_data: LatentOutput = None,
        name: str = None,
    ) -> None:
        logger.debug(f"Creating nonlinear plots {self.plots}...")
        for plot, config in self.plots.items():
            # check for correct dataset
            if "covariance" in plot and (
                covariance_data is None or covariance_labels is None
            ):
                logger.warning(
                    f"Covariance plot {plot} requires a dictionary to be passed as the <covariance_data> argument and a dataframe to be passed as the <covariance_labels> argument."
                )
                continue
            if "sensitivity" in plot and sensitivity_data is None:
                logger.warning(
                    f"Sensitivity plot {plot} requires a dictionary to be passed as the <sensitivity_data> argument."
                )
                continue
            if "latents" in plot and latent_data is None:
                logger.warning(
                    f"Latent space plot {plot} requires a dictionary to be passed as the <latent_data> argument."
                )
                continue
            # plot
            match plot:
                case "gp_covariance":
                    self._plot_gp_covariance(
                        covariance_data, covariance_labels, name=name, **config
                    )
                case "gp_sensitivity":
                    self._plot_gp_sensitivity(
                        sensitivity_data, name=name, **config
                    )
                case "gp_latents":
                    self._plot_gp_latents(latent_data, name=name, **config)
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_gp_covariance(
        self,
        covariance_data: Dict[str, CovarianceOutput],
        covariance_labels: pd.DataFrame,
        name: str = None,
        **kwargs,
    ) -> None:
        # initialize brain plotting
        import sys

        sys.path.insert(1, BRAINPLOT_PATH)
        from PlotBrains import plot_brain

        color_indices = covariance_labels.copy()
        for label in color_indices.columns:
            color = (
                "hls"
                if (
                    label == "Region"
                    and len(color_indices[label].unique()) <= 20
                )
                or label == "Feature"
                else "mako"
            )
            cmap = sns.color_palette(
                color, n_colors=len(color_indices[label].unique())
            )
            lut = dict(zip(color_indices[label].unique(), cmap))
            color_indices[label] = color_indices[label].map(lut)
        for feature, covariance in covariance_data.items():
            # plot covariance matrix
            df_covariance = pd.DataFrame(covariance)
            sns.clustermap(
                df_covariance,
                row_colors=color_indices,
                col_colors=color_indices,
                row_cluster=False,
                col_cluster=False,
                cmap="mako",
                **kwargs,
            )
            plt.suptitle(f"GP Covariance Matrix for {feature}")
            plt.savefig(
                f"{RESULTS_PATH}/nonlinear/{'' if name is None else name}_gp_correlation_map_{feature}_{get_time_identifier()}.png"
            )
            plt.close()
            # plot corresponding colors on brain map
            if "Region" in color_indices.columns:
                plot_brain(
                    covariance_labels["Region"].unique(),
                    parc="HCP",
                    cbar=True,
                    cbartitle="Region Indices",
                    cmap=(
                        sns.color_palette(
                            "hls",
                            n_colors=len(color_indices["Region"].unique()),
                            as_cmap=True,
                        )
                        if len(color_indices["Region"].unique()) <= 20
                        else "mako"
                    ),
                    outfile=f"{RESULTS_PATH}/nonlinear/{'' if name is None else name}_gp_correlation_regions_{feature}_{get_time_identifier()}.png",
                    categorical=True,
                )
                plt.close()

    def _plot_gp_sensitivity(
        self,
        sensitivity_data: Dict[str, SensitivityOutput],
        name: str = None,
        **kwargs,
    ) -> None:
        data_dict = {"Component": [], "Sensitivity": [], "Feature": []}
        for feature, sensitivity in sensitivity_data.items():
            num_items = len(sensitivity.flatten())
            data_dict["Feature"].extend([feature] * num_items)
            data_dict["Component"].extend([i + 1 for i in range(num_items)])
            data_dict["Sensitivity"].extend(
                list(sensitivity.flatten() / np.max(sensitivity.flatten()))
            )
        data = pd.DataFrame(data=data_dict)
        p = sns.barplot(
            data,
            x="Component",
            y="Sensitivity",
            hue="Feature",
            **kwargs,
        )
        p.set_xlabel("Component")
        p.set_ylabel("Sensitivity")
        p.set_title(f"GP Sensitivity")
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{'' if name is None else name}_gp_sensitivity_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_gp_latents(
        self, latent_data: LatentOutput, name: str = None, **kwargs
    ) -> None:
        # get all possible combinations of components
        inputs = {}
        for x in latent_data.keys():
            for y in set(latent_data.keys()) - set([x]):
                if x not in inputs:
                    inputs[x] = set()
                # check for re-ordered duplicates
                if y not in inputs:
                    inputs[x].add(y)
            # prevent empty sets
            if len(inputs[x]) == 0:
                del inputs[x]
        # graph all possible combinations of 2-way components
        fig = plt.figure(figsize=(15, len(inputs) * 3))
        subfigs = fig.subfigures(nrows=len(inputs), ncols=1)
        for i, x_component in enumerate(inputs):
            axs = (
                subfigs[i].subplots(nrows=1, ncols=len(inputs[x_component]))
                if len(inputs) > 1
                else subfigs.subplots(nrows=1, ncols=len(inputs[x_component]))
            )
            for j, y_component in enumerate(inputs[x_component]):
                p = sns.scatterplot(
                    x=latent_data[x_component],
                    y=latent_data[y_component],
                    ax=axs[j] if len(inputs[x_component]) > 1 else axs,
                    **kwargs,
                )
                p.set_xlabel(x_component)
                p.set_ylabel(y_component)
        plt.suptitle("PCA Latents")
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{'' if name is None else name}_gp_latents_{get_time_identifier()}.png"
        )
        plt.close()
