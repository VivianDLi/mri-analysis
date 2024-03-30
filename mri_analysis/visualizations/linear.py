"""Plots linear decomposition information about data."""

from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from mri_analysis.constants import RESULTS_PATH
from mri_analysis.datatypes import (
    DATA_FEATURES,
    ComponentOutput,
    CovarianceOutput,
    ExplainedVarianceOutput,
    LatentOutput,
    LinearPlotType,
    PlotConfig,
)
from mri_analysis.utils import get_time_identifier

from loguru import logger


class LinearPlotter:
    plots: Dict[LinearPlotType, PlotConfig] = None

    def __init__(self, plots: List[LinearPlotType]):
        self.plots = {plot: {} for plot in plots}

    def set_plot_configs(
        self, configs: Dict[LinearPlotType, PlotConfig]
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
        covariance_data: CovarianceOutput = None,
        variance_data: List[ExplainedVarianceOutput] = None,
        component_data: ComponentOutput = None,
        latent_data: LatentOutput = None,
    ) -> None:
        logger.debug(f"Creating linear plots {self.plots}...")
        for plot, config in self.plots.items():
            # check for correct dataset
            if "covariance" in plot and covariance_data is None:
                logger.warning(
                    f"Covariance plot {plot} requires a numpy array to be passed as the <covariance_data> argument."
                )
                continue
            if "variance" in plot and variance_data is None:
                logger.warning(
                    f"Explained variance plot {plot} requires a list to be passed as the <variance_data> argument."
                )
                continue
            if "eigenvectors" in plot and component_data is None:
                logger.warning(
                    f"Eigenvector plot {plot} requires a dictionary to be passed as the <component_data> argument."
                )
                continue
            if "latents" in plot and latent_data is None:
                logger.warning(
                    f"Latent space plot {plot} requires a dictionary to be passed as the <latent_data> argument."
                )
                continue
            # plot
            match plot:
                case "pca_covariance":
                    self._plot_pca_covariance(covariance_data, **config)
                case "pca_variance":
                    self._plot_pca_variance(variance_data, **config)
                case "pca_eigenvectors":
                    self._plot_pca_eigenvectors(component_data, **config)
                case "pca_latents":
                    self._plot_pca_latents(latent_data, **config)
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_pca_covariance(
        self, covariance_data: CovarianceOutput, **kwargs
    ) -> None:
        p = sns.heatmap(
            data=covariance_data,
            cmap=sns.color_palette("mako", as_cmap=True),
            square=True,
            annot=True,
            fmt=".2f",
            **kwargs,
        )
        p.set_xticks(p.get_xticks(), DATA_FEATURES)
        p.set_yticks(p.get_yticks(), DATA_FEATURES)
        plt.xlabel("")
        plt.ylabel("")
        plt.suptitle("PCA Covariance")
        plt.savefig(
            f"{RESULTS_PATH}/linear/pca_covariance_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_pca_variance(
        self, variance_data: List[ExplainedVarianceOutput], **kwargs
    ) -> None:
        variances = []
        num_components = []
        for i, data in enumerate(variance_data):
            variances.append(data["total_variance"])
            num_components.append(i + 1)
        sns.lineplot(x=num_components, y=variances, **kwargs)
        plt.xlabel("Number of PCA Components")
        plt.ylabel("Percentage of Explained Variance")
        plt.title("PCA Explained Variance")
        plt.savefig(
            f"{RESULTS_PATH}/linear/pca_variance_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_pca_eigenvectors(
        self, component_data: ComponentOutput, **kwargs
    ) -> None:
        # get components per feature
        feature_components = {}
        for component_name in component_data:
            for i, feature in enumerate(DATA_FEATURES):
                if feature not in feature_components:
                    feature_components[feature] = []
                feature_components[feature].append(
                    component_data[component_name][i]
                )
        fig = plt.figure(figsize=(4 * len(component_data), 12))
        subfigures = fig.subfigures(nrows=2, ncols=1)
        c_axs = subfigures[0].subplots(
            nrows=1, ncols=len(component_data), sharey="row"
        )
        f_axs = subfigures[1].subplots(
            nrows=1, ncols=len(feature_components), sharey="row"
        )
        # plot components
        for i, (component_name, features) in enumerate(component_data.items()):
            p = sns.barplot(
                x=DATA_FEATURES,
                y=features,
                ax=c_axs[i],
                **kwargs,
            )
            p.set_xlabel("Feature")
            p.set_ylabel(component_name)
        # plot features
        for i, (feature_name, components) in enumerate(
            feature_components.items()
        ):
            p = sns.barplot(
                x=component_data.keys(),
                y=components,
                ax=f_axs[i],
                **kwargs,
            )
            p.set_xlabel("Component")
            p.set_ylabel(feature_name)
        fig.suptitle("PCA Eigenvectors")
        plt.savefig(
            f"{RESULTS_PATH}/linear/pca_eigenvectors_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_pca_latents(self, latent_data: LatentOutput, **kwargs) -> None:
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
            f"{RESULTS_PATH}/linear/pca_latents_{get_time_identifier()}.png"
        )
        plt.close()
