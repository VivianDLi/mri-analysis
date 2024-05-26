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
    PredictionOutput,
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
        covariance_data: CovarianceOutput = None,
        covariance_labels: pd.DataFrame = None,
        sensitivity_data: SensitivityOutput = None,
        latent_data: LatentOutput = None,
        prediction_data: PredictionOutput = None,
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
            if "brain" in plot and ("Region" not in covariance_labels.columns):
                logger.warning(
                    f"Brain plot {plot} requires a dataframe to be passed as the <covariance_labels> argument with a 'Region' column."
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
            if "prediction" in plot and prediction_data is None:
                logger.warning(
                    f"Prediction plot {plot} requires a dictionary to be passed as the <prediction_data> argument."
                )
                continue
            # plot
            match plot:
                case "gp_covariance":
                    self._plot_gp_covariance(
                        covariance_data, covariance_labels, name=name, **config
                    )
                case "gp_covariance_brain":
                    self._plot_gp_covariance_brain(
                        covariance_data, covariance_labels, name=name, **config
                    )
                case "gp_sensitivity":
                    self._plot_gp_sensitivity(
                        sensitivity_data, name=name, **config
                    )
                case "gp_latents":
                    self._plot_gp_latents(latent_data, name=name, **config)
                case "gp_prediction":
                    self._plot_gp_prediction(
                        prediction_data, name=name, **config
                    )
                case _:
                    logger.warning(
                        f"Plot {plot} not found in available plots."
                    )
                    continue

    def _plot_gp_covariance(
        self,
        covariance_data: CovarianceOutput,
        covariance_labels: pd.DataFrame,
        name: str = None,
        **kwargs,
    ) -> None:
        color_indices = covariance_labels.copy()
        labels = []
        for label in color_indices.columns:
            match label:
                case "Labels":
                    color_indices.sort_values(by=[label], inplace=True)
                    cmap = sns.color_palette(
                        "mako", n_colors=len(color_indices[label].unique())
                    )
                    lut = dict(zip(color_indices[label].unique(), cmap))
                    color_indices[label] = color_indices[label].map(lut)
                    labels.append(label)
                case "Sex":
                    color_indices.sort_values(by=["Sex", "Age"], inplace=True)
                    cmap = sns.color_palette(
                        "mako", n_colors=len(color_indices[label].unique())
                    )
                    lut = dict(zip(color_indices[label].unique(), cmap))
                    color_indices[label] = color_indices[label].map(lut)
                    labels.append(label)
                case "Age":
                    color_indices.sort_values(by=["Sex", "Age"], inplace=True)
                    cmap = sns.color_palette(
                        "mako", n_colors=len(color_indices[label].unique())
                    )
                    lut = dict(zip(color_indices[label].unique(), cmap))
                    color_indices[label] = color_indices[label].map(lut)
                    labels.append(label)
                case _:
                    continue
        for kernel_type in covariance_data:
            if len(covariance_data[kernel_type]) == 1:
                # non-MRD model
                df_covariance = pd.DataFrame(
                    covariance_data[kernel_type]["All"]
                )
                indices = color_indices.reset_index(drop=True)[labels]
                p = sns.clustermap(
                    df_covariance,
                    row_colors=indices,
                    col_colors=indices,
                    row_cluster=False,
                    col_cluster=False,
                    cmap="mako",
                    cbar_pos=(0.05, 0.6, 0.05, 0.18),
                    figsize=(8, 8),
                    **kwargs,
                )
                p.figure.suptitle(f"Covariance Matrix for {kernel_type}")
                plt.savefig(
                    f"{RESULTS_PATH}/nonlinear/{kernel_type}_correlation_{'' if name is None else name}_{get_time_identifier()}.png"
                )
                plt.close()
            else:
                # MRD model
                for view, covariance in covariance_data[kernel_type].items():
                    df_covariance = pd.DataFrame(covariance)
                    indices = color_indices.reset_index(drop=True)[labels]
                    p = sns.clustermap(
                        df_covariance,
                        row_colors=indices,
                        col_colors=indices,
                        row_cluster=False,
                        col_cluster=False,
                        cmap="mako",
                        cbar_pos=(0.05, 0.6, 0.05, 0.18),
                        figsize=(8, 8),
                        **kwargs,
                    )
                    p.figure.suptitle(
                        f"Covariance Matrix for {kernel_type}: {view}"
                    )
                    plt.savefig(
                        f"{RESULTS_PATH}/nonlinear/{kernel_type}_correlation_{view}_{'' if name is None else name}_{get_time_identifier()}.png"
                    )
                    plt.close()

    def _plot_gp_covariance_brain(
        self,
        covariance_data: CovarianceOutput,
        covariance_labels: pd.DataFrame,
        name: str = None,
        **kwargs,
    ) -> None:
        # initialize brain plotting
        import sys

        sys.path.insert(1, BRAINPLOT_PATH)
        from PlotBrains import plot_brain

        sorted_indices = (
            covariance_labels["Region Index"].to_numpy().argsort().flatten()
        )
        for kernel_type in covariance_data:
            if len(covariance_data[kernel_type]) == 1:
                # non-MRD model
                sorted_covariance = covariance_data[kernel_type]["All"][
                    sorted_indices
                ][:, sorted_indices]
                mean_covariance = np.mean(sorted_covariance, axis=0)
                region_covariances = {
                    region: sorted_covariance[i]
                    for i, region in enumerate(covariance_labels["Region"])
                }
                # plot average covariance per region
                plot_brain(
                    mean_covariance,
                    parc="HCP",
                    cbar=True,
                    cbartitle=f"Average {kernel_type} Covariance",
                    outfile=f"{RESULTS_PATH}/brain/{kernel_type}_brain_correlation_{'' if name is None else name}_{get_time_identifier()}.png",
                )
                plt.close()
                # plot relative covariances per region
                for region, region_covariance in region_covariances.items():
                    plot_brain(
                        region_covariance,
                        parc="HCP",
                        cbar=True,
                        cbartitle=f"Relative {kernel_type} Covariance for {region}",
                        outfile=f"{RESULTS_PATH}/brain/{kernel_type}_brain_correlation_{region}_{'' if name is None else name}_{get_time_identifier()}.png",
                    )
                    plt.close()
            else:
                # MRD model
                for view, covariance in covariance_data[kernel_type].items():
                    sorted_covariance = covariance[sorted_indices][
                        :, sorted_indices
                    ]
                    mean_covariance = np.mean(sorted_covariance, axis=0)
                    region_covariances = {
                        region: sorted_covariance[i]
                        for i, region in enumerate(covariance_labels["Region"])
                    }
                    # plot average covariance per region
                    plot_brain(
                        mean_covariance,
                        parc="HCP",
                        cbar=True,
                        cbartitle=f"Average {kernel_type}: {view} Covariance",
                        outfile=f"{RESULTS_PATH}/brain/{kernel_type}_brain_correlation_{view}_{'' if name is None else name}_{get_time_identifier()}.png",
                    )
                    plt.close()
                    # plot relative covariances per region
                    for (
                        region,
                        region_covariance,
                    ) in region_covariances.items():
                        plot_brain(
                            region_covariance,
                            parc="HCP",
                            cbar=True,
                            cbartitle=f"Relative {kernel_type}: {view} Covariance for {region}",
                            outfile=f"{RESULTS_PATH}/brain/{kernel_type}_brain_correlation_{region}_{view}_{'' if name is None else name}_{get_time_identifier()}.png",
                        )
                        plt.close()

    def _plot_gp_sensitivity(
        self,
        sensitivity_data: SensitivityOutput,
        name: str = None,
        **kwargs,
    ) -> None:
        palettes = ["deep", "muted", "pastel", "dark", "bright"]
        f, ax = plt.subplots(
            figsize=(
                5 * len(sensitivity_data[list(sensitivity_data.keys())[0]]),
                8 * len(sensitivity_data),
            )
        )
        for i, kernel_type in enumerate(sensitivity_data):
            sns.set_color_codes(palettes[i % len(palettes)])
            if len(sensitivity_data[kernel_type]) == 1:
                # non-MRD model
                data = pd.DataFrame(
                    data={
                        "Component": [
                            i + 1
                            for i in range(
                                len(sensitivity_data[kernel_type]["All"])
                            )
                        ],
                        "Sensitivity": sensitivity_data[kernel_type]["All"],
                    }
                )
                sns.barplot(
                    x="Component",
                    y="Sensitivity",
                    data=data,
                    label=kernel_type,
                    **kwargs,
                )
                ax.legend(ncol=2, loc="upper right", frameon=True)
            else:
                # MRD model
                data_dict = {"Component": [], "Sensitivity": [], "View": []}
                for view, sensitivity in sensitivity_data[kernel_type].items():
                    data_dict["View"].extend([view] * len(sensitivity))
                    data_dict["Component"].extend(
                        [i + 1 for i in range(len(sensitivity))]
                    )
                    data_dict["Sensitivity"].extend(
                        list(
                            sensitivity.flatten()
                            / np.max(sensitivity.flatten())
                        )
                    )
                data = pd.DataFrame(data_dict)
                sns.barplot(
                    x="Component",
                    y="Sensitivity",
                    hue="View",
                    data=data,
                    legend="auto" if i == 0 else False,
                    **kwargs,
                )
        ax.set(
            xlabel="Latent Dimension",
            ylabel="Relative Sensitivity",
            title="Sensitivity Plot",
        )
        f.savefig(
            f"{RESULTS_PATH}/nonlinear/sensitivity_{'' if name is None else name}_{get_time_identifier()}.png"
        )
        plt.close()
        sns.set_color_codes()

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
            f"{RESULTS_PATH}/nonlinear/latents_{'' if name is None else name}_{get_time_identifier()}.png"
        )
        plt.close()

    def _plot_gp_prediction(
        self, prediction_data: PredictionOutput, name: str = None, **kwargs
    ) -> None:
        for latent_component in prediction_data:
            for prediction_type in prediction_data[latent_component]:
                predictions = prediction_data[latent_component][
                    prediction_type
                ]
                match prediction_type:
                    case "Average":
                        # setup dataframe
                        data_dict = {"Latent Value": []}
                        for i, feat in enumerate(predictions):
                            data_dict[feat] = []
                            # check for MRD model
                            if isinstance(predictions[feat], dict):
                                data_dict["View"] = []
                                for view, prediction in predictions[
                                    feat
                                ].items():
                                    data_dict[feat].extend(
                                        prediction.flatten()
                                    )
                                    data_dict["View"].extend(
                                        [view] * len(prediction.flatten())
                                    )
                                data_dict["Latent Value"] = np.repeat(
                                    np.linspace(-5, 5, 20),
                                    len(data_dict[feat]) // 20,
                                )
                            else:
                                data_dict["Latent Value"] = np.repeat(
                                    np.linspace(-5, 5, 20),
                                    predictions[feat].shape[1],
                                )
                                data_dict[feat].extend(
                                    predictions[feat].flatten()
                                )
                        data_df = pd.DataFrame(data_dict)
                        # plot average prediction
                        f, axes = plt.subplots(
                            len(predictions),
                            1,
                            figsize=(4 * len(predictions), 12),
                        )
                        for i, feat in enumerate(predictions):
                            sns.lineplot(
                                x="Latent Value",
                                y=feat,
                                data=data_df,
                                ax=axes[i],
                                hue=(
                                    "View"
                                    if "View" in data_df.columns
                                    else None
                                ),
                            )
                            axes[i].set(
                                xlabel=f"Latent Dimension {latent_component}",
                                ylabel=f"{feat}",
                                title="",
                            )
                        f.suptitle(
                            f"Averaged Predictions for Latent Component {latent_component}"
                        )
                        f.savefig(
                            f"{RESULTS_PATH}/nonlinear/average_prediction_{latent_component}_{'' if name is None else name}_{get_time_identifier()}.png"
                        )
                        plt.close()
                    case _:
                        for data_point in predictions:
                            # setup dataframe
                            data_dict = {"Latent Value": []}
                            for i, feat in enumerate(predictions[data_point]):
                                data_dict[feat] = []
                                # check for MRD model
                                if isinstance(
                                    predictions[data_point][feat], dict
                                ):
                                    data_dict["View"] = []
                                    for view, prediction in predictions[
                                        data_point
                                    ][feat].items():
                                        data_dict[feat].extend(
                                            prediction.flatten()
                                        )
                                        data_dict["View"].extend(
                                            [view] * len(prediction.flatten())
                                        )
                                    data_dict["Latent Value"] = np.repeat(
                                        np.linspace(-5, 5, 20),
                                        len(data_dict[feat]) // 20,
                                    )
                                else:
                                    data_dict["Latent Value"] = np.repeat(
                                        np.linspace(-5, 5, 20),
                                        predictions[data_point][feat].shape[1],
                                    )
                                    data_dict[feat].extend(
                                        predictions[data_point][feat].flatten()
                                    )
                            data_df = pd.DataFrame(data_dict)
                            # plot individual predictions per region/subject
                            f, axes = plt.subplots(
                                len(predictions[data_point]),
                                1,
                                figsize=(4 * len(predictions[data_point]), 12),
                            )
                            for i, feat in enumerate(predictions[data_point]):
                                sns.lineplot(
                                    x="Latent Value",
                                    y=feat,
                                    data=data_df,
                                    ax=axes[i],
                                    hue=(
                                        "View"
                                        if "View" in data_df.columns
                                        else None
                                    ),
                                )
                                axes[i].set(
                                    xlabel=f"Latent Dimension {latent_component}",
                                    ylabel=f"{feat}",
                                    title="",
                                )
                            f.suptitle(
                                f"{data_point} Predictions for Latent Component {latent_component}"
                            )
                            f.savefig(
                                f"{RESULTS_PATH}/nonlinear/{data_point}_prediction_{latent_component}_{'' if name is None else name}_{get_time_identifier()}.png"
                            )
                            plt.close()
