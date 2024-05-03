import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from GPy.kern import RBF, Linear, White

import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mri_analysis.data import *
from mri_analysis.linear import ComponentAnalysis, RegressionAnalysis
from mri_analysis.nonlinear import GPAnalysis
from mri_analysis.visualizations import (
    DistributionPlotter,
    ClusterPlotter,
    LinearPlotter,
    NonlinearPlotter,
    BrainPlotter,
)

from mri_analysis.constants import RESULTS_PATH
from mri_analysis.datatypes import DATA_FEATURES
from mri_analysis.utils import get_time_identifier

from loguru import logger

logger.add(RESULTS_PATH + "/logs/" + get_time_identifier() + ".log")


def get_results(cfg: DictConfig):
    pass


if __name__ == "__main__":
    np.random.seed(42)

    ## get dataset
    logger.info("Loading dataset... ")
    dataset = load_dataset()

    ## get plotters
    logger.info("Creating plotters...")
    nonlinear_plotter = NonlinearPlotter(
        plots=[
            "gp_covariance",
            "gp_sensitivity",
        ]
    )

    ## get data
    logger.info("Initializing data...")
    # get averaged data
    averaged_data = dataset.get_data()
    # get total data
    total_data = dataset.get_data(average=False)
    # get region x subject data
    subject_data = dataset.get_data(average=False, pivot="subject")
    # get subject x region data
    region_data = dataset.get_data(average=False, pivot="label")

    ## get analyzers
    logger.info("Creating analyzers...")
    n_components = 10

    # normal data
    # logger.info("Fitting GP...")
    # rbf_kernel = (
    #     RBF(n_components, ARD=True)
    #     + Linear(n_components)
    #     + White(n_components)
    # )
    # base_gp = GPAnalysis(
    #     gp_type="Bayesian",
    #     kernels=rbf_kernel,
    #     use_mrd=False,
    #     name="ct_slrbf_bgp",
    # )
    # base_gp.fit(ct_data, n_components)
    # base_gp.save_model()
    # logger.info(base_gp.model)
    # logger.info(base_gp.model.kern.rbf.lengthscale)

    # base_gp.model.kern.plot_ARD()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/ct_lrbf_gp_ard_{get_time_identifier()}.png"
    # )
    # plt.close()
    # base_gp.model.plot_latent()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/ct_lrbf_gp_latent_{get_time_identifier()}.png"
    # )
    # plt.close()

    # expanded data
    # logger.info("Fitting expanded GP...")
    # rbf_kernel = (
    #     RBF(n_components, ARD=True)
    #     + Linear(n_components)
    #     + White(n_components)
    # )
    # expand_gp = GPAnalysis(
    #     gp_type="Bayesian",
    #     kernels=rbf_kernel,
    #     data_processing="expand",
    #     use_mrd=False,
    #     name="ct_expand_slrbf_bgp",
    # )
    # expand_gp.fit(subject_data, n_components, features=["CT"])
    # expand_gp.save_model()
    # logger.info(expand_gp.model)
    # logger.info(expand_gp.model.kern.rbf.lengthscale)

    # expand_gp.model.kern.plot_ARD()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/ct_expand_lrbf_gp_ard_{get_time_identifier()}.png"
    # )
    # plt.close()
    # expand_gp.model.plot_latent()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/ct_expand_lrbf_gp_latent_{get_time_identifier()}.png"
    # )
    # plt.close()

    # subject mrd
    # logger.info("Fitting subject MRD...")
    # rbf_kernel = (
    #     RBF(n_components, ARD=True)
    #     + Linear(n_components)
    #     + White(n_components)
    # )
    # subject_gp = GPAnalysis(
    #     gp_type="Bayesian", kernels=rbf_kernel, name="subject_mrd_lrbf_bgp"
    # )
    # subject_gp.fit(subject_data, n_components)
    # subject_gp.save_model()
    # logger.info(subject_gp.model)
    # for i, gp in enumerate(subject_gp.model.bgplvms):
    #     logger.info(f"Model {i}: {gp.kern.rbf.lengthscale}")

    # subject_gp.model.plot_scales()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/subject_mrd_ard_{get_time_identifier()}.png"
    # )
    # plt.close()
    # subject_gp.model.plot_latent()
    # plt.savefig(
    #     f"{RESULTS_PATH}/nonlinear/subject_mrd_latent_{get_time_identifier()}.png"
    # )
    # plt.close()

    # region mrd
    logger.info("Fitting region MRD...")
    for feature in DATA_FEATURES:
        logger.info(f"Fitting MRD for feature {feature}...")
        rbf_kernel = (
            RBF(n_components, ARD=True)
            + Linear(n_components)
            + White(n_components)
        )
        region_gp = GPAnalysis(
            gp_type="Bayesian",
            kernels=rbf_kernel,
            name=f"{feature}_region_mrd_lrbf_bgp",
        )
        region_gp.fit(
            region_data[feature],
            n_components,
            features=["L_1", "L_2", "L_3", "L_4", "L_5", "L_6", "L_7"],
        )
        region_gp.save_model()
        logger.info(region_gp.model)
        for i, gp in enumerate(region_gp.model.bgplvms):
            logger.info(f"Model {i}: {gp.kern.rbf.lengthscale}")

        region_gp.model.plot_scales()
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{feature}_region_mrd_ard_{get_time_identifier()}.png"
        )
        plt.close()
        region_gp.model.plot_latent()
        plt.savefig(
            f"{RESULTS_PATH}/nonlinear/{feature}_region_mrd_latent_{get_time_identifier()}.png"
        )
        plt.close()

        nonlinear_plotter.create_plots(
            covariance_data=region_gp.get_covariance(),
            covariance_labels=region_gp.labels,
            sensitivity_data=region_gp.get_sensitivity(),
            latent_data=region_gp.get_latents(),
            name=region_gp.name,
        )

    # get gp results
    logger.info("Plotting GP results...")
    # nonlinear_plotter.create_plots(
    #     covariance_data=base_gp.get_covariance(),
    #     covariance_labels=base_gp.labels,
    #     sensitivity_data=base_gp.get_sensitivity(),
    #     latent_data=base_gp.get_latents(),
    #     name=base_gp.name,
    # )
    # nonlinear_plotter.create_plots(
    #     covariance_data=expand_gp.get_covariance(),
    #     covariance_labels=expand_gp.labels,
    #     sensitivity_data=expand_gp.get_sensitivity(),
    #     latent_data=expand_gp.get_latents(),
    #     name=expand_gp.name,
    # )
    # nonlinear_plotter.create_plots(
    #     covariance_data=subject_gp.get_covariance(),
    #     covariance_labels=subject_gp.labels,
    #     sensitivity_data=subject_gp.get_sensitivity(),
    #     latent_data=subject_gp.get_latents(),
    #     name=subject_gp.name,
    # )
