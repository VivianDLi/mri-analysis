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

    # expanded data
    logger.info("Fitting expanded GP...")
    for burst_opt in [True, False]:
        for latent_type in ["pca", "random", "data"]:
            for inducing_type in ["permute", "random"]:
                if burst_opt:
                    for n_iters in [10, 50, 100, 500, 1000]:
                        logger.info(
                            f"Running burst {burst_opt} latent {latent_type} inducing {inducing_type} iters {n_iters}..."
                        )
                        rbf_kernel = (
                            RBF(n_components, ARD=True)
                            + Linear(n_components)
                            + White(n_components)
                        )
                        expand_gp = GPAnalysis(
                            gp_type="Bayesian",
                            kernels=rbf_kernel,
                            data_processing="expand",
                            latent_initialization=latent_type,
                            induced_initialization=inducing_type,
                            use_mrd=False,
                            burst_optimization=burst_opt,
                            n_restarts=n_iters,
                            name=f"fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}_bursts{n_iters}",
                        )
                        expand_gp.fit(
                            averaged_data,
                            n_components,
                        )
                        logger.info(expand_gp.model)
                        logger.info(expand_gp.model.kern.rbf.lengthscale)
                        expand_gp.model.kern.plot_ARD()
                        plt.savefig(
                            f"{RESULTS_PATH}/nonlinear/fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}_bursts{n_iters}_ard_{get_time_identifier()}.png"
                        )
                        plt.close()
                        expand_gp.model.plot_latent()
                        plt.savefig(
                            f"{RESULTS_PATH}/nonlinear/fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}_bursts{n_iters}_latent_{get_time_identifier()}.png"
                        )
                        plt.close()
                        nonlinear_plotter.create_plots(
                            covariance_data=expand_gp.get_covariance(),
                            covariance_labels=expand_gp.labels,
                            sensitivity_data=expand_gp.get_sensitivity(),
                            latent_data=expand_gp.get_latents(),
                            name=expand_gp.name,
                        )
                else:
                    logger.info(
                        f"Running burst {burst_opt} latent {latent_type} inducing {inducing_type}..."
                    )
                    rbf_kernel = (
                        RBF(n_components, ARD=True)
                        + Linear(n_components)
                        + White(n_components)
                    )
                    expand_gp = GPAnalysis(
                        gp_type="Bayesian",
                        kernels=rbf_kernel,
                        data_processing="expand",
                        latent_initialization=latent_type,
                        induced_initialization=inducing_type,
                        use_mrd=False,
                        burst_optimization=burst_opt,
                        name=f"fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}",
                    )
                    expand_gp.fit(
                        averaged_data,
                        n_components,
                    )
                    logger.info(expand_gp.model)
                    logger.info(expand_gp.model.kern.rbf.lengthscale)
                    expand_gp.model.kern.plot_ARD()
                    plt.savefig(
                        f"{RESULTS_PATH}/nonlinear/fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}_ard_{get_time_identifier()}.png"
                    )
                    plt.close()
                    expand_gp.model.plot_latent()
                    plt.savefig(
                        f"{RESULTS_PATH}/nonlinear/fixedkern_gtol0.5_burst{burst_opt}_latent{latent_type}_inducing{inducing_type}_latent_{get_time_identifier()}.png"
                    )
                    plt.close()
                    nonlinear_plotter.create_plots(
                        covariance_data=expand_gp.get_covariance(),
                        covariance_labels=expand_gp.labels,
                        sensitivity_data=expand_gp.get_sensitivity(),
                        latent_data=expand_gp.get_latents(),
                        name=expand_gp.name,
                    )
