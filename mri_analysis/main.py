import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from GPy.kern import (
    RBF,
    Linear,
    White,
    Matern32,
    Exponential,
    Cosine,
    Sinc,
    Kern,
)
from GPy.examples.dimensionality_reduction import bgplvm_oil, brendan_faces

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
from mri_analysis.datatypes import DATA_COORD_FEATURES, DATA_FEATURES
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
    region_subset = None
    feature_subset = None

    # get averaged data
    averaged_data = dataset.get_data()
    # get region x subject data
    subject_data = dataset.get_data(
        average=False,
        pivot="subject",
        region_subset=region_subset,
        feature_subset=feature_subset,
    )
    # get subject x region data
    region_data = dataset.get_data(
        average=False,
        pivot="label",
        region_subset=region_subset,
        feature_subset=feature_subset,
    )

    ## get analyzers
    logger.info("Creating analyzers...")
    n_components = 10
    rbf_variance = 1.0
    linear_variance = 1.0

    latent_type = "pca"
    inducing_type = "permute"
    expand = False
    burst_opt = False
    fixed_opt = False
    spatial = False
    remove_components = None
    remove_variance = False

    n_samples = 360
    n_dim = 500

    # expanded data
    np.random.seed(42)

    logger.info("Fitting expanded GP...")
    rbf_kernel = (
        RBF(
            n_components,
            variance=rbf_variance,
            ARD=True,
        )
        + Linear(
            n_components,
            variances=np.ones(n_components) * linear_variance,
            ARD=True,
        )
        + White(n_components, variance=1e-4)
    )

    gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        data_processing=("expand" if expand else "none"),
        latent_initialization=latent_type,
        induced_initialization=inducing_type,
        use_mrd=False,
        remove_components=remove_components,
        burst_optimization=burst_opt,
        fixed_optimization=fixed_opt,
        name=f"subject_data",
    )
    # region_data = region_data.swaplevel(0, 1, axis="columns")
    gp.fit(
        subject_data,
        n_components,
        features=(DATA_FEATURES if not spatial else DATA_COORD_FEATURES),
    )
    # gp.fit(
    #     data,
    #     n_components,
    #     features=[str(i) for i in range(4)],
    # )
    gp.print_model_weights()
    # gp.plot_data()
    gp.plot_latent()
    if remove_variance:
        temp_name = gp.name
        gp.name = temp_name + "_nolinear"
        temp = gp.model.kern.linear.variances
        gp.model.kern.linear.variances = gp.model.kern.linear.variances * 0.0
        gp.plot_latent()
        gp.model.kern.linear.variances = temp
        gp.name = temp_name + "_norbf"
        gp.model.kern.rbf.variance = 0.0
        gp.plot_latent()
    gp.plot_scales()
    gp.plot_prediction()

    np.random.seed(42)
    logger.info("Fitting expanded GP 2...")
    rbf_kernel = (
        RBF(
            n_components,
            variance=rbf_variance,
            ARD=True,
        )
        + Linear(
            n_components,
            variances=np.ones(n_components) * linear_variance,
            ARD=True,
        )
        + White(n_components, variance=1e-4)
    )
    gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        data_processing=("expand" if expand else "none"),
        latent_initialization=latent_type,
        induced_initialization=inducing_type,
        use_mrd=True,
        remove_components=remove_components,
        burst_optimization=burst_opt,
        fixed_optimization=fixed_opt,
        name=f"subject_data",
    )
    gp.fit(
        subject_data,
        n_components,
        features=(DATA_FEATURES if not spatial else DATA_COORD_FEATURES),
    )
    gp.print_model_weights()
    gp.plot_latent()
    gp.plot_scales()
    gp.plot_prediction()

    # pca = ComponentAnalysis()
    # pca.fit(averaged_data, 4)
    # linear_plotter = LinearPlotter(plots=["pca_covariance"])
    # linear_plotter.create_plots(
    #     covariance_data=pca.get_covariance(), name="pca_covariance"
    # )

    # kernel_plotter = NonlinearPlotter(plots=["gp_covariance"])
    # result_dict = {}
    # X = gp.model.X.mean
    # result_dict["All"] = gp.model.kern.linear.K(X)
    # kernel_plotter.create_plots(
    #     covariance_data=result_dict,
    #     covariance_labels=gp.labels,
    #     name="linear_kern_covariance",
    # )

    # nonlinear_plotter.create_plots(
    #     covariance_data=gp.get_covariance(),
    #     covariance_labels=gp.labels,
    #     sensitivity_data=gp.get_sensitivity(),
    #     latent_data=gp.get_latents(),
    #     name=gp.get_name(),
    # )
