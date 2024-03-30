import numpy as np
from omegaconf import DictConfig

from GPy.kern import RBF, Linear

import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mri_analysis.data import load_dataset
from mri_analysis.linear import ComponentAnalysis, RegressionAnalysis
from mri_analysis.nonlinear import GPAnalysis
from mri_analysis.visualizations import (
    DistributionPlotter,
    ClusterPlotter,
    LinearPlotter,
    NonlinearPlotter,
    BrainPlotter,
)

from loguru import logger


def get_results(cfg: DictConfig):
    pass


if __name__ == "__main__":
    np.random.seed(42)

    # get dataset
    logger.info("Loading dataset... ")
    dataset = load_dataset()

    # get analyzers
    logger.info("Creating analyzers...")
    pca = ComponentAnalysis()
    regression = RegressionAnalysis()
    linear_kernel = Linear(2, ARD=True)
    rbf_kernel = RBF(2, ARD=True)
    linear_gp = GPAnalysis(
        kernels=linear_kernel, expand_dims=False, name="linear_gp"
    )
    linear_expand_gp = GPAnalysis(
        kernels=linear_kernel, expand_dims=True, name="linear_expand_gp"
    )
    rbf_gp = GPAnalysis(kernels=rbf_kernel, expand_dims=False, name="rbf_gp")
    rbf_expand_gp = GPAnalysis(
        kernels=rbf_kernel, expand_dims=True, name="rbf_expand_gp"
    )

    # get plotters
    logger.info("Creating plotters...")
    distribution_plotter = DistributionPlotter(
        plots=[
            "feature_histogram",
            "feature_regression",
            "feature_scatter",
            "feature_strip",
            "regression_histogram",
        ]
    )
    cluster_plotter = ClusterPlotter(plots=["cluster_scatter", "cluster_map"])
    linear_plotter = LinearPlotter(
        plots=[
            "pca_covariance",
            "pca_eigenvectors",
            "pca_variance",
            "pca_latents",
        ]
    )
    nonlinear_plotter = NonlinearPlotter(
        plots=["gp_covariance", "gp_sensitivity", "gp_latents"]
    )
    brain_plotter = BrainPlotter(plots=["brain_feature", "brain_regression"])

    ## get data
    logger.info("Initializing data...")
    # get averaged data
    averaged_data = dataset.get_data()
    averaged_flat_data = dataset.get_data(flatten=True)
    # get subset data (subset 20)
    subset_data = dataset.get_data(subset=20, average=False)
    subset_flat_data = dataset.get_data(subset=20, average=False, flatten=True)
    # get total data
    total_data = dataset.get_data(average=False)
    total_flat_data = dataset.get_data(average=False, flatten=True)

    ## get distribution results
    logger.info("Plotting distribution results...")
    regression.fit(total_data)
    distribution_plotter.create_plots(
        regression_data=regression.get_regressions(), data=total_data
    )

    ## get cluster results
    logger.info("Plotting cluster results...")
    # averaged data
    logger.info("Plotting averaged clusters...")
    cluster_plotter.create_plots(
        data=averaged_data, flat_data=averaged_flat_data
    )
    # total data
    logger.info("Plotting total clusters...")
    cluster_plotter.create_plots(data=total_data, flat_data=total_flat_data)

    ## get pca results
    logger.info("Plotting PCA results...")
    # averaged data
    logger.info("Plotting averaged PCA...")
    variances = []
    for i in range(2, 4):
        pca.fit(averaged_data, n_components=i)
        variances.append(pca.get_explained_variance())
        linear_plotter.create_plots(
            covariance_data=pca.get_covariance(),
            variance_data=variances,
            component_data=pca.get_components(),
            latent_data=pca.get_latents(),
        )
    # total data
    logger.info("Plotting total PCA...")
    variances = []
    for i in range(2, 4):
        pca.fit(total_data, n_components=i)
        variances.append(pca.get_explained_variance())
        linear_plotter.create_plots(
            covariance_data=pca.get_covariance(),
            variance_data=variances,
            component_data=pca.get_components(),
            latent_data=pca.get_latents(),
        )

    ## get gp results
    logger.info("Plotting GP results for averaged dataset...")
    # flat results
    logger.info("Plotting flat GP results...")
    logger.info("Linear GP results...")
    linear_gp.fit(averaged_flat_data, 2, flat_data=True)
    linear_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=linear_gp.get_covariance(),
        covariance_labels=linear_gp.labels,
        sensitivity_data=linear_gp.get_sensitivity(),
        latent_data=linear_gp.get_latents(),
    )
    logger.info("RBF GP results...")
    rbf_gp.fit(averaged_flat_data, 2, flat_data=True)
    rbf_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=rbf_gp.get_covariance(),
        covariance_labels=rbf_gp.labels,
        sensitivity_data=rbf_gp.get_sensitivity(),
        latent_data=rbf_gp.get_latents(),
    )
    # md results
    logger.info("Plotting MD GP results...")
    logger.info("Linear GP results...")
    linear_gp.fit(averaged_data, 2)
    linear_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=linear_gp.get_covariance(),
        covariance_labels=linear_gp.labels,
        sensitivity_data=linear_gp.get_sensitivity(),
        latent_data=linear_gp.get_latents(),
    )
    logger.info("RBF GP results...")
    rbf_gp.fit(averaged_data, 2)
    rbf_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=rbf_gp.get_covariance(),
        covariance_labels=rbf_gp.labels,
        sensitivity_data=rbf_gp.get_sensitivity(),
        latent_data=rbf_gp.get_latents(),
    )
    nonlinear_plotter.plots = ["gp_covariance", "gp_sensitivity", "gp_latents"]
    logger.info("Linear Expanded GP results...")
    linear_expand_gp.fit(averaged_data, 2)
    linear_expand_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=linear_expand_gp.get_covariance(),
        covariance_labels=linear_expand_gp.labels,
        sensitivity_data=linear_expand_gp.get_sensitivity(),
        latent_data=linear_expand_gp.get_latents(),
    )
    logger.info("RBF Expanded GP results...")
    rbf_expand_gp.fit(averaged_data, 2)
    rbf_expand_gp.save_model()
    nonlinear_plotter.create_plots(
        covariance_data=rbf_expand_gp.get_covariance(),
        covariance_labels=rbf_expand_gp.labels,
        sensitivity_data=rbf_expand_gp.get_sensitivity(),
        latent_data=rbf_expand_gp.get_latents(),
    )

    ## get brain results
    logger.info("Plotting brain results...")
    regression.fit(total_data)
    brain_plotter.create_plots(
        data=averaged_data, regression_data=regression.get_regressions()
    )
