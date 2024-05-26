import numpy as np

from GPy.kern import RBF, Linear, White

import pathlib, sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from mri_analysis.data import *
from mri_analysis.linear import ComponentAnalysis, RegressionAnalysis
from mri_analysis.nonlinear import GPAnalysis
from mri_analysis.visualizations import (
    DistributionPlotter,
    LinearPlotter,
    NonlinearPlotter,
)

from mri_analysis.constants import RESULTS_PATH
from mri_analysis.utils import get_time_identifier

from loguru import logger

logger.add(RESULTS_PATH + "/logs/" + get_time_identifier() + ".log")

USE_SAVED_MODELS = False

if __name__ == "__main__":
    np.random.seed(42)

    ## get data
    logger.info("Loading dataset... ")
    dataset = load_dataset()

    # get averaged and total data
    averaged_data = dataset.get_data()
    total_data = dataset.get_data(average=False)
    # get region x subject data
    subject_data = dataset.get_data(
        average=False,
        pivot="subject",
        region_subset=None,
        feature_subset=None,
    )
    gender_data = subject_data.swaplevel(0, 1, axis="columns")
    # get subject x region data
    region_data = dataset.get_data(
        average=False,
        pivot="label",
        region_subset=None,
        feature_subset=None,
    )
    region_label_data = region_data.swaplevel(0, 1, axis="columns")
    # simulated data
    nonlinear_simulated_data_4 = generate_nonlinear_synthetic_data(1000, 4)
    nonlinear_simulated_data_100 = generate_nonlinear_synthetic_data(1000, 100)
    nonlinear_simulated_data_500 = generate_nonlinear_synthetic_data(1000, 500)
    combined_simulated_data = generate_combined_synthetic_data(1000, 500)

    ## get analyzers
    logger.info("Creating analyzers...")
    n_components = 10
    rbf_variance = 1.0
    linear_variance = 1.0
    kernel = (
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

    # linear analysis
    component_analysis = ComponentAnalysis()
    total_component_analysis = ComponentAnalysis()
    regression_analysis = RegressionAnalysis()
    total_regression_analysis = RegressionAnalysis()
    # simulated analysis
    nonlinear_simulated_gp_4 = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="nonlinear_simulated_4",
    )
    nonlinear_simulated_gp_100 = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="nonlinear_simulated_100",
    )
    nonlinear_simulated_gp_500 = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="nonlinear_simulated_500",
    )
    combined_simulated_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="combined_simulated",
    )
    combined_simulated_nonlinear_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=2,
        name="combined_simulated",
    )
    # subject analysis
    subject_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="subject",
    )
    subject_nonlinear_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=2,
        name="subject_nonlinear",
    )
    subject_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=None,
        name="subject_mrd",
    )
    subject_nonlinear_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=2,
        name="subject_nonlinear_mrd",
    )
    gender_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=None,
        name="gender_mrd",
    )
    gender_nonlinear_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=2,
        name="gender_nonlinear_mrd",
    )
    # region analysis
    region_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=None,
        name="region",
    )
    region_nonlinear_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=False,
        remove_components=2,
        name="region_nonlinear",
    )
    region_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=None,
        name="region_mrd",
    )
    region_nonlinear_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=2,
        name="region_nonlinear_mrd",
    )
    label_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=None,
        name="label_mrd",
    )
    label_nonlinear_mrd_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=kernel.copy(),
        use_mrd=True,
        remove_components=2,
        name="label_nonlinear_mrd",
    )

    ## train analyzers
    logger.info("Training analyzers...")
    # linear training
    component_analysis.fit(averaged_data, 4)
    total_component_analysis.fit(total_data, 4)
    regression_analysis.fit(averaged_data)
    total_regression_analysis.fit(total_data)
    # simulated training
    nonlinear_simulated_gp_4.fit(
        nonlinear_simulated_data_4,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[str(i) for i in range(4)],
    )
    nonlinear_simulated_gp_100.fit(
        nonlinear_simulated_data_100,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[str(i) for i in range(100)],
    )
    nonlinear_simulated_gp_500.fit(
        nonlinear_simulated_data_500,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[str(i) for i in range(500)],
    )
    combined_simulated_gp.fit(
        combined_simulated_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[str(i) for i in range(500)],
    )
    combined_simulated_nonlinear_gp.fit(
        combined_simulated_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[str(i) for i in range(500)],
    )
    # subject training
    subject_gp.fit(subject_data, n_components, optimize=not USE_SAVED_MODELS)
    subject_nonlinear_gp.fit(
        subject_data, n_components, optimize=not USE_SAVED_MODELS
    )
    subject_mrd_gp.fit(
        subject_data, n_components, optimize=not USE_SAVED_MODELS
    )
    subject_nonlinear_mrd_gp.fit(
        subject_data, n_components, optimize=not USE_SAVED_MODELS
    )
    gender_mrd_gp.fit(
        gender_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=["Female", "Male"],
    )
    gender_nonlinear_mrd_gp.fit(
        gender_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=["Female", "Male"],
    )
    # region training
    region_gp.fit(region_data, n_components, optimize=not USE_SAVED_MODELS)
    region_nonlinear_gp.fit(
        region_data, n_components, optimize=not USE_SAVED_MODELS
    )
    region_mrd_gp.fit(region_data, n_components, optimize=not USE_SAVED_MODELS)
    region_nonlinear_mrd_gp.fit(
        region_data, n_components, optimize=not USE_SAVED_MODELS
    )
    label_mrd_gp.fit(
        region_label_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[f"Label {i}" for i in range(1, 8)],
    )
    label_nonlinear_mrd_gp.fit(
        region_label_data,
        n_components,
        optimize=not USE_SAVED_MODELS,
        features=[f"Label {i}" for i in range(1, 8)],
    )

    ## loading analyzers
    if USE_SAVED_MODELS:
        logger.info("Loading saved analyzers...")
        # simulated loading
        nonlinear_simulated_gp_4.load_model_weights()
        nonlinear_simulated_gp_100.load_model_weights()
        nonlinear_simulated_gp_500.load_model_weights()
        combined_simulated_gp.load_model_weights()
        combined_simulated_nonlinear_gp.load_model_weights()
        # subject loading
        subject_gp.load_model_weights()
        subject_nonlinear_gp.load_model_weights()
        subject_mrd_gp.load_model_weights()
        subject_nonlinear_mrd_gp.load_model_weights()
        gender_mrd_gp.load_model_weights()
        gender_nonlinear_mrd_gp.load_model_weights()
        # region loading
        region_gp.load_model_weights()
        region_nonlinear_gp.load_model_weights()
        region_mrd_gp.load_model_weights()
        region_nonlinear_mrd_gp.load_model_weights()
        label_mrd_gp.load_model_weights()
        label_nonlinear_mrd_gp.load_model_weights()

    ## saving analyzers
    if not USE_SAVED_MODELS:
        logger.info("Saving analyzers...")
        # simulated saving
        nonlinear_simulated_gp_4.save_model()
        nonlinear_simulated_gp_100.save_model()
        nonlinear_simulated_gp_500.save_model()
        combined_simulated_gp.save_model()
        combined_simulated_nonlinear_gp.save_model()
        # subject saving
        subject_gp.save_model()
        subject_nonlinear_gp.save_model()
        subject_mrd_gp.save_model()
        subject_nonlinear_mrd_gp.save_model()
        gender_mrd_gp.save_model()
        gender_nonlinear_mrd_gp.save_model()
        # region saving
        region_gp.save_model()
        region_nonlinear_gp.save_model()
        region_mrd_gp.save_model()
        region_nonlinear_mrd_gp.save_model()
        label_mrd_gp.save_model()
        label_nonlinear_mrd_gp.save_model()

    ## create analyzer plotters
    logger.info("Creating plotters...")
    distribution_plotter = DistributionPlotter(plots=["feature_histogram"])
    linear_plotter = LinearPlotter(
        plots=[
            "pca_covariance",
            "pca_eigenvectors",
        ],
    )
    simulated_plotter = NonlinearPlotter(plots=["gp_sensitivity"])
    nonlinear_plotter = NonlinearPlotter(
        plots=[
            "gp_covariance",
            "gp_sensitivity",
            "gp_prediction",
        ]
    )
    nonlinear_brain_plotter = NonlinearPlotter(
        plots=[
            "gp_covariance_brain",
            "gp_covariance",
            "gp_sensitivity",
            "gp_prediction",
        ]
    )

    def plot_gp(model):
        model.print_model_weights()
        model.plot_latent()

    ## plot analyzers
    logger.info("Plotting analyzers...")
    # distribution plots
    distribution_plotter.create_plots(averaged_data)
    distribution_plotter.create_plots(total_data)
    # linear plots
    linear_plotter.create_plots(
        covariance_data=component_analysis.get_covariance(),
        variance_data=component_analysis.get_explained_variance(),
        component_data=component_analysis.get_components(),
        name="average",
    )
    linear_plotter.create_plots(
        covariance_data=total_component_analysis.get_covariance(),
        variance_data=total_component_analysis.get_explained_variance(),
        component_data=total_component_analysis.get_components(),
        name="total",
    )
    # simulated plots
    simulated_plotter.create_plots(
        sensitivity_data=nonlinear_simulated_gp_4.get_sensitivity(),
        name=nonlinear_simulated_gp_4.get_name(),
    )
    plot_gp(nonlinear_simulated_gp_4)
    simulated_plotter.create_plots(
        sensitivity_data=nonlinear_simulated_gp_100.get_sensitivity(),
        name=nonlinear_simulated_gp_100.get_name(),
    )
    plot_gp(nonlinear_simulated_gp_100)
    simulated_plotter.create_plots(
        sensitivity_data=nonlinear_simulated_gp_500.get_sensitivity(),
        name=nonlinear_simulated_gp_500.get_name(),
    )
    plot_gp(nonlinear_simulated_gp_500)
    simulated_plotter.create_plots(
        sensitivity_data=combined_simulated_gp.get_sensitivity(),
        name=combined_simulated_gp.get_name(),
    )
    plot_gp(combined_simulated_gp)
    simulated_plotter.create_plots(
        sensitivity_data=combined_simulated_nonlinear_gp.get_sensitivity(),
        name=combined_simulated_nonlinear_gp.get_name(),
    )
    plot_gp(combined_simulated_nonlinear_gp)
    # subject plots
    nonlinear_brain_plotter.create_plots(
        covariance_data=subject_gp.get_covariance(),
        covariance_labels=subject_gp.labels,
        sensitivity_data=subject_gp.get_sensitivity(),
        prediction_data=subject_gp.get_predictions(),
        name=subject_gp.get_name(),
    )
    plot_gp(subject_gp)
    nonlinear_brain_plotter.create_plots(
        covariance_data=subject_nonlinear_gp.get_covariance(),
        covariance_labels=subject_nonlinear_gp.labels,
        sensitivity_data=subject_nonlinear_gp.get_sensitivity(),
        prediction_data=subject_nonlinear_gp.get_predictions(),
        name=subject_nonlinear_gp.get_name(),
    )
    plot_gp(subject_nonlinear_gp)
    nonlinear_brain_plotter.create_plots(
        covariance_data=subject_mrd_gp.get_covariance(),
        covariance_labels=subject_mrd_gp.labels,
        sensitivity_data=subject_mrd_gp.get_sensitivity(),
        prediction_data=subject_mrd_gp.get_predictions(),
        name=subject_mrd_gp.get_name(),
    )
    plot_gp(subject_mrd_gp)
    nonlinear_brain_plotter.create_plots(
        covariance_data=subject_nonlinear_mrd_gp.get_covariance(),
        covariance_labels=subject_nonlinear_mrd_gp.labels,
        sensitivity_data=subject_nonlinear_mrd_gp.get_sensitivity(),
        prediction_data=subject_nonlinear_mrd_gp.get_predictions(),
        name=subject_nonlinear_mrd_gp.get_name(),
    )
    plot_gp(subject_nonlinear_mrd_gp)
    nonlinear_brain_plotter.create_plots(
        covariance_data=gender_mrd_gp.get_covariance(),
        covariance_labels=gender_mrd_gp.labels,
        sensitivity_data=gender_mrd_gp.get_sensitivity(),
        prediction_data=gender_mrd_gp.get_predictions(),
        name=gender_mrd_gp.get_name(),
    )
    plot_gp(gender_mrd_gp)
    nonlinear_brain_plotter.create_plots(
        covariance_data=gender_nonlinear_mrd_gp.get_covariance(),
        covariance_labels=gender_nonlinear_mrd_gp.labels,
        sensitivity_data=gender_nonlinear_mrd_gp.get_sensitivity(),
        prediction_data=gender_nonlinear_mrd_gp.get_predictions(),
        name=gender_nonlinear_mrd_gp.get_name(),
    )
    plot_gp(gender_nonlinear_mrd_gp)
    # region plots
    nonlinear_plotter.create_plots(
        covariance_data=region_gp.get_covariance(),
        covariance_labels=region_gp.labels,
        sensitivity_data=region_gp.get_sensitivity(),
        prediction_data=region_gp.get_predictions(),
        name=region_gp.get_name(),
    )
    plot_gp(region_gp)
    nonlinear_plotter.create_plots(
        covariance_data=region_nonlinear_gp.get_covariance(),
        covariance_labels=region_nonlinear_gp.labels,
        sensitivity_data=region_nonlinear_gp.get_sensitivity(),
        prediction_data=region_nonlinear_gp.get_predictions(),
        name=region_nonlinear_gp.get_name(),
    )
    plot_gp(region_nonlinear_gp)
    nonlinear_plotter.create_plots(
        covariance_data=region_mrd_gp.get_covariance(),
        covariance_labels=region_mrd_gp.labels,
        sensitivity_data=region_mrd_gp.get_sensitivity(),
        prediction_data=region_mrd_gp.get_predictions(),
        name=region_mrd_gp.get_name(),
    )
    plot_gp(region_mrd_gp)
    nonlinear_plotter.create_plots(
        covariance_data=region_nonlinear_mrd_gp.get_covariance(),
        covariance_labels=region_nonlinear_mrd_gp.labels,
        sensitivity_data=region_nonlinear_mrd_gp.get_sensitivity(),
        prediction_data=region_nonlinear_mrd_gp.get_predictions(),
        name=region_nonlinear_mrd_gp.get_name(),
    )
    plot_gp(region_nonlinear_mrd_gp)
    nonlinear_plotter.create_plots(
        covariance_data=label_mrd_gp.get_covariance(),
        covariance_labels=label_mrd_gp.labels,
        sensitivity_data=label_mrd_gp.get_sensitivity(),
        prediction_data=label_mrd_gp.get_predictions(),
        name=label_mrd_gp.get_name(),
    )
    plot_gp(label_mrd_gp)
    nonlinear_plotter.create_plots(
        covariance_data=label_nonlinear_mrd_gp.get_covariance(),
        covariance_labels=label_nonlinear_mrd_gp.labels,
        sensitivity_data=label_nonlinear_mrd_gp.get_sensitivity(),
        prediction_data=label_nonlinear_mrd_gp.get_predictions(),
        name=label_nonlinear_mrd_gp.get_name(),
    )
    plot_gp(label_nonlinear_mrd_gp)
