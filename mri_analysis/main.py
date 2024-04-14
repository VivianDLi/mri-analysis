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

from mri_analysis.datatypes import DATA_FEATURES

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

    # linear_gp = GPAnalysis(
    #     kernels=linear_kernel, expand_dims=False, name="linear_gp"
    # )
    # linear_expand_gp = GPAnalysis(
    #     kernels=linear_kernel, expand_dims=True, name="linear_expand_gp"
    # )
    variance = 1.0
    lengthscale = 1.0
    n_components = 20
    rbf_kernel = RBF(
        n_components, variance=variance, lengthscale=lengthscale, ARD=True
    )
    rbf_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        expand_dims=False,
        name="hd_rbf_bgp",
    )
    rbf_expand_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        expand_dims=True,
        name="hd_expand_rbf_bgp",
    )
    full_rbf_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        expand_dims=False,
        name="full_hd_rbf_bgp",
    )
    full_rbf_expand_gp = GPAnalysis(
        gp_type="Bayesian",
        kernels=rbf_kernel,
        expand_dims=True,
        name="full_hd_expand_rbf_bgp",
    )
    # rbf_md_gp = GPAnalysis(
    #     gp_type="Bayesian",
    #     kernels=rbf_kernel,
    #     expand_dims=False,
    #     multi_dimensional=True,
    #     name="hd_md_rbf_bgp",
    # )
    # rbf_md_expand_gp = GPAnalysis(
    #     gp_type="Bayesian"
    #     kernels=rbf_kernel,
    #     expand_dims=True,
    #     multi_dimensional=True,
    #     name="hd_md_expand_rbf_bgp",
    # )

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
    averaged_data = dataset.get_data(normalize=False)
    averaged_flat_data = dataset.get_data(flatten=True)
    # get subset data (subset 20)
    subset_data = dataset.get_data(subset=20, average=False)
    subset_flat_data = dataset.get_data(subset=20, average=False, flatten=True)
    # get total data
    total_data = dataset.get_data(normalize=False, average=False)
    total_flat_data = dataset.get_data(average=False, flatten=True)

    import GPy
    import matplotlib.pyplot as plt

    normal_data = averaged_data[DATA_FEATURES].to_numpy()
    expanded_data = np.matmul(
        normal_data.copy(), np.random.randn(normal_data.shape[1], 100)
    ) + 1e-6 * np.random.randn(normal_data.shape[0], 100)

    normal_list = [
        np.reshape(averaged_data[feature].to_numpy(), (-1, 1))
        for feature in DATA_FEATURES
    ]
    expanded_list = [
        np.matmul(Y.copy(), np.random.randn(Y.shape[1], 100))
        + 1e-6 * np.random.randn(Y.shape[0], 100)
        for Y in normal_list
    ]
    subject_df = total_data.pivot_table(
        values=DATA_FEATURES, index="Region", columns="Subject"
    )
    subject_list = [
        subject_df[feature].to_numpy() for feature in DATA_FEATURES
    ]

    # input_dim = 10
    # print("normal_data")
    # kernel = RBF(input_dim, ARD=True)
    # kernel += GPy.kern.White(input_dim)
    # m = GPy.models.BayesianGPLVM(
    #     normal_data,
    #     input_dim,
    #     kernel=kernel,
    #     num_inducing=25,
    # )
    # m.likelihood.variance = m.Y.var() / 100.0
    # m.optimize("bfgs", messages=1, max_f_eval=10000, max_iters=10000)
    # print(m)
    # print(m.kern.rbf.lengthscale)
    # m.kern.plot_ARD()
    # plt.savefig("normal_data_ard.png")
    # plt.close()
    # m.plot_latent()
    # plt.savefig("normal_data_latent.png")
    # plt.close()

    # print("expanded_data")
    # kernel = RBF(input_dim, ARD=True)
    # kernel += GPy.kern.White(input_dim)
    # m = GPy.models.BayesianGPLVM(
    #     expanded_data,
    #     input_dim,
    #     kernel=kernel,
    #     num_inducing=25,
    # )
    # m.likelihood.variance = m.Y.var() / 100.0
    # m.optimize("bfgs", messages=1, max_f_eval=10000, max_iters=10000)
    # print(m)
    # print(m.kern.rbf.lengthscale)
    # m.kern.plot_ARD()
    # plt.savefig("expand_data_ard.png")
    # plt.close()
    # m.plot_latent()
    # plt.savefig("expand_data_latent.png")
    # plt.close()

    # print("normal_list")
    # kernels = [
    #     RBF(input_dim, ARD=True) + GPy.kern.White(input_dim)
    #     for _ in range(len(DATA_FEATURES))
    # ]
    # m = GPy.models.MRD(normal_list, input_dim, kernel=kernels, num_inducing=25)
    # m.optimize("bfgs", messages=1, max_iters=10000)
    # print(m)
    # print(m.kern.rbf.lengthscale)
    # m.plot_scales()
    # plt.savefig("normal_md_ard.png")
    # plt.close()
    # m.X.plot()
    # plt.savefig("normal_md_latent.png")
    # plt.close()

    # print("expanded_list")
    # kernels = [
    #     RBF(input_dim, ARD=True) + GPy.kern.White(input_dim)
    #     for _ in range(len(DATA_FEATURES))
    # ]
    # m = GPy.models.MRD(
    #     expanded_list, input_dim, kernel=kernels, num_inducing=25
    # )
    # m.optimize("bfgs", messages=1, max_iters=10000)
    # print(m)
    # print(m.kern.rbf.lengthscale)
    # m.plot_scales()
    # plt.savefig("expand_md_ard.png")
    # plt.close()
    # m.X.plot()
    # plt.savefig("expand_md_latent.png")
    # plt.close()

    # print("subject_list")
    # kernels = [
    #     RBF(input_dim, ARD=True) + GPy.kern.White(input_dim)
    #     for _ in range(len(DATA_FEATURES))
    # ]
    # m = GPy.models.MRD(
    #     subject_list, input_dim, kernel=kernels, num_inducing=25
    # )
    # m.optimize("bfgs", messages=1, max_iters=10000)
    # print(m)
    # print(m.kern.rbf.lengthscale)
    # m.plot_scales()
    # plt.savefig("subject_md_ard.png")
    # plt.close()
    # m.X.plot()
    # plt.savefig("subject_md_latent.png")
    # plt.close()

    ## get region-specific results
    # for region, group in total_data.groupby("Region"):
    #     full_rbf_expand_gp = GPAnalysis(
    #         gp_type="Bayesian",
    #         kernels=rbf_kernel,
    #         expand_dims=True,
    #         name=f"{region}_hd_expand_rbf_bgp",
    #     )
    #     full_rbf_expand_gp.fit(group, n_components)
    #     full_rbf_expand_gp.save_model()
    #     nonlinear_plotter.create_plots(
    #         covariance_data=full_rbf_expand_gp.get_covariance(),
    #         covariance_labels=full_rbf_expand_gp.labels,
    #         sensitivity_data=full_rbf_expand_gp.get_sensitivity(),
    #         latent_data=full_rbf_expand_gp.get_latents(),
    #     )

    # ## get distribution results
    # logger.info("Plotting distribution results...")
    # regression.fit(total_data)
    # distribution_plotter.create_plots(
    #     regression_data=regression.get_regressions(), data=total_data
    # )

    # ## get cluster results
    # logger.info("Plotting cluster results...")
    # # averaged data
    # logger.info("Plotting averaged clusters...")
    # cluster_plotter.create_plots(
    #     data=averaged_data, flat_data=averaged_flat_data
    # )
    # # total data
    # logger.info("Plotting total clusters...")
    # cluster_plotter.create_plots(data=total_data, flat_data=total_flat_data)

    # ## get pca results
    # logger.info("Plotting PCA results...")
    # # averaged data
    # logger.info("Plotting averaged PCA...")
    # variances = []
    # for i in range(2, 5):
    #     pca.fit(averaged_data, n_components=i)
    #     variances.append(pca.get_explained_variance())
    #     linear_plotter.create_plots(
    #         covariance_data=pca.get_covariance(),
    #         variance_data=variances,
    #         component_data=pca.get_components(),
    #         latent_data=pca.get_latents(),
    #     )
    # # total data
    # logger.info("Plotting total PCA...")
    # variances = []
    # for i in range(2, 5):
    #     pca.fit(total_data, n_components=i)
    #     variances.append(pca.get_explained_variance())
    #     linear_plotter.create_plots(
    #         covariance_data=pca.get_covariance(),
    #         variance_data=variances,
    #         component_data=pca.get_components(),
    #         latent_data=pca.get_latents(),
    #     )

    ## get gp results
    # logger.info("Plotting GP results for averaged dataset...")
    # # flat results
    # logger.info("Plotting flat GP results...")
    # logger.info("Linear GP results...")
    # linear_gp.fit(averaged_flat_data, 2, flat_data=True)
    # linear_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=linear_gp.get_covariance(),
    #     covariance_labels=linear_gp.labels,
    #     sensitivity_data=linear_gp.get_sensitivity(),
    #     latent_data=linear_gp.get_latents(),
    # )
    # logger.info("RBF GP results...")
    # rbf_gp.fit(averaged_flat_data, 2, flat_data=True)
    # rbf_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=rbf_gp.get_covariance(),
    #     covariance_labels=rbf_gp.labels,
    #     sensitivity_data=rbf_gp.get_sensitivity(),
    #     latent_data=rbf_gp.get_latents(),
    # )
    # md results
    # logger.info("Plotting MD GP results...")
    # logger.info("Linear GP results...")
    # linear_gp.fit(averaged_data, 2)
    # linear_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=linear_gp.get_covariance(),
    #     covariance_labels=linear_gp.labels,
    #     sensitivity_data=linear_gp.get_sensitivity(),
    #     latent_data=linear_gp.get_latents(),
    # )
    # logger.info("RBF GP results...")
    # rbf_gp.fit(averaged_data, n_components)
    # rbf_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=rbf_gp.get_covariance(),
    #     covariance_labels=rbf_gp.labels,
    #     sensitivity_data=rbf_gp.get_sensitivity(),
    #     latent_data=rbf_gp.get_latents(),
    #     name=rbf_gp.name,
    # )
    # full_rbf_gp.fit(total_data, n_components)
    # full_rbf_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=full_rbf_gp.get_covariance(),
    #     covariance_labels=full_rbf_gp.labels,
    #     sensitivity_data=full_rbf_gp.get_sensitivity(),
    #     latent_data=full_rbf_gp.get_latents(),
    #     name=full_rbf_gp.name,
    # )
    # logger.info("Linear Expanded GP results...")
    # linear_expand_gp.fit(averaged_data, 2)
    # linear_expand_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=linear_expand_gp.get_covariance(),
    #     covariance_labels=linear_expand_gp.labels,
    #     sensitivity_data=linear_expand_gp.get_sensitivity(),
    #     latent_data=linear_expand_gp.get_latents(),
    # )
    # logger.info("RBF Expanded GP results...")
    # rbf_expand_gp.fit(averaged_data, n_components)
    # rbf_expand_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=rbf_expand_gp.get_covariance(),
    #     covariance_labels=rbf_expand_gp.labels,
    #     sensitivity_data=rbf_expand_gp.get_sensitivity(),
    #     latent_data=rbf_expand_gp.get_latents(),
    #     name=rbf_expand_gp.name,
    # )
    # full_rbf_expand_gp.fit(total_data, n_components)
    # full_rbf_expand_gp.save_model()
    # nonlinear_plotter.create_plots(
    #     covariance_data=full_rbf_expand_gp.get_covariance(),
    #     covariance_labels=full_rbf_expand_gp.labels,
    #     sensitivity_data=full_rbf_expand_gp.get_sensitivity(),
    #     latent_data=full_rbf_expand_gp.get_latents(),
    #     name=full_rbf_expand_gp.name
    # )
    # ## get brain results
    # logger.info("Plotting brain results...")
    # regression.fit(total_data)
    # brain_plotter.create_plots(
    #     data=averaged_data, regression_data=regression.get_regressions()
    # )
