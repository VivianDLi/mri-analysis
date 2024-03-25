from mri_analysis.nonlinear.latent_methods import gp_model, save_gp_model
from mri_analysis.visualizations.plotting import *

if __name__ == "__main__":
    # df = get_dataset()

    # plot_pca_variance(df, average=False)
    # plot_pca_variance(df, average=True)
    # for i in range(1, 5):
    #     plot_pca_eigenvectors(df, i)
    #     plot_pca_eigenvectors(df, i, group_name="Region")
    # for feature in ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"]:
    #     ablated_df = df[list(set(df.columns) - set([feature]))]
    #     plot_pca_eigenvectors(ablated_df, 3, removed_columns=[feature])
    #     plot_pca_eigenvectors(
    #         ablated_df, 3, removed_columns=[feature], group_name="Region"
    #     )

    print("single region no average")
    # gp_df = get_gp_dataset(subset=1, average=False)
    # model = gp_model(gp_df, sort_by=["Region", "Feature", "Subject"])
    # save_gp_model(model, "./src/gp_models/1_random_model.npy")

    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     subset=1,
    #     model_name="1_random_model",
    # )

    print("single region no average md features")
    # gp_df = get_gp_dataset(subset=1, average=False, md=True)
    # model = gp_model(gp_df, sort_by=["Region", "Subject"], md=True)
    # save_gp_model(model, "./src/gp_models/1_random_multi_model.npy")

    print("10 subset no average md features")
    # gp_df = get_gp_dataset(subset=10, average=False, md=True)
    # # model = gp_model(gp_df, sort_by=["Region", "Subject"], md=True)
    # # save_gp_model(model, "./src/gp_models/10_random_md_model.npy")
    # model = gp_model(gp_df[["CT_norm"]], sort_by=["Region", "Subject"])
    # save_gp_model(model, "./src/gp_models/10_random_ct_model.npy")
    # plot_gp_correlation(
    #     gp_df[["CT_norm"]],
    #     sort_by=["Region", "Subject"],
    #     subset=10,
    #     model_name="10_random_ct_model",
    # )

    # model = gp_model(gp_df[["SD_norm"]], sort_by=["Region", "Subject"])
    # save_gp_model(model, "./src/gp_models/10_random_sd_model.npy")
    # plot_gp_correlation(
    #     gp_df[["SD_norm"]],
    #     sort_by=["Region", "Subject"],
    #     subset=10,
    #     model_name="10_random_sd_model",
    # )

    # model = gp_model(gp_df[["MD_norm"]], sort_by=["Region", "Subject"])
    # save_gp_model(model, "./src/gp_models/10_random_md_model.npy")
    # plot_gp_correlation(
    #     gp_df[["MD_norm"]],
    #     sort_by=["Region", "Subject"],
    #     subset=10,
    #     model_name="10_random_md_model",
    # )

    # model = gp_model(gp_df[["ICVF_norm"]], sort_by=["Region", "Subject"])
    # save_gp_model(model, "./src/gp_models/10_random_icvf_model.npy")
    # plot_gp_correlation(
    #     gp_df[["ICVF_norm"]],
    #     sort_by=["Region", "Subject"],
    #     subset=10,
    #     model_name="10_random_icvf_model",
    # )

    print("all regions average md features")
    gp_df = get_gp_dataset(subset=None, md=True)
    model = gp_model(gp_df, sort_by=["Subject", "Region"], expand_dims=True)
    save_gp_model(model, "./src/gp_models/md_model.npy")

    # gp_df = get_gp_dataset(subset=None, average=True, md=True)
    # # model = gp_model(gp_df, sort_by=["Region"], md=True)
    # # save_gp_model(model, "./src/gp_models/average_md_model.npy")
    # model = gp_model(gp_df[["CT_norm"]], sort_by=["Region"])
    # save_gp_model(model, "./src/gp_models/average_ct_model.npy")
    # plot_gp_correlation(
    #     gp_df[["CT_norm"]],
    #     sort_by=["Region"],
    #     subset=None,
    #     model_name="average_ct_model",
    # )

    # model = gp_model(gp_df[["SD_norm"]], sort_by=["Region"])
    # save_gp_model(model, "./src/gp_models/average_sd_model.npy")
    # plot_gp_correlation(
    #     gp_df[["SD_norm"]],
    #     sort_by=["Region"],
    #     subset=None,
    #     model_name="average_sd_model",
    # )

    # model = gp_model(gp_df[["MD_norm"]], sort_by=["Region"])
    # save_gp_model(model, "./src/gp_models/average_md_model.npy")
    # plot_gp_correlation(
    #     gp_df[["MD_norm"]],
    #     sort_by=["Region"],
    #     subset=None,
    #     model_name="average_md_model",
    # )

    # model = gp_model(gp_df[["ICVF_norm"]], sort_by=["Region"])
    # save_gp_model(model, "./src/gp_models/average_icvf_model.npy")
    # plot_gp_correlation(
    #     gp_df[["ICVF_norm"]],
    #     sort_by=["Region"],
    #     subset=None,
    #     model_name="average_icvf_model",
    # )

    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature"],
    #     subset=None,
    #     model_name="average_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature"],
    #     subset=None,
    #     model_name="average_modified_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region"],
    #     subset=None,
    #     model_name="average_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region"],
    #     subset=None,
    #     model_name="average_modified_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region"],
    #     subset=None,
    #     sort=True,
    #     model_name="average_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region"],
    #     subset=None,
    #     sort=True,
    #     model_name="average_modified_model",
    # )
    # plot_gp_latents(
    #     gp_df, sort_by=["Feature", "Region"], model_name="average_model"
    # )
    # plot_gp_latents(
    #     gp_df,
    #     sort_by=["Feature", "Region"],
    #     model_name="average_modified_model",
    # )

    # gp_df = (
    #     pd.read_csv("./src/gp_models/10_random_data.csv")
    #     .set_index(["Region", "Feature", "Subject"])
    #     .sort_index()
    # )

    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     model_name="10_random_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     model_name="10_random_modified_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region", "Subject"],
    #     model_name="10_random_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Feature", "Region", "Subject"],
    #     model_name="10_random_modified_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     sort=True,
    #     model_name="10_random_model",
    # )
    # plot_gp_correlation(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     sort=True,
    #     model_name="10_random_modified_model",
    # )
    # plot_gp_latents(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     model_name="10_random_model",
    # )
    # plot_gp_latents(
    #     gp_df,
    #     sort_by=["Region", "Feature", "Subject"],
    #     model_name="10_random_modified_model",
    # )
