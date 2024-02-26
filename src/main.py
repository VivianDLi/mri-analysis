from dataset_tests import *

if __name__ == "__main__":
    gp_df = get_gp_dataset(subset=None, average=True)

    plot_gp_correlation(
        gp_df,
        sort_by=["Region", "Feature"],
        subset=None,
        model_name="average_model",
    )
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
