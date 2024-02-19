from dataset_tests import *

if __name__ == "__main__":
    df = get_dataset()
    gp_df = get_gp_dataset()

    plot_gp_correlation(
        gp_df, sort_by=["Region", "Feature", "Subject"], model_name="model"
    )
    plot_gp_correlation(
        gp_df,
        sort_by=["Region", "Feature", "Subject"],
        model_name="modified_model",
    )
    # plot_gp_correlation(gp_df, use_file=False, model_name="bayesian_model")
    # plot_gp_correlation(
    #     gp_df, use_file=False, model_name="modified_bayesian_model"
    # )
    plot_gp_correlation(
        gp_df,
        sort_by=["Region", "Feature", "Subject"],
        model_name="sparse_model",
    )
    plot_gp_correlation(
        gp_df,
        sort_by=["Region", "Feature", "Subject"],
        model_name="modified_sparse_model",
    )
