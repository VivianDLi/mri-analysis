from dataset_tests import *

if __name__ == "__main__":
    df = get_dataset()

    # plot_histograms(df, "feature", normalize=False)
    # plot_histograms(df, "feature", "Subject")
    # plot_histograms(df, "feature", "Label")
    # plot_histograms(df, "regression", "Subject")
    # plot_histograms(df.copy(), "regression", "Age")
    # plot_histograms(df.copy(), "regression", "Brain Volume")

    # plot_strip_plots(df)
    # plot_violin_plots(df)

    # plot_regression_plots(df.copy())
    # plot_scatter_plots(df.copy(), "feature", "feature", "feature")
    plot_scatter_plots(df.copy(), "feature", "feature", ["Age"], average=False)
    plot_scatter_plots(
        df.copy(), "feature", "feature", ["Brain Volume"], average=False
    )
    # plot_3D_features(df)

    # plot_pcas(df.copy(), ["CT_norm", "Age", "Brain Volume"], "Subject")
    # plot_pcas(df.copy(), ["SD_norm", "Age", "Brain Volume"], "Subject")
    # plot_pcas(df.copy(), ["MD_norm", "Age", "Brain Volume"], "Subject")
    # plot_pcas(df.copy(), ["ICVF_norm", "Age", "Brain Volume"], "Subject")
    # plot_pcas(df.copy(), ["CT_norm", "Age", "Brain Volume"], "Region")
    # plot_pcas(df.copy(), ["SD_norm", "Age", "Brain Volume"], "Region")
    # plot_pcas(df.copy(), ["MD_norm", "Age", "Brain Volume"], "Region")
    # plot_pcas(df.copy(), ["ICVF_norm", "Age", "Brain Volume"], "Region")

    # plot_pcas(df.copy(), ["CT_norm", "Age"], "Subject")
    # plot_pcas(df.copy(), ["SD_norm", "Age"], "Subject")
    # plot_pcas(df.copy(), ["MD_norm", "Age"], "Subject")
    # plot_pcas(df.copy(), ["ICVF_norm", "Age"], "Subject")
    # plot_pcas(df.copy(), ["CT_norm", "Age"], "Region")
    # plot_pcas(df.copy(), ["SD_norm", "Age"], "Region")
    # plot_pcas(df.copy(), ["MD_norm", "Age"], "Region")
    # plot_pcas(df.copy(), ["ICVF_norm", "Age"], "Region")

    # plot_pcas(
    #     df.copy(), ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"], "Subject"
    # )
    # plot_pcas(
    #     df.copy(), ["CT_norm", "SD_norm", "MD_norm", "ICVF_norm"], "Region"
    # )

    # plot_clusters(df)
