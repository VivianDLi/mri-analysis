from nilearn import datasets
from nilearn import plotting
from nibabel.freesurfer.io import read_morph_data, read_annot
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import scipy.io
import pandas as pd
import numpy as np
import copy
from matplotlib import cm
import os


def plot_brain(
    myvec,
    parc,
    cbar=False,
    cbartitle="",
    outfile=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    categorical=False,
):
    # this works for aparc, HCP, 500.aparc, 500.sym.aparc
    if parc not in [
        "aparc",
        "500.sym.aparc",
        "500.aparc",
        "HCP",
        "Schaefer2018_200Parcels_7Networks_order",
        "aparc.a2009s",
        "BrodmannAll",
    ]:
        raise "Only valid for parcellations: aparc, 500.sym.aparc, 500.aparc, HCP, Schaefer2018_200Parcels_7Networks_order, aparc.a2009s, BrodmannAll"

    surfer_location = os.path.dirname(__file__)

    # additional code for plotting keyed regions on brain
    if not categorical:
        myvec = np.array(myvec) + 1e-15
    annot_lh = read_annot(
        surfer_location + "/label/lh." + parc + ".annot", orig_ids=True
    )
    annot_rh = read_annot(
        surfer_location + "/label/rh." + parc + ".annot", orig_ids=True
    )

    regions_l = annot_lh[-1]
    regions_r = annot_rh[-1]

    regions_l = ["lh_" + str(x).split("'")[1] for x in regions_l]
    regions_r = ["rh_" + str(x).split("'")[1] for x in regions_r]

    regions = np.hstack((regions_l, regions_r))

    if parc == "BN_atlas":
        regions = regions_l

    if parc == "HCP":
        regions = [x.replace("-", "_") for x in regions]

    # might need to get rid of some other labels here for other parcellations depending on what automatically shows up
    drop_regions = np.array(
        [
            x
            for x in regions
            if ("?" in x)
            | ("unknown" in x)
            | ("Unknown" in x)
            | ("corpuscallosum" in x)
            | ("Unknown" in x)
            | ("Medial_Wall" in x)
        ]
    )
    ROI_labels = np.array([x for x in regions if x not in drop_regions])

    value_dict = dict(zip(ROI_labels, myvec))

    if categorical:
        # myvec is a unique list of region names
        region_indices = np.zeros(len(ROI_labels))
        color_index = 1
        for region in myvec:
            region_indices[
                np.where(ROI_labels == region.replace("-", "_"))
            ] = color_index
            color_index += 1
        value_dict = dict(zip(ROI_labels, region_indices))

    label_nums_l = list(annot_lh[-2][:, -1])
    label_nums_r = list(annot_rh[-2][:, -1])

    # To ensure R and L have different number labels
    if (parc == "aparc") | (parc == "500.sym.aparc"):
        label_nums_l = list(1 + annot_lh[-2][:, -1])

    new_labels_l = np.zeros((len(annot_lh[0]), 1))
    new_labels_r = np.zeros((len(annot_rh[0]), 1))

    all_label_numbers = np.concatenate(
        (np.array(label_nums_l), np.array(label_nums_r))
    )

    convert_dict = dict(zip(all_label_numbers, regions))

    # only keep good regions in conversion dict
    drop_vals = []
    for key, value in convert_dict.items():
        if value in drop_regions:
            drop_vals.append(key)

    for x in drop_vals:
        del convert_dict[x]

    # Deal with strange numbering
    for i in range(len(label_nums_l)):
        if label_nums_l[i] in convert_dict.keys():
            if convert_dict[label_nums_l[i]] in value_dict.keys():
                new_labels_l[np.where(annot_lh[0] == label_nums_l[i])] = (
                    value_dict[convert_dict[label_nums_l[i]]]
                )
                if (parc == "aparc") | (parc == "500.sym.aparc"):
                    new_labels_l[
                        np.where(annot_lh[0] + 1 == label_nums_l[i])
                    ] = value_dict[convert_dict[label_nums_l[i]]]

    for i in range(len(label_nums_r)):
        if label_nums_r[i] in convert_dict.keys():
            if convert_dict[label_nums_r[i]] in value_dict.keys():
                new_labels_r[np.where(annot_rh[0] == label_nums_r[i])] = (
                    value_dict[convert_dict[label_nums_r[i]]]
                )

    new_labels_l[np.where(new_labels_l == 0)] = np.nan
    new_labels_r[np.where(new_labels_r == 0)] = np.nan

    all_labels = np.hstack((new_labels_r, new_labels_l))
    cmap_min = np.nanmin(all_labels)
    cmap_max = np.nanmax(all_labels)

    if vmin != None:
        cmap_min = vmin

    if vmax != None:
        cmap_max = vmax

    new_labels_l = (
        new_labels_l - (np.nanmax(all_labels) + np.nanmin(all_labels)) / 2
    )
    new_labels_r = (
        new_labels_r - (np.nanmax(all_labels) + np.nanmin(all_labels)) / 2
    )

    # THESE FILES ARE FROM THE required_data/fsavergeSubP/surf/ directory in brainsforpublication

    annot_l = surfer_location + "/surf/lh.inflated"
    bg_map_l = surfer_location + "/surf/lh.sulc"

    annot_r = surfer_location + "/surf/rh.inflated"
    bg_map_r = surfer_location + "/surf/rh.sulc"

    # Set colormap; copying the cmap prevents it from overwriting the original plt copy
    cmap = copy.copy(cm.get_cmap(cmap))
    cmap.set_bad(color="grey", alpha=1.0)

    plt.rcParams.update({"font.size": 24})
    plt.rcParams.update({"font.family": "Avenir"})

    # Colormap title
    cmap_title = cbartitle

    fig, ax = plt.subplots(
        1, 4, figsize=(15, 4), subplot_kw={"projection": "3d"}
    )

    plotting.plot_surf_stat_map(
        annot_r,
        stat_map=new_labels_r,
        hemi="right",
        bg_map=bg_map_r,
        view="lateral",
        bg_on_data=True,
        darkness=0.5,
        axes=ax[3],
        colorbar=False,
        figure=fig,
        cmap=cmap,
    )  #'coolwarm')

    plotting.plot_surf_stat_map(
        annot_l,
        stat_map=new_labels_l,
        hemi="left",
        bg_map=bg_map_l,
        view="lateral",
        bg_on_data=True,
        darkness=0.5,
        axes=ax[0],
        colorbar=False,
        figure=fig,
        cmap=cmap,
    )  #'coolwarm')

    plotting.plot_surf_stat_map(
        annot_r,
        stat_map=new_labels_r,
        hemi="right",
        bg_map=bg_map_r,
        view="medial",
        bg_on_data=True,
        darkness=0.5,
        axes=ax[2],
        colorbar=False,
        figure=fig,
        cmap=cmap,
    )  #'coolwarm')

    plotting.plot_surf_stat_map(
        annot_l,
        stat_map=new_labels_l,
        hemi="left",
        bg_map=bg_map_l,
        view="medial",
        bg_on_data=True,
        darkness=0.5,
        axes=ax[1],
        colorbar=False,
        figure=fig,
        cmap=cmap,
    )  #'coolwarm')

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    )

    if cbar == True:
        cbar = plt.colorbar(
            sm, location="bottom", ax=ax, aspect=30, shrink=0.75
        )
        cbar.ax.set_xlabel(cmap_title)

    # plt.tight_layout()

    if outfile:
        fig.savefig(outfile)

    # plt.show()
