"""
Script for running training and inference of trained BYOL model.

Example:
python -m byol_embeddings -t
"""
from __future__ import annotations

import os
import argparse
import json
from typing import Any
from datetime import datetime

import scipy.stats
import torch
import kornia.augmentation.augmentation as aug
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans, DBSCAN
import umap
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from .byol_utils import FullDataset, BYOLHandler


def _parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_dataset", type=str, default=None,
        help="path to the dataset on which to train the model"
    )
    parser.add_argument(
        "--path_to_labels", type=str, default=None,
        help="path to the label numpy array for calculating feature importances"
    )
    parser.add_argument(
        "--path_to_label_dictionary", type=str, default=None,
        help="path to the label dictionary for calculating feature importances"
    )
    parser.add_argument(
        "--ctrl_labels", nargs="+",
        help="which labels correspond to control conditions"
    )
    parser.add_argument(
        "-l", "--path_to_model", type=str, default=None,
        help="path to model state to load"
    )
    parser.add_argument(
        "--path_to_embeddings", type=str, default=None,
        help="path to embeddings to load"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1,
        help="number of epochs to train the model for"
    )
    parser.add_argument(
        "--tqdm", action="store_true",
        help="whether to use a tqdm progress bar"
    )
    parser.add_argument(
        "--no_train", action="store_true",
        help="whether to train model"
    )
    return parser.parse_args()


def plot_kde_distribution(dataframe: pd.DataFrame, condition: str, dirpath: str) -> None:
    """Plot kde distribution for a given condition."""
    fig, ax = plt.subplots(figsize=(10, 10))
    filtered_df = dataframe.loc[dataframe["Condition"] == condition]
    sns.kdeplot(
        data=filtered_df, x="UMAP_1", y="UMAP_2",
        fill=True, thresh=0, levels=100, cmap="mako",
        ax=ax
    )
    out_path = os.path.join(dirpath, condition + "_umap_kdeplot.png")
    fig.savefig(out_path)
    plt.close()


def generate_image_grid(image_array: np.ndarray) -> np.ndarray:
    """Plot ten by ten grid of images from provided array."""
    count = 0
    row_list: list[np.ndarray] = []
    for i in range(10):
        temp_row: list[np.ndarray] = []
        for j in range(10):
            temp_row.append(np.squeeze(image_array[count]))
            count += 1
        row_list.append(np.concatenate(temp_row, axis=1))

    return np.concatenate(row_list, axis=0)


def plot_image_umap(embeddings: np.ndarray, image_array: np.ndarray) -> None:
    """Plot images onto points in the umap embedding space."""
    # Generating scatterplot:
    fig, ax = plt.subplots(figsize=(60, 60))
    artists = []
    for i in range(image_array.shape[0]):
        img_array = image_array[i, 0, :, :].squeeze()
        img = OffsetImage(img_array, zoom=1, cmap='gray')
        ab = AnnotationBbox(
            img, (embeddings[i, 0], embeddings[i, 1]),
            xycoords='data', frameon=False
        )
        artists.append(ax.add_artist(ab))

    # Tweaking plot:
    ax.update_datalim(embeddings)
    ax.autoscale()
    fig.set_facecolor("k")
    ax.set_facecolor("k")
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='x', colors='w')
    ax.tick_params(axis='y', colors='w')

    # Saving to file:
    fig.tight_layout()
    fig.savefig("outputs/image_umap_embeddings.png", dpi=300)
    plt.close()


def plot_umap_embeddings(
        data: np.ndarray, labels: np.ndarray, dirpath: str,
        ctrl_mask: np.ndarray, image_dataset: np.ndarray
        ) -> None:
    """Plot a scatterplot for a given PCA reduction factor."""
    # Generating folder:
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Running embeddings:
    mapper = umap.UMAP(densmap=True, dens_lambda=2, random_state=0)
    umap_embeddings = mapper.fit_transform(data)

    # Segregating into control and dox populations:
    ctrl_embeddings = umap_embeddings[ctrl_mask]
    not_ctrl_embeddings = umap_embeddings[np.logical_not(ctrl_mask)]

    print("Plotting & saving scatterplots...")
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), sharex=True, sharey=True)
    axs[0].scatter(ctrl_embeddings[:, 0], ctrl_embeddings[:, 1], s=0.1, marker=".")
    axs[0].set_title("CTRL Embeddings")
    axs[1].scatter(not_ctrl_embeddings[:, 0], not_ctrl_embeddings[:, 1], s=0.1, marker=".")
    axs[1].set_title("DOX Embeddings")
    axs[2].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, s=0.1, marker=".")
    axs[2].set_title("All embeddings")

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "umap_scatterplot.png"), dpi=300)
    plt.close()

    print("Plotting and saving kdeplots...")
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), sharex=True, sharey=True)
    sns.kdeplot(x=ctrl_embeddings[:, 0], y=ctrl_embeddings[:, 1], ax=axs[0])
    axs[0].set_title("CTRL Embeddings")
    sns.kdeplot(x=not_ctrl_embeddings[:, 0], y=not_ctrl_embeddings[:, 1], ax=axs[1])
    axs[1].set_title("DOX Embeddings")
    sns.kdeplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], ax=axs[2])
    axs[2].set_title("All Embeddings")

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "umap_kdeplot.png"), dpi=300)
    plt.close()

    print("Plotting and saving kdeplots of different induced mutants...")
    # Creating mutant subslices:
    f91s_embeddings = umap_embeddings[labels == 1]
    l4q_embeddings = umap_embeddings[labels == 3]
    r45h_embeddings = umap_embeddings[labels == 5]
    wt_embeddings = umap_embeddings[labels == 7]
    wtctrl_embeddings = umap_embeddings[labels == 6]

    # Plotting data:
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(50, 10), sharex=True, sharey=True)

    sns.kdeplot(x=wtctrl_embeddings[:, 0], y=wtctrl_embeddings[:, 1], ax=axs[0])
    axs[0].set_title("WTCTRL Embeddings")

    sns.kdeplot(x=wt_embeddings[:, 0], y=wt_embeddings[:, 1], ax=axs[1])
    axs[1].set_title("WTDOX Embeddings")

    sns.kdeplot(x=f91s_embeddings[:, 0], y=f91s_embeddings[:, 1], ax=axs[2])
    axs[2].set_title("F91S Embeddings")

    sns.kdeplot(x=l4q_embeddings[:, 0], y=l4q_embeddings[:, 1], ax=axs[3])
    axs[3].set_title("L4Q Embeddings")

    sns.kdeplot(x=r45h_embeddings[:, 0], y=r45h_embeddings[:, 1], ax=axs[4])
    axs[4].set_title("R45H Embeddings")

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "mutant_kdeplot.png"), dpi=300)
    plt.close()

    print("Plotting and saving kdeplots of different ctrl mutants...")
    # Creating mutant subslices:
    f91sctrl_embeddings = umap_embeddings[labels == 0]
    l4qctrl_embeddings = umap_embeddings[labels == 2]
    r45hctrl_embeddings = umap_embeddings[labels == 4]

    # Plotting data:
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(40, 10), sharex=True, sharey=True)

    sns.kdeplot(x=wtctrl_embeddings[:, 0], y=wtctrl_embeddings[:, 1], ax=axs[0])
    axs[0].set_title("WTCTRL Embeddings")

    sns.kdeplot(x=f91sctrl_embeddings[:, 0], y=f91sctrl_embeddings[:, 1], ax=axs[1])
    axs[1].set_title("F91SCTRL Embeddings")

    sns.kdeplot(x=l4qctrl_embeddings[:, 0], y=l4qctrl_embeddings[:, 1], ax=axs[2])
    axs[2].set_title("L4QCTRL Embeddings")

    sns.kdeplot(x=r45hctrl_embeddings[:, 0], y=r45hctrl_embeddings[:, 1], ax=axs[3])
    axs[3].set_title("R45HCTRL Embeddings")

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "mutant_ctrl_kdeplot.png"), dpi=300)
    plt.close()

    # Finding KDEs of full data, and density gating:
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), sharex=True, sharey=True)

    # Performing kernel density estimation:
    kde = KernelDensity(kernel='gaussian', bandwidth=0.8).fit(umap_embeddings)
    log_density = kde.score_samples(umap_embeddings)
    axs[0].scatter(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], c=log_density, s=0.05)

    # Filtering based on density:
    density_mask = log_density > np.percentile(log_density, 75)
    density_filtered = umap_embeddings[density_mask]
    axs[1].scatter(x=density_filtered[:, 0], y=density_filtered[:, 1], s=0.05)

    # Performing clustering:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(density_filtered)
    scatter = axs[2].scatter(
        x=density_filtered[:, 0], y=density_filtered[:, 1], c=kmeans.labels_, s=0.05
    )
    class_legend = axs[2].legend(*scatter.legend_elements(), loc="upper left", title="Classes")
    axs[2].add_artist(class_legend)

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "density_filtering_steps.png"), dpi=300)
    plt.close()

    # Plotting image grids of different clusters:
    density_filtered_images = image_dataset[density_mask]
    label0_images = density_filtered_images[kmeans.labels_ == 0]
    label1_images = density_filtered_images[kmeans.labels_ == 1]
    # label2_images = density_filtered_images[kmeans.labels_ == 2]
    # label3_images = density_filtered_images[kmeans.labels_ == 3]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)

    axs[0].imshow(generate_image_grid(label0_images))
    axs[0].set_title("Cluster 0")
    axs[0].axis("off")

    axs[1].imshow(generate_image_grid(label1_images))
    axs[1].set_title("Cluster 1")
    axs[1].axis("off")

    # axs[2].imshow(generate_image_grid(label2_images))
    # axs[2].set_title("Cluster 2")
    # axs[2].axis("off")

    # axs[3].imshow(generate_image_grid(label3_images))
    # axs[3].set_title("Cluster 3")
    # axs[3].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(dirpath, "cluster_images.png"), dpi=600)
    plt.close()

    # Attempting DBSCAN analysis:
    # clustering = DBSCAN(eps=0.2, min_samples=25).fit(umap_embeddings)
    # label_array = np.unique(clustering.labels_)
    # num_labels = label_array.shape[0] - 1

    # fig, ax = plt.subplots()
    # scatter = ax.scatter(
    #     x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], c=clustering.labels_, s=0.05
    # )
    # class_legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")
    # ax.add_artist(class_legend)

    # fig.tight_layout()
    # fig.savefig(os.path.join(dirpath, "dbscan_clusters.png"), dpi=300)
    # plt.close()

    # fig, axs = plt.subplots(nrows=1, ncols=num_labels, figsize=(num_labels * 10, 10))
    # for class_id in label_array:
    #     if class_id == -1:
    #         continue
    #     images = image_dataset[clustering.labels_ == class_id]
    #     axs[class_id].imshow(generate_image_grid(images))
    #     axs[class_id].set_title("Cluster " + str(class_id))
    #     axs[class_id].axis("off")

    # fig.tight_layout()
    # fig.savefig(os.path.join(dirpath, "dbscan_images.png"), dpi=300)
    # plt.close()

    return umap_embeddings


def main() -> None:
    """Parse arguments + run relevant bits of the package as directed."""
    # Parse command line arguments & setting device:
    args = _parse_args()
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    print(timestamp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",  device)

    # Loading dataset:
    print("Loading dataset...")
    np_dataset = np.expand_dims(np.load(args.path_to_dataset)[:, :, :, 1], axis=1)
    dataset = FullDataset(np.repeat(np_dataset, 3, axis=1))
    labels = np.load(args.path_to_labels)[:]
    with open(args.path_to_label_dictionary, "r") as read_file:
        label_dict = json.load(read_file)

    # Defining augmentations:
    augmentations = torch.nn.Sequential(
        aug.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            p=0.3
        ),
        aug.RandomSharpness(0.5),
        aug.RandomHorizontalFlip(),
        aug.RandomVerticalFlip(),
        aug.GaussianBlur(kernel_size=(3, 3), sigma=(1, 2), p=0.2),
        aug.RandomResizedCrop((64, 64)),
        aug.RandomAffine(degrees=180, p=0.2)
    )

    # Training BYOL:
    handler = BYOLHandler(
        device=device, load_from_path=args.path_to_model, augmentations=augmentations
    )
    if not args.no_train:
        print("Training handler...")
        handler.train(dataset, epochs=args.epochs, use_tqdm=args.tqdm)
        handler.save()

    # Generating embeddings:
    if args.path_to_embeddings is None:
        embeddings = handler.infer(dataset, use_tqdm=args.tqdm)
        print("Embedding dimensions:", embeddings.shape)

    if args.path_to_embeddings is not None:
        embeddings = np.load(args.path_to_embeddings)

    # Finding most relevant features:
    print("Calculating batch scaled importance of features...")

    # Perform variance selection:
    ctrl_labels: list[int] = []
    for num_label in label_dict:
        if label_dict[num_label] in args.ctrl_labels:
            ctrl_labels.append(int(num_label))

    ctrl_mask = np.array([0] * labels.shape[0], dtype=bool)
    for ctrl_label in ctrl_labels:
        ctrl_mask = ctrl_mask | (labels == ctrl_label)
    print("Control mask shape:", embeddings[ctrl_mask].shape)

    ctrl_std = scipy.stats.median_abs_deviation(embeddings[ctrl_mask], axis=0)
    total_std = scipy.stats.median_abs_deviation(embeddings, axis=0)
    ratio_std = total_std / ((ctrl_std) + 1e-16)

    sorted_indices = np.argsort(ratio_std)
    sorted_embeddings = embeddings[:, sorted_indices]

    # Transforming into uniform distribution:
    # lambdas = 1 / sorted_embeddings.mean(axis=0)
    # uniform_embeddings = 1 - (lambdas * np.exp(sorted_embeddings * -lambdas))
    # print("Size of final embeddings array:", uniform_embeddings.shape)

    # zero_mask = np.min(sorted_embeddings[:, -5:], axis=1) > 0
    # zero_filter = sorted_embeddings[zero_mask]
    # zero_labels = labels[zero_mask]

    outlier_mask = np.max(sorted_embeddings[:, -10:], axis=1) < 2
    # outlier_filtered_embeddings = sorted_embeddings[outlier_mask]
    # outlier_labels = labels[outlier_mask]

    # print("Size of zero filtered dataset:", zero_filter.shape)

    # Running inference:
    print("Performing UMAP analyis and plotting...")
    if not os.path.exists("plots"):
        os.mkdir("plots")

    plots_dir = os.path.join("plots", timestamp)

    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)

    # plot_umap_embeddings(
    #     sorted_embeddings[outlier_mask][::10, -750:],
    #     labels[outlier_mask][::10],
    #     plots_dir,
    #     ctrl_mask[outlier_mask][::10],
    #     np_dataset[outlier_mask][::10]
    # )

    embeddings_path = os.path.join(plots_dir, "embeddings.npy")
    np.save(embeddings_path, sorted_embeddings)

    labels_path = os.path.join(plots_dir, "labels.npy")
    np.save(labels_path, labels)

    return None


main()
