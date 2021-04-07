"""
Script for running training and inference of trained BYOL model.

Example:
python -m byol_embeddings -t
"""
from __future__ import annotations

import os
import argparse
import random
from typing import Any

import scipy.stats
import torch
import torch.nn as nn
import kornia.augmentation.augmentation as aug
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
        help="where to source the dataset on which to train the model"
    )
    parser.add_argument(
        "-l", "--load_path", type=str, default=None,
        help="path to model state to load"
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


class RandomApply(nn.Module):
    """Helper class for randomly applying transformations to images."""

    def __init__(self, fn: Any, p: float) -> None:
        """Initialise with given transforms."""
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly apply transforms."""
        if random.random() > self.p:
            return x
        return self.fn(x)


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
        data: np.ndarray, labels: np.ndarray, reduction_factor: int, name: str = "test"
        ) -> np.ndarray:
    """Plot a scatterplot for a given PCA reduction factor."""
    dirpath = os.path.join("outputs", name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    # Running embeddings:
    mapper = umap.UMAP(densmap=True, dens_lambda=2, random_state=0)
    umap_embeddings = mapper.fit_transform(data[:, ::reduction_factor])

    print("Plotting & saving figures...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, s=0.01, marker=".")
    fig.savefig(os.path.join(dirpath, "umap_scatterplot.png"), dpi=300)
    plt.close()

    # Constructing pandas dataframe:
    embeddings_df = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
    embeddings_df["Condition"] = ""
    label_list = [
        "ONE",
        "CTRL",
        "PTTHREE",
        "THREE"
    ]

    for i in range(4):
        start = i * 20000
        end = (i + 1) * 20000
        embeddings_df.loc[start:end, ("Condition")] = label_list[i]

    # Plotting kdeplots:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.kdeplot(
        data=embeddings_df, x="UMAP_1", y="UMAP_2",
        fill=True, thresh=0, levels=100, cmap="mako",
        ax=ax
    )
    fig.savefig(os.path.join(dirpath, "total_kde_plot.png"))
    plt.close()

    for label in label_list:
        plot_kde_distribution(embeddings_df, label, dirpath)

    return umap_embeddings


def main() -> None:
    """Parse arguments + run relevant bits of the package as directed."""
    # Parse command line arguments & setting device:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",  device)

    # Loading dataset:
    print("Loading dataset...")
    file_list = os.listdir(args.path_to_dataset)
    dataset_list: list[np.ndarray] = []
    for filename in file_list:
        print(filename)
        filepath = os.path.join(args.path_to_dataset, filename)
        array = np.load(filepath)[:, :, :, 1]
        array = array[0:20000]
        array = np.expand_dims(array, 1)
        array = np.repeat(array, 3, axis=1)
        print(array.shape)
        dataset_list.append(array)
    np_dataset = np.concatenate(dataset_list, axis=0)
    dataset = FullDataset(np_dataset)
    label_list = [np.ones(20000) * 2, np.ones(20000) * 0, np.ones(20000) * 1, np.ones(20000) * 3]
    labels = np.concatenate(label_list, axis=0)

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
        device=device, load_from_path=args.load_path, augmentations=augmentations
    )
    if not args.no_train:
        print("Training handler...")
        handler.train(dataset, epochs=args.epochs, use_tqdm=args.tqdm)
        handler.save(filename="wide101.pt")
        fig, ax = plt.subplots()
        ax.plot(handler.loss_history)
        fig.tight_layout()
        fig.savefig("outputs/loss_history.png", dpi=300)
        plt.close()

    # Generating embeddings:
    embeddings = handler.infer(dataset, use_tqdm=args.tqdm)
    print("Embedding dimensions:", embeddings.shape)

    # Finding most relevant features:
    print("Calculating batch scaled importance of features...")

    # Selecting:
    ctrl_std = scipy.stats.median_abs_deviation(embeddings[20000:40000], axis=0)
    total_std = scipy.stats.median_abs_deviation(embeddings, axis=0)
    ratio_std = total_std / ((ctrl_std) + 1e-16)

    sorted_indices = np.argsort(ratio_std)
    sorted_embeddings = embeddings[:, sorted_indices]

    # Running inference:
    print("Performing UMAP analyis and plotting...")
    # umap_embeddings = plot_umap_embeddings(
    #     sorted_embeddings[:, -5:], labels, 1, "test0"
    # )
    umap_embeddings = plot_umap_embeddings(
        sorted_embeddings[:, :], labels, 1, "test1"
    )
    # umap_embeddings = plot_umap_embeddings(
    #     sorted_embeddings[:, -100:], labels, 1, "test2"
    # )
    # umap_embeddings = plot_umap_embeddings(
    #     sorted_embeddings, labels, 1, "test3"
    # )
    print("Plotting images onto UMAP...")
    plot_image_umap(umap_embeddings[::100], np_dataset[::100])

    # Saving embeddings:
    np.save("outputs/embeddings.npy", sorted_embeddings)

    return None


main()
