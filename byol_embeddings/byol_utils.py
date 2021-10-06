"""
Utilities for training & interpreting BYOL model.
"""
from __future__ import annotations

import os
from typing import Union, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from byol_pytorch import BYOL


class FullDataset(Dataset):
    """Basic class encapulating dataset functions, for use with dataloader."""

    def __init__(self, numpy_array: np.ndarray) -> None:
        """Initialise with numpy array."""
        self.data = numpy_array
        print("Array shape loaded into torch dataset:", numpy_array.shape)

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: Any) -> torch.Tensor:
        """Return items from dataset as torch tensors."""
        data_selection = self.data[idx]
        tensor = torch.from_numpy(data_selection).float()
        return tensor


class BYOLHandler:
    """Encapsulates different utility methods for working with BYOL model."""

    def __init__(
            self, device: Union[str, torch.device] = "cpu", load_from_path: str = None,
            augmentations: nn.Sequential = None
            ) -> None:
        """Initialise model and learner setup."""
        self.device = device
        self.model = models.wide_resnet101_2(pretrained=False).to(self.device)
        self.timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        if load_from_path is not None:
            print("Loading model...")
            state_dict = torch.load(load_from_path)
            self.model.load_state_dict(state_dict)

        if augmentations is None:
            self.learner = BYOL(
                self.model,
                image_size=64,
                hidden_layer="avgpool"
            )

        if augmentations is not None:
            self.learner = BYOL(
                self.model,
                image_size=64,
                hidden_layer="avgpool",
                augment_fn=augmentations
            )

        self.opt = torch.optim.Adam(self.learner.parameters(), lr=0.0001, betas=(0.9, 0.999))
        self.loss_history: list[float] = []

    def train(
            self, dataset: torch.utils.data.Dataset,
            epochs: int = 1, use_tqdm: bool = False
            ) -> None:
        """Train model on dataset for specified number of epochs."""
        for i in range(epochs):
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=64, shuffle=True
            )
            if use_tqdm:
                dataloader = tqdm(dataloader)
            for images in dataloader:
                device_images = images.to(self.device)
                loss = self.learner(device_images)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.learner.update_moving_average()
                self.loss_history.append(loss.detach().item())
                del device_images
                torch.cuda.empty_cache()
            print("Epochs performed:", i + 1)

    def save(self) -> None:
        """Save model."""
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        dirpath = os.path.join("outputs", self.timestamp)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        save_path = os.path.join(dirpath, "model.pt")
        torch.save(self.model.state_dict(), save_path)

        # Plot loss history:
        fig, ax = plt.subplots()
        ax.plot(self.loss_history)
        fig.tight_layout()
        fig.savefig(os.path.join(dirpath, "loss_v_batch.png"), dpi=300)
        plt.close()

    def infer(self, dataset: torch.utils.data.Dataset, use_tqdm: bool = False) -> np.ndarray:
        """Use model to infer embeddings of provided dataset."""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=0
        )
        embeddings_list: list[np.ndarray] = []
        with torch.no_grad():
            self.model.eval()
            if use_tqdm:
                dataloader = tqdm(dataloader)
            for data in dataloader:
                device_data = data.to(self.device)
                projection, embedding = self.learner(device_data, return_embedding=True)
                np_embedding = embedding.detach().cpu().numpy()
                np_embedding = np.reshape(np_embedding, (data.shape[0], -1))
                embeddings_list.append(np_embedding)
                del device_data
                del projection
                del embedding
                torch.cuda.empty_cache()
            self.model.train()

        return np.concatenate(embeddings_list, axis=0)
