from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 batch_size=1000,
                 shuffle=True,
                 data_shape=(28, 28),
                 ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataloader = None
        self._test_dataloader = None
        self._data_shape = data_shape
        self._num_workers = 12
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self._data_dir, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self._data_dir, train=False, transform=transforms.ToTensor()),
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True)

        self._train_dataloader = train_loader
        self._test_dataloader = test_loader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(f"No VAL data loader for MNIST")

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
