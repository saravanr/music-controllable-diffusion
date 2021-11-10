from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np

from utils.cuda_utils import get_device


class MNISTTensorDataSet(Dataset):
    def __init__(self, tensors, batch_size):
        batched_tensors = [tensors[i * batch_size:(i + 1) * batch_size] for i in range((len(tensors) + batch_size - 1) // batch_size)]
        data_set = []
        for batch in batched_tensors:
            data_set.append(torch.stack(batch))

        self.batched_tensors = data_set

    def __getitem__(self, index):
        return self.batched_tensors[index]

    def __len__(self):
        return len(self.batched_tensors)


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 batch_size=100,
                 shuffle=True,
                 data_shape=(28, 28),
                 ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataloader = None
        self._test_dataloader = None
        self._data_shape = data_shape
        self._num_workers = 0
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        device = get_device()
        train_x = np.array([np.array(x) for x, _ in datasets.MNIST(self._data_dir, train=True, download=True)])
        train_y = np.array([np.array(y) for _, y in datasets.MNIST(self._data_dir, train=True, download=True)])
        train_x = train_x.astype(np.float32)
        train_x = [torch.tensor(x).to(device) for x in train_x]
        train_data_set = MNISTTensorDataSet(train_x, batch_size=self._batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_data_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        test_x = np.array([np.array(x) for x, _ in datasets.MNIST(self._data_dir, train=False, download=True)])
        test_y = np.array([np.array(y) for _, y in datasets.MNIST(self._data_dir, train=False, download=True)])
        test_x = test_x.astype(np.float32)
        test_x = [torch.tensor(x).to(device) for x in test_x]
        test_y = test_y.astype(np.float32)
        test_data_set = MNISTTensorDataSet(test_x, batch_size=self._batch_size)
        test_loader = torch.utils.data.DataLoader(
            test_data_set,
            batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers)

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
