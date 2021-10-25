from typing import Optional
import multiprocessing as mp
import torch
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from data.midi_dataset import MidiDataset, Reshape, data_loader_collate_fn


class MidiDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 batch_size=4,
                 shuffle=False,
                 data_shape=(245, 286),
                 ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._data_shape = data_shape
        self._num_workers = 12
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        data_set = MidiDataset(self._data_dir,
                               transform=transforms.Compose([Reshape(self._data_shape)]))
        train_size = int(0.7 * len(data_set))
        test_size = int((len(data_set) - train_size) / 2.0)
        val_size = len(data_set) - train_size - test_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data_set,
                                                                                 [train_size, val_size, test_size])
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._val_dataset = val_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        _dataloader = DataLoader(self._train_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=self._shuffle,
                                 num_workers=self._num_workers,
                                 pin_memory=True,
                                 collate_fn=data_loader_collate_fn,
                                 )
        return _dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        _dataloader = DataLoader(self._test_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=self._shuffle,
                                 num_workers=self._num_workers,
                                 pin_memory=True,
                                 collate_fn=data_loader_collate_fn,
                                 )
        return _dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        _dataloader = DataLoader(self._val_dataset,
                                 batch_size=self._batch_size,
                                 shuffle=self._shuffle,
                                 num_workers=self._num_workers,
                                 pin_memory=True,
                                 collate_fn=data_loader_collate_fn,
                                 )
        return _dataloader

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
