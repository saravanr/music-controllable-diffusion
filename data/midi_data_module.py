import os.path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms

from data.midi_dataset import MidiDataset, Trim, Reshape, data_loader_collate_fn

MAX_MIDI_ENCODING_ROWS = 1600
MIDI_ENCODING_WIDTH = 6


class MidiDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 batch_size=10,
                 shuffle=True,
                 data_shape=(MAX_MIDI_ENCODING_ROWS, MIDI_ENCODING_WIDTH),
                 ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._test_dataloader = None
        self._val_dataloader = None
        self._data_shape = data_shape
        self._num_workers = 0
        self._shuffle = shuffle

    def setup(self, stage: Optional[str] = None) -> None:
        data_set = MidiDataset(self._data_dir,
                               transform=transforms.Compose(
                                   [Trim(max_rows=MAX_MIDI_ENCODING_ROWS), Reshape(self._data_shape)]))
        train_size = int(0.7 * len(data_set))
        test_size = int((len(data_set) - train_size) / 2.0)
        val_size = len(data_set) - train_size - test_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data_set,
                                                                                 [train_size, val_size, test_size])
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._val_dataset = val_dataset
        self._train_dataloader = DataLoader(self._train_dataset,
                                            batch_size=self._batch_size,
                                            shuffle=self._shuffle,
                                            num_workers=self._num_workers,
                                            collate_fn=data_loader_collate_fn
                                            )
        self._test_dataloader = DataLoader(self._test_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=self._shuffle,
                                           num_workers=self._num_workers,
                                           collate_fn=data_loader_collate_fn,
                                           )

        self._val_dataloader = DataLoader(self._val_dataset,
                                          batch_size=self._batch_size,
                                          shuffle=self._shuffle,
                                          num_workers=self._num_workers,
                                          collate_fn=data_loader_collate_fn,
                                          )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._test_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._val_dataloader

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


if __name__ == "__main__":
    import tqdm

    # Gathers stats on the data loader
    _midi_module = MidiDataModule(data_dir=os.path.expanduser("~/midi_features"),
                                  batch_size=20)
    _midi_module.setup()
    _train = _midi_module.train_dataloader()
    _test = _midi_module.test_dataloader()
    _val = _midi_module.val_dataloader()

    _global_min = None
    _global_max = None


    def _find_min_max(_data_set, _global_min, _global_max):
        for sample in tqdm.tqdm(_data_set):
            _min = torch.min(sample, dim=2).values
            _max = torch.max(sample, dim=2).values
            if _global_min is None:
                _global_min = _min

            if _global_max is None:
                _global_max = _max
            _global_min = torch.minimum(_min, _global_min)
            _global_max = torch.maximum(_max, _global_max)
        return _global_min, _global_max


    with torch.no_grad():
        _global_min, _global_max = _find_min_max(_train, _global_min, _global_max)
        _global_min, _global_max = _find_min_max(_test, _global_min, _global_max)
        _global_min, _global_max = _find_min_max(_val, _global_min, _global_max)

    print(f"Global mins = {_global_min}")
    print(f"Global max = {_global_max}")
