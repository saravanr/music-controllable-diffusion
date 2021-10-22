import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.file_utils import get_files_in_path
from utils.midi_utils import MidiConverter
from torchvision import transforms


def data_loader_collate_fn(batch):
    """
    Custom collate function to remove Nones without throwing exceptions
    :param batch: The batch
    :return: Collate Fun
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Reshape(object):
    """
    Reshapes the sample to be
    """

    def __init__(self, shape):
        self._shape = shape

    def __call__(self, sample):
        if sample is None:
            return None
        sample = sample.reshape(self._shape)
        return sample


class MidiDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = get_files_in_path(data_dir, matching_pattern="*.mid")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            raise NotImplementedError(f"Torch indexes are not implemented")
        midi_file_name = self.data_files[index]

        try:
            sample = MidiConverter().load_file(midi_file_name).to_nd_array()
        except Exception as e:
            print(f"Unable to load {midi_file_name} possibly corrupt? - {e}")
            sample = None

        if sample is not None and self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    import os
    data_shape = (245, 286)
    _data_dir = os.path.expanduser("~/midi/")
    _data_set = MidiDataset(_data_dir,
                            transform=transforms.Compose([Reshape(data_shape)]))

    print(f"Data set length = {len(_data_set)}")
    print(f"Data set sample = {_data_set[10]}")

    _dataloader = DataLoader(_data_set,
                             batch_size=4,
                             shuffle=True,
                             num_workers=15,
                             collate_fn=data_loader_collate_fn,

                             )

    for _batch, _sample in enumerate(_dataloader):
        print(f"_batch={_batch} _sample_shape={_sample.shape}")
