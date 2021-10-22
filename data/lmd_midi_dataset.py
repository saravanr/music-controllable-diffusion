import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.file_utils import get_files_in_path
import magenta

class LMDMidiDataset(Dataset):

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
        mid = MidiFile(midi_file_name)

        # Midi types:
        # 0 -> Single track
        # 1 -> Synchronous
        # 2 -> Asynchronous
        midi_type = mid.type
        print(mid)
        sample = mid
        magenta.scripts
        if self.transform:
            sample = self.transform(sample)


        return sample


if __name__ == "__main__":
    import os
    _data_dir = os.path.expanduser("~/midi/")
    _data_set = LMDMidiDataset(_data_dir)

    print(f"Data set length = {len(_data_set)}")
    print(f"Data set sample = {_data_set[10]}")

    _dataloader = DataLoader(_data_set, batch_size=4, shuffle=True, num_workers=8)

    for _batch, _sample in enumerate(_dataloader):
        print(f"_batch={_batch}")

