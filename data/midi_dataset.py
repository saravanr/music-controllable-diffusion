import torch
import numpy as np
import os
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tqdm
from utils.file_utils import get_files_in_path
from utils.midi_utils import get_encoding

def data_loader_collate_fn(batch):
    """
    Custom collate function to remove Nones without throwing exceptions
    :param batch: The batch
    :return: Collate Fun
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Trim(object):
    """
    Trims the sample to appropriate size
    """

    def __init__(self, max_rows):
        self._max_rows = max_rows

    def __call__(self, sample):
        if sample is None:
            return None
        diff_rows = self._max_rows - sample.shape[0]
        if diff_rows > 0:
            sample = np.pad(sample, ((0, diff_rows), (0, 0)), constant_values=0)
        else:
            sample = sample[0:self._max_rows]

        return sample


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
        self.data_files = get_files_in_path(data_dir, matching_pattern=f"*.npy")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            raise NotImplementedError(f"Torch indexes are not implemented")
        numpy_file_name = self.data_files[index]
        sample = None

        try:
            if os.path.exists(numpy_file_name):
                sample = np.load(numpy_file_name)
        except Exception as _e:
            sample = None
            print(f"Unable to load {numpy_file_name} -- {_e}")

        if sample is not None and self.transform:
            sample = self.transform(sample)

        return sample


def process(files):
    import rmi2mid
    output_dir = os.path.expanduser("~/midi_processed")
    print(f"Data set length = {len(files)}")
    count = 0
    partition = 0
    pid = os.getpid()
    for file in tqdm.tqdm(files):
        try:
            partition_dir = os.path.join(output_dir, f"{partition}-{pid}")
            os.makedirs(partition_dir, exist_ok=True)
            output_file = os.path.join(partition_dir, f"{os.path.basename(file)}.npy")
            if os.path.exists(output_file):
                continue
            encoding = get_encoding(file)
            np.save(output_file, encoding)
            count = count + 1
            if count % 2000 == 0:
                partition = partition + 1
            print(f"Saved to {output_file}")
        except Exception as _e:
            output_file = f"{file}.mid"
            try:
                rmi2mid.rmi2mid(file, output_file)
                encoding = get_encoding(output_file)
                np.save(output_file, encoding)
                print(f"Converted and saved to {output_file}")
                count = count + 1
                shutil.move(file, os.path.join(os.path.expanduser('~/midijunk'), os.path.basename(file)))
            except Exception as e:
                pass


if __name__ == "__main__":
    import os
    import multiprocessing as mp
    _data_dir = os.path.expanduser("~/midi/")
    _output_dir = os.path.expanduser("~/midi_processed")
    os.makedirs(_output_dir, exist_ok=True)
    _files = get_files_in_path(_data_dir, matching_pattern="*.mid")
    num_proc = mp.cpu_count() - 1
    parts = [_files[i:i + num_proc] for i in range(0, len(_files), num_proc)]
    p = mp.Pool(num_proc)
    p.map(process, parts)
    p.join()
