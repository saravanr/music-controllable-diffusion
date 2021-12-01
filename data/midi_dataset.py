import torch
import numpy as np
import os
from torch.utils.data import Dataset
import tqdm

from utils.cuda_utils import get_device
from utils.file_utils import get_files_in_path
from utils.midi_utils import get_encoding


def data_loader_collate_fn(batch):
    """
    Custom collate function to remove Nones without throwing exceptions
    :param batch: The batch
    :return: Collate Fun
    """
    batch = list(filter(lambda x: x is not None and len(x) != 0, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Trim(object):
    """
    Trims the sample to appropriate size
    """

    def __init__(self, max_rows):
        self._max_rows = max_rows

    def __call__(self, sample):
        if sample is None or len(sample) == 0:
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
        self.data_files = get_files_in_path(data_dir, matching_pattern=f"*.npy")[0:50000]
        self.tensors = self.generate_tensors()

    def generate_tensors(self):
        device = get_device()
        tensors = []
        print(f"Generating input tensors on {device}")
        for data_file in tqdm.tqdm(self.data_files):
            if not os.path.exists(data_file):
                continue
            try:
                data = np.load(data_file)
            except Exception as e:
                print(f"Unable to load {data_file} -- {e}")
                continue

            if self.transform:
                data = self.transform(data)

            if data is None:
                continue
            tensor = torch.Tensor(data).to(device)
            tensors.append(tensor)

        return tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            raise NotImplementedError(f"Torch indexes are not implemented")

        return self.tensors[index]


def process(files):
    import rmi2mid
    output_dir = os.path.expanduser("~/midi_features")
    print(f"Data set length = {len(files)}")
    count = 0
    partition = 0
    pid = os.getpid()
    existing_files = get_files_in_path(output_dir, matching_pattern="*.npy")
    existing_files = [os.path.basename(x) for x in existing_files]
    existing_files = set(existing_files)
    print(f"Existing files = {len(existing_files)}")
    for file in files:
        try:
            partition_dir = os.path.join(output_dir, f"{partition}-{pid}")
            os.makedirs(partition_dir, exist_ok=True)
            output_file = os.path.join(partition_dir, f"{os.path.basename(file)}.npy")
            if os.path.basename(output_file) in existing_files:
                print(f"Skipping {file}")
                continue
            print(f"Processing {file}")
            encoding = get_encoding(file)
            np.save(output_file, encoding)
            count = count + 1
            if count % 2000 == 0:
                partition = partition + 1
        except Exception as _e:
            output_file = f"{file}.mid"
            try:
                rmi2mid.rmi2mid(file, output_file)
                encoding = get_encoding(output_file)
                np.save(output_file, encoding)
                print(f"Converted and saved to {output_file}")
                count = count + 1
            except Exception as e:
                pass


def generate_numpy_files():
    import os
    import multiprocessing as mp
    output_dir = os.path.expanduser("~/midi_features")
    os.makedirs(output_dir, exist_ok=True)
    files = get_files_in_path(_data_dir, matching_pattern="*.mid")
    num_proc = 1
    if num_proc > 1:
        parts = [files[i:i + num_proc] for i in range(0, len(files), num_proc)]
        p = mp.Pool(num_proc)
        p.map(process, parts)
    else:
        process(files)


def find_min_max():
    _files = get_files_in_path(_data_dir, matching_pattern="*.npy")
    print(f"Number of files - {len(_files)}")
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _global_min = torch.Tensor(np.zeros(8)).to(_device)
    _global_max = torch.Tensor(np.zeros(8)).to(_device)
    for _file in tqdm.tqdm(_files):
        _sample = np.load(_file)
        try:
            _sample = torch.Tensor(_sample).to(_device)

            _min = torch.min(_sample, dim=-2).values
            _max = torch.max(_sample, dim=-2).values
            _global_min = torch.minimum(_min, _global_min)
            _global_max = torch.maximum(_max, _global_max)
        except IndexError as _e:
            print(f"Unable to process {_file} with shape {_sample.shape} -> {_e}")

    print(f"Min - {_global_min}")
    print(f"Max - {_global_max}")


if __name__ == "__main__":
    _data_dir = os.path.expanduser("~/midi/")
    generate_numpy_files()
