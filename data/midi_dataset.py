import torch
import numpy as np
import os
import shutil
from torch.utils.data import Dataset
import tqdm
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


class Normalize(object):
    """
    Normalizes the input to a range of [-1.0 to 1.0]
    The Octuple Token in OctupleMIDI encoding using the following scheme:
    [Time Sig, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity]
    The corresponding number of variations are:
    [254, [16-256], [256], [256], [128], 128, 128, 128]

    Min - tensor([0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
    Max - tensor(
    """

    def __init__(self, max_values=np.array([9280., 127., 128., 255., 127., 31., 157., 48.])):
        self._max_values = max_values

    def __call__(self, sample):
        if sample is None or len(sample) == 0:
            return sample
        sample = np.divide(sample, self._max_values).astype(np.float32)
        return sample


class Trim(object):
    """
    Trims the sample to appropriate size
    """

    def __init__(self, max_rows):
        self._max_rows = max_rows

    def __call__(self, sample):
        if sample is None or len(sample)==0:
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


def generate_numpy_files():
    import os
    import multiprocessing as mp
    output_dir = os.path.expanduser("~/midi_processed")
    os.makedirs(output_dir, exist_ok=True)
    files = get_files_in_path(_data_dir, matching_pattern="*.mid")
    num_proc = mp.cpu_count() - 1
    parts = [files[i:i + num_proc] for i in range(0, len(files), num_proc)]
    p = mp.Pool(num_proc)
    p.map(process, parts)


if __name__ == "__main__":
    _data_dir = os.path.expanduser("~/midi_processed/")
    _files = get_files_in_path(_data_dir, matching_pattern="*.npy")
    print(f"Number of files - {len(_files)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _global_min = torch.Tensor(np.zeros(8)).to(device)
    _global_max = torch.Tensor(np.zeros(8)).to(device)
    for _file in tqdm.tqdm(_files):
        _sample = np.load(_file)
        try:
            _sample = torch.Tensor(_sample).to(device)

            _min = torch.min(_sample, dim=-2).values
            _max = torch.max(_sample, dim=-2).values
            _global_min = torch.minimum(_min, _global_min)
            _global_max = torch.maximum(_max, _global_max)
        except IndexError as _e:
            print(f"Unable to process {_file} with shape {_sample.shape} -> {_e}")

    print(f"Min - {_global_min}")
    print(f"Max - {_global_max}")
