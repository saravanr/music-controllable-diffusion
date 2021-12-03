import torch
import numpy as np
import os
from torch.utils.data import Dataset
import tqdm
import shutil

from utils.cuda_utils import get_device
from utils.file_utils import get_files_in_path
from utils.midi_utils import get_encoding
import torch.nn.functional as F

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
            # Do not pad just return none
            # sample = np.pad(sample, ((0, diff_rows), (0, 0)), constant_values=0)
            sample = None
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


class ConvertEndTimeToDuration(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample is None:
            return None

        # End time - start time
        try:
            sample.T[5] = sample.T[5]-sample.T[4]
        except Exception as _:
            return None
        return sample


class Rescale(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if sample is None:
            return None

        # Renormalize (this is because the stored .npy files have the reverse transform
        sample = np.arctanh(sample) * 254. + 127.
        return sample


class MidiDataset(Dataset):

    def __init__(self, data_dir, transform=None, combined_file=os.path.expanduser("/dev/shm/midi_features_v2_combined.npy")):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = get_files_in_path(data_dir, matching_pattern=f"*.npy")
        self.mean = 0.0
        self.std = 0.0
        self.combined_file = combined_file
        self.tensors = self.generate_tensors()

    def get_mean_and_std(self):
        return self.mean, self.std

    def generate_tensors(self):
        device = get_device()
        tensors = []
        print(f"Generating input tensors on {device}")

        #clean_features_dir = os.path.expanduser("~/midi_clean_features_v2")
        #os.makedirs(clean_features_dir, exist_ok=True)

        data_array = []
        if not os.path.exists(self.combined_file):
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

                mean = np.mean(data, axis=0)
                std = np.mean(data, axis=0)
                if True in np.isinf(mean) or True in np.isnan(mean):
                    continue

                if True in np.isinf(std) or True in np.isnan(std):
                    continue

                #shutil.copy(data_file, os.path.join(clean_features_dir, os.path.basename(data_file)))
                data_array.append(data)

            print(f"Normalizing.. {len(data_array)} samples...")
            data_array = [x for x in data_array if x is not None]
            p = np.array(data_array)
            np.save(self.combined_file, p)
        else:
            p = np.load(self.combined_file)
            data_array = p

        self.mean = np.mean(p, axis=(0, 1))
        self.std = np.std(p, axis=(0, 1))

        print(f"Generating tensors")
        index = 0
        from data.midi_data_module import MAX_MIDI_ENCODING_ROWS
        for data in tqdm.tqdm(data_array):
            # TODO: CLean this
            data = data[0:MAX_MIDI_ENCODING_ROWS]
            pitches = torch.tensor(data.T[0])
            velocity = torch.tensor(data.T[1])
            instrument = torch.tensor(data.T[2])
            program = torch.tensor(data.T[3])

            norm_data = (data - self.mean) / self.std
            norm_data = np.tanh(norm_data)

            start_times = torch.tensor(norm_data.T[4])
            duration = torch.tensor(norm_data.T[5])

            tensor = torch.vstack((pitches.T, velocity.T, instrument.T, program.T, start_times, duration)).to(device)
            tensors.append(tensor)
            #if index > 40000:
                #break
            index = index + 1

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
    output_dir = os.path.expanduser("~/midi_features_norm")
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
