import numpy as np
import librosa
from torch.utils.data import Dataset
from multiprocessing import Manager
from hearline_train.utils import read_hdf5


class AudiosetDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        path_list,
        keys=["feats"],
        allow_cache=False,
    ):
        """Initialize dataset.
        Args:
            path_list: (list): List of dataset.
            keys: (list): List of key of dataset.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        self.path_list = path_list
        self.keys = keys
        self.allow_cache = allow_cache
        self.manager = Manager()
        self.caches = self.manager.list()
        self.caches += [() for _ in range(len(path_list))]

    def __getitem__(self, idx):
        """Get specified idx items.
        Args:
            idx (int): Index of the item.
        Returns:
            items: Dict
                wave: (ndarray) wave (T,).
        """
        if self.allow_cache and (len(self.caches[idx]) != 0):
            return self.caches[idx]

        items = {}
        for key in self.keys:
            items[key] = read_hdf5(self.path_list[idx], key)
            if key == "feat":
                items[key] = items[key].astype(np.float32)
        if self.allow_cache:
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.
        Returns:
            int: The length of dataset.
        """
        return len(self.path_list)


class WaveAudiosetDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        path_list,
        allow_cache=False,
    ):
        """Initialize dataset.
        Args:
            path_list: (list): List of dataset.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        self.path_list = path_list
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(path_list))]

    def __getitem__(self, idx):
        """Get specified idx items.
        Args:
            idx (int): Index of the item.
        Returns:
            wave: (ndarray) Feature (T,).
        """
        if self.allow_cache and (len(self.caches[idx]) != 0):
            return self.caches[idx]
        # wave, _ = librosa.load(path=self.path_list[idx], sr=16000)
        wave = read_hdf5(self.path_list[idx], "wave")
        if self.allow_cache:
            self.caches[idx] = wave
        return wave

    def __len__(self):
        """Return dataset length.
        Returns:
            int: The length of dataset.
        """
        return len(self.path_list)


class EpochWaveAudiosetDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(self, path_list, keys=["anchor", "positive"], allow_cache=False):
        """Initialize dataset.
        Args:
            path_list: (list): List of dataset.
            keys: ["anchor", "positive", "shifted_anchor", "pitch_shift"]
        """
        self.path_list = path_list
        self.keys = keys
        self.allow_cache = allow_cache
        if allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(path_list))]

    def __getitem__(self, idx):
        """Get specified idx items.
        Args:
            idx (int): Index of the item.
        """
        if self.allow_cache and (len(self.caches[idx]) != 0):
            return self.caches[idx]
        items = {}
        for key in self.keys:
            items[key] = read_hdf5(self.path_list[idx], key)
        if self.allow_cache:
            self.caches[idx] = items
        return items

    def __len__(self):
        """Return dataset length.
        Returns:
            int: The length of dataset.
        """
        return len(self.path_list)
