# -*- coding: utf-8 -*-

"""Utility functions."""

import fnmatch
import random
import logging
import os
import sys

import h5py
import librosa
import numpy as np
import torch
import torch.nn as nn


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def scp_to_list(scp_path: str, shuffle=False):
    """Load scp file and return list of path.

    Args:
        scp_path (str): path of scp
    Returns:
        List of scp file's path.
    """
    scp_list = []
    with open(scp_path, "r") as f:
        cnt = 0
        error_cnt = 0
        for line in f:
            path = line.split(" ")[1].replace("\n", "")
            cnt += 1
            scp_list.append(path)
            # if os.path.exists(path):
            #     scp_list.append(path)
            # else:
            #     logging.warning(f"{path} doesn't exist.")
            #     error_cnt += 1
    logging.info(f"{scp_path}: {cnt}, error file: {error_cnt}")
    if shuffle:
        random.shuffle(scp_list)
        logging.info(f"{scp_path} was shuffled.")
    return scp_list


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.
    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.
    Return:
        any: Dataset values.
    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(
            f"There is no such a data in hdf5 file. ({hdf5_path} @ {hdf5_name})"
        )
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.
    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-5,
):
    """Compute log-Mel filterbank feature.
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def toSyncBatchNorm(model):
    bn_name_list = []
    for i, (name, modul) in enumerate(model.named_modules()):
        if isinstance(modul, nn.modules.batchnorm.BatchNorm1d) or isinstance(
            modul, nn.modules.batchnorm.BatchNorm2d
        ):
            bn_name_list.append(name)
    for name in bn_name_list:
        n_property = len(name.split("."))
        for i, n in enumerate(name.split(".")):
            if i == 0:
                batchnorm = getattr(model, n)
                new_batchnorm = getattr(model, n)
            else:
                batchnorm = getattr(batchnorm, n)
                if i < n_property - 1:
                    new_batchnorm = getattr(new_batchnorm, n)
        num_features = getattr(batchnorm, "num_features")
        setattr(new_batchnorm, n, nn.modules.batchnorm.SyncBatchNorm(num_features))
    return model


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
