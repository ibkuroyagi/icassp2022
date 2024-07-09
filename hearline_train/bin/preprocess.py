# -*- coding: utf-8 -*-


"""Perform preprocessing and raw feature extraction."""
import argparse
import logging
import os
import librosa
import pathlib
import torchaudio
import numpy as np
import yaml

from tqdm import tqdm

from hearline_train.utils import logmelfilterbank  # noqa: E402
from hearline_train.utils import write_hdf5  # noqa: E402


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features."
    )
    parser.add_argument(
        "--datadir", required=True, type=str, help="directory including flac files.",
    )
    parser.add_argument(
        "--dumpdir", type=str, required=True, help="directory to dump feature files."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    sr = config["sr"]
    datadir = pathlib.Path(config["datadir"])
    data_list = sorted(list(datadir.glob("*.wav")))
    print(len(data_list))
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=config["fft_size"],
        hop_length=config["hop_size"],
        n_mels=config["num_mels"],
        normalized=True,
        f_min=config["fmin"],
        f_max=config["fmax"],
    )
    # process each data
    if not os.path.exists(config["dumpdir"]):
        os.makedirs(config["dumpdir"], exist_ok=True)
    with open(os.path.join(config["dumpdir"], "feats.scp"), "w") as f:
        for wav_path in tqdm(data_list):
            wave_id = str(wav_path).split("/")[-1].split(".")[0]
            feat_path = os.path.join(config["dumpdir"], f"{wave_id}.h5")
            # if os.path.isfile(wav_path):
            #     if os.path.getsize(wav_path) < 100000:
            #         os.remove(wav_path)
            #         os.remove(feat_path)
            #         continue
            # else:
            #     if os.path.isfile(feat_path):
            #         os.remove(feat_path)
            # if os.path.isfile(feat_path):
            #     f.write(f"{wave_id} {feat_path}\n")
            #     continue
            # x, _ = librosa.load(path=wav_path, sr=sr)
            x, sample_rate = torchaudio.load(wav_path)
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            x = resample_transform(x)
            logging.info(f"{wave_id} {x.shape} {feat_path}")
            if x.shape[1] >= 16000 * 5:
                mel_spectrogram = mel_spectrogram_transform(x)
                feat = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
                write_hdf5(
                    feat_path, "wave", x[0].numpy().astype(np.float32),
                )
                write_hdf5(
                    feat_path, "feat", feat.numpy().astype(np.float32),
                )
                f.write(f"{wave_id} {feat_path}\n")
    logging.info("Successfully finished preprocess.")


if __name__ == "__main__":
    main()
