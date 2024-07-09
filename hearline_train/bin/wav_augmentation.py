# -*- coding: utf-8 -*-


"""Apply wave form augmentation."""
import argparse
import logging
import os
import librosa
import pathlib
import soundfile as sf
import numpy as np

from tqdm import tqdm


def main():
    """Run preprocessing process."""
    # https://www.wizard-notes.com/entry/music-analysis/librosa-pitch-shift
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features."
    )
    parser.add_argument(
        "--datadir",
        required=True,
        type=str,
        help="directory including flac files.",
    )
    parser.add_argument(
        "--target_datadir", type=str, required=True, help="target directory."
    )
    parser.add_argument("--factor", type=float, required=True, help="shift rate.")
    parser.add_argument("--fc", type=str, required=True, help="augmentation function.")
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
    config = {"sr": 16000}
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    datadir = pathlib.Path(config["datadir"])
    data_list = sorted(list(datadir.glob("*.wav")))

    # process each data
    if not os.path.exists(config["target_datadir"]):
        os.makedirs(config["target_datadir"], exist_ok=True)
    with open(os.path.join(config["target_datadir"], "wav.scp"), "w") as f:
        for wav_path in tqdm(data_list):
            wave_id = str(wav_path).split("/")[-1].split(".")[0]
            aug_wav_path = os.path.join(config["target_datadir"], f"{wave_id}.wav")
            if os.path.isfile(aug_wav_path):
                f.write(f"{wave_id} {aug_wav_path}\n")
                continue
            if os.path.isfile(wav_path):
                if os.path.getsize(wav_path) < 100000:
                    os.remove(wav_path)
                    continue
            x, _ = librosa.load(path=wav_path, sr=config["sr"])
            # extract feature
            if args.fc == "pitch":
                aug_x = librosa.effects.pitch_shift(
                    x, sr=config["sr"], n_steps=np.log2(args.factor)
                )
            elif args.fc == "time":
                aug_x = librosa.effects.time_stretch(x, args.factor)
            sf.write(aug_wav_path, aug_x, config["sr"])
            f.write(f"{wave_id} {aug_wav_path}\n")
    logging.info("Successfully finished augmentation.")


if __name__ == "__main__":
    main()
