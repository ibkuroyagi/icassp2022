#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import pathlib


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(description="Preprocess audio files.")
    parser.add_argument(
        "--datadir", type=str, required=True, help="directory to data files."
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
    config = {}
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    datadir = pathlib.Path(config["datadir"])
    data_list = sorted(list(datadir.glob("*.wav")))
    scp_path = os.path.join(config["datadir"], "wav.scp")
    with open(scp_path, "w") as f:
        for path in data_list:
            id_ = str(path).split("/")[-1].split(".")[0]
            f.write(f"{id_} {path}\n")

    logging.info(f"Successfully create scp file at {scp_path}.")


if __name__ == "__main__":
    main()
