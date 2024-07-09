import argparse
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from hearline_train import models
from hearline_train.utils import scp_to_list
from hearline_train.trainers import COLATrainer
from hearline_train import optimizers
from hearline_train.datasets import EpochWaveAudiosetDataset
from hearline_train.datasets import EpochWaveCollater

# from IPython import embed


def main():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--train_audioset_scp",
        type=str,
        required=True,
        help="Path of Audioset's scp file for training.",
    )
    parser.add_argument(
        "--valid_audioset_scp",
        type=str,
        required=True,
        help="Path of Audioset's scp file for validation.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path of config file.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path of output dir.",
    )
    parser.add_argument(
        "--mode", type=str, default="wave", help="wave or feat.",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="checkpoint file path to resume training. (default='')",
    )
    parser.add_argument(
        "--n_gpus", default=1, type=int, help="The number of gpu. (default=1)",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    args = parser.parse_args()
    args.distributed = False
    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")
    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.getLogger("matplotlib.font_manager").disabled = True
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    logging.info("Start main program.")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.cuda.set_device(args.rank)
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            logging.info("Start to init distributed setting.")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            logging.info("Successfully init distributed setting.")
    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    for key, value in config.items():
        logging.info(f"{key} = {value}")
    # for dataset
    logging.info("Set Dataset.")
    train_audioset_path_list = scp_to_list(
        config["train_audioset_scp"], shuffle=args.mode == "epoch_wave"
    )
    valid_audioset_path_list = scp_to_list(config["valid_audioset_scp"])
    keys = ["anchor", "positive"]
    if config.get("pitch_shift", 0) != 0:
        keys += ["pitch_shift", "shifted_anchor"]
    if args.mode == "wave":
        from hearline_train.datasets import WaveAudiosetDataset
        from hearline_train.datasets import WaveTrainCollater

        train_audioset_dataset = WaveAudiosetDataset(
            path_list=train_audioset_path_list, allow_cache=True,
        )
        collater_fc = WaveTrainCollater(
            sample_rate=config["sr"],
            sec=config["sec"],
            hop_size=config["hop_size"],
            noise=config.get("noise", 0.001),
            pitch_shift=config.get("pitch_shift", 0),
            embed_time_frame=config.get("embed_time_frame", 12),
        )
    elif args.mode == "epoch_wave":
        train_audioset_dataset = EpochWaveAudiosetDataset(
            path_list=train_audioset_path_list, keys=keys,
        )
        collater_fc = EpochWaveCollater(
            use_pitch_shift=config.get("pitch_shift", 0) != 0,
            embed_time_frame=config.get("embed_time_frame", 12),
        )
    elif args.mode == "feat":
        from hearline_train.datasets import AudiosetDataset
        from hearline_train.datasets import FeatTrainCollater

        train_audioset_dataset = AudiosetDataset(
            path_list=train_audioset_path_list, keys=["feat"], allow_cache=False,
        )
        valid_audioset_dataset = AudiosetDataset(
            path_list=valid_audioset_path_list, keys=["feat"], allow_cache=False
        )
        collater_fc = FeatTrainCollater(max_frames=config["max_frames"])
    # valid_audioset_dataset = EpochWaveAudiosetDataset(
    #     path_list=valid_audioset_path_list,
    #     keys=keys,
    #     allow_cache=True,
    # )
    # valid_collater_fc = EpochWaveCollater(
    #     use_pitch_shift=config.get("pitch_shift", 0) != 0,
    #     embed_time_frame=config.get("embed_time_frame", 12),
    # )
    valid_audioset_dataset = WaveAudiosetDataset(
        path_list=valid_audioset_path_list, allow_cache=True,
    )
    valid_collater_fc = collater_fc = WaveTrainCollater(
        sample_rate=config["sr"],
        sec=config["sec"],
        hop_size=config["hop_size"],
        noise=config.get("noise", 0.001),
        pitch_shift=config.get("pitch_shift", 0),
        embed_time_frame=config.get("embed_time_frame", 12),
    )
    # for sampler
    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_audioset_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        valid_sampler = DistributedSampler(
            valid_audioset_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        valid_sampler = None

    audioset_loader = {
        "train": DataLoader(
            train_audioset_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            collate_fn=collater_fc,
            sampler=train_sampler,
            shuffle=False if args.distributed else True,
            drop_last=True,
        ),
        "valid": DataLoader(
            valid_audioset_dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            collate_fn=valid_collater_fc,
            sampler=valid_sampler,
            shuffle=False,
            drop_last=True,
        ),
    }
    logging.info("Set model.")
    model_class = getattr(models, config.get("model_type", "EfficientNet_b0"),)
    model = model_class(**config["model_params"])
    if args.distributed:
        from hearline_train.utils import toSyncBatchNorm

        model = toSyncBatchNorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank])

    logging.info(model)
    if config["similar_fc_type"] == "BilinearSimilarity":
        from hearline_train.losses import BilinearSimilarity

        similar_fc = BilinearSimilarity(n_embedding=config["n_embedding"])
    elif config["similar_fc_type"] == "CosineSimilarity":
        from hearline_train.losses import CosineSimilarity

        similar_fc = CosineSimilarity(temperature=config["temperature"])
    logging.info("Set optimizer.")
    params = []
    for name, param in model.named_parameters():
        params.append(param)
    for name, param in similar_fc.named_parameters():
        params.append(param)
    optimizer_class = getattr(optimizers, config.get("optimizer_type", "Adam"))
    optimizer = optimizer_class(params, **config["optimizer_params"])
    scheduler = None
    logging.info("Set trainer.")
    # define trainer
    trainer = COLATrainer(
        steps=1,
        epochs=1,
        data_loader=audioset_loader,
        train_sampler=train_sampler,
        model=model.to(device),
        optimizer=optimizer,
        scheduler=scheduler,
        similar_fc=similar_fc.to(device),
        config=config,
        device=device,
    )
    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume, load_only_params=False)
        logging.info(f"Successfully resumed from {args.resume}.")
    # run training loop
    try:
        trainer.run(args.rank)
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    main()
