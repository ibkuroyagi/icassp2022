import logging
import os
import json
from collections import OrderedDict
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter


class COLATrainer(object):
    """Customized trainer module for COLA training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        optimizer,
        scheduler,
        similar_fc,
        config,
        train_sampler=None,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders.
            model (dict): Dict of models..
            optimizer (object): Optimizers.
            scheduler (object): Schedulers.
            similar_fc (object): Similar function.
            train_sampler (object): train sampler for multi gpu.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.
        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.similar_fc = similar_fc
        self.train_sampler = train_sampler
        self.config = config
        self.device = device
        self.finish_train = False
        self.best_score = 1e6
        self.total_train_loss = defaultdict(float)
        self.total_valid_loss = defaultdict(float)
        self.last_checkpoint = ""
        self.forward_count = 0
        # self.writer = SummaryWriter(config["outdir"])
        self.use_pitch_shift = config.get("pitch_shift", 0) != 0
        if self.use_pitch_shift:
            self.mse = torch.nn.MSELoss()
        self.use_frame_loss = config.get("frame_criterion_params", None) is not None
        if self.use_frame_loss:
            from hearline_train.losses import WideArangeCrossEntropy
            from hearline_train.losses import FramewiseContrastiveLoss

            framewise_criterion = WideArangeCrossEntropy(
                **config["frame_criterion_params"], device=device
            )
            self.frame_criterion = FramewiseContrastiveLoss(
                similar_fc, framewise_criterion
            )

    def run(self, rank=None):
        """Run training."""
        if self.config["distributed"]:
            torch.cuda.set_device(rank)
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        self.save_checkpoint(
            os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl",),
            save_model_only=False,
        )
        logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")
        logging.info("Finished training.")
        if self.config["distributed"]:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, checkpoint_path, save_model_only=True):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            save_model_only (bool): Whether to save model parameters only.
        """
        state_dict = {
            "steps": self.steps,
            "epochs": self.epochs,
            "best_score": self.best_score,
        }
        self.model.cpu()
        if self.config["distributed"]:
            state_dict["model"] = self.model.module.state_dict()
        else:
            state_dict["model"] = self.model.state_dict()
        if not save_model_only:
            state_dict["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state_dict["scheduler"] = self.scheduler.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)
        self.last_checkpoint = checkpoint_path
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.best_score = state_dict.get("best_score", 0)
            logging.info(
                f"Steps:{self.steps}, Epochs:{self.epochs}, BEST score:{self.best_score}"
            )
            if (self.optimizer is not None) and (
                state_dict.get("optimizer", None) is not None
            ):
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if (self.scheduler is not None) and (
                state_dict.get("scheduler", None) is not None
            ):
                self.scheduler.load_state_dict(state_dict["scheduler"])

    def _train_step(self, batch):
        """Train model one step."""
        # shifted_anchor = 時間同じ,かつ,拡張違う(ピッチ分類ではフレーム単位での出力を受け取る)
        # cola:CE (anchor & positive), (shifted_anchor, positive)
        # pitch:MSE (anchor & shifted_anchor)
        # frame:CE (anchor & anchor), (positive, positive), (shifted_anchor, shifted_anchor)
        # for k, v in batch.items():
        #     logging.info(f"{k}:{v}")
        y_anchors = self.model(batch["anchors"].to(self.device))
        y_positives = self.model(batch["positives"].to(self.device))
        y_similarities = self.similar_fc(
            y_anchors["embedding_z"], y_positives["embedding_z"]
        )
        y = torch.arange(y_anchors["embedding_z"].size(0), device=self.device)
        loss = F.cross_entropy(y_similarities, y)
        sim_loss = loss.item()
        add_loss_list = []
        if self.use_pitch_shift:
            # clip level contrastive loss
            y_shifted_anchors = self.model(batch["shifted_anchors"].to(self.device))
            # y_shifted_similarities = self.similar_fc(
            #     y_shifted_anchors["embedding_z"], y_positives["embedding_z"]
            # )
            # shifted_sim_loss = F.cross_entropy(y_shifted_similarities, y)
            # loss += shifted_sim_loss
            # sim_loss = (sim_loss + shifted_sim_loss.item()) / 2
            # pitch shift loss
            pitch_shift = y_anchors["pitch_shift"] - y_shifted_anchors["pitch_shift"]
            pitch_loss = self.mse(
                pitch_shift, batch["pitch_shift"].to(self.device)
            ) * self.config.get("pitch_lambda", 1.0)
            add_loss_list.append(pitch_loss)
            # loss += pitch_loss
        if self.use_frame_loss:
            # frame level contrastive loss
            frame_loss = self.frame_criterion(
                y_anchors["frame_embedding_z"], y_anchors["frame_embedding_z"]
            )
            frame_loss += self.frame_criterion(
                y_positives["frame_embedding_z"], y_positives["frame_embedding_z"]
            )
            frame_loss_cnt = 2
            if self.use_pitch_shift:
                frame_loss += self.frame_criterion(
                    y_shifted_anchors["frame_embedding_z"],
                    y_shifted_anchors["frame_embedding_z"],
                )
                #     frame_loss += self.frame_criterion(
                #         y_anchors["frame_embedding_z"],
                #         y_shifted_anchors["frame_embedding_z"],
                #     )
                frame_loss_cnt += 1
            frame_loss *= self.config.get("frame_lambda", 1.0)
            frame_loss /= frame_loss_cnt
            # loss += frame_loss
            add_loss_list.append(frame_loss)
        if self.config.get("loss_type", "sum") == "sum":
            for add_loss in add_loss_list:
                loss += add_loss
        elif self.config.get("loss_type", "sum") == "product":
            tmp = 1
            for add_loss in add_loss_list:
                tmp += add_loss
            loss *= tmp

        if not torch.isnan(loss):
            loss = loss / self.config["accum_grads"]
            loss.backward()
            self.forward_count += 1
            if self.forward_count == self.config["accum_grads"]:
                # update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.forward_count = 0

                # update scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # update counts
                self.steps += 1
                self.tqdm.update(1)
                self._check_train_finish()

        else:
            logging.warn("Loss contained NaN. Couldn't back-poropagat.")
        # report
        self.total_train_loss["train/loss"] += loss.item()
        self.total_train_loss["train/smi_loss"] += sim_loss / self.config["accum_grads"]
        _, predicted = torch.max(y_similarities, 1)
        self.total_train_loss["train/acc"] += (
            predicted == y
        ).double().mean().detach().cpu().numpy() / self.config["accum_grads"]
        if self.use_pitch_shift:
            self.total_train_loss["train/pitch_loss"] += (
                pitch_loss.item() / self.config["accum_grads"]
            )
        if self.use_frame_loss:
            self.total_train_loss["train/frame_loss"] += (
                frame_loss.item() / self.config["accum_grads"]
            )

    def _train_epoch(self):
        """Train model one epoch."""
        if self.config["distributed"]:
            torch.distributed.barrier()
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_valid_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # log
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({train_steps_per_epoch} steps per epoch)."
        )

        # update
        self.epochs += 1

        if self.config["distributed"]:
            self.train_sampler.set_epoch(self.epochs)

    @torch.no_grad()
    def _valid_step(self, batch):
        """Validate model one step."""
        y_anchors = self.model(batch["anchors"].to(self.device))
        y_positives = self.model(batch["positives"].to(self.device))
        y_similarities = self.similar_fc(
            y_anchors["embedding_z"], y_positives["embedding_z"]
        )
        y = torch.arange(y_anchors["embedding_z"].size(0), device=self.device)
        loss = F.cross_entropy(y_similarities, y)
        sim_loss = loss.item()
        add_loss_list = []
        if self.use_pitch_shift:
            y_shifted_anchors = self.model(batch["shifted_anchors"].to(self.device))
            # y_shifted_similarities = self.similar_fc(
            #     y_shifted_anchors["embedding_z"], y_positives["embedding_z"]
            # )
            # shifted_sim_loss = F.cross_entropy(y_shifted_similarities, y)
            # loss += shifted_sim_loss
            # sim_loss = (sim_loss + shifted_sim_loss.item()) / 2
            pitch_shift = y_anchors["pitch_shift"] - y_shifted_anchors["pitch_shift"]
            pitch_loss = self.mse(pitch_shift, batch["pitch_shift"].to(self.device))

            add_loss_list.append(pitch_loss)
            # loss += pitch_loss
        if self.use_frame_loss:
            frame_loss = self.frame_criterion(
                y_anchors["frame_embedding_z"], y_anchors["frame_embedding_z"]
            )
            frame_loss += self.frame_criterion(
                y_positives["frame_embedding_z"], y_positives["frame_embedding_z"]
            )
            frame_loss_cnt = 2
            if self.use_pitch_shift:
                frame_loss += self.frame_criterion(
                    y_shifted_anchors["frame_embedding_z"],
                    y_shifted_anchors["frame_embedding_z"],
                )
                # frame_loss += self.frame_criterion(
                #     y_anchors["frame_embedding_z"],
                #     y_shifted_anchors["frame_embedding_z"],
                # )
                frame_loss_cnt += 1
            frame_loss /= frame_loss_cnt
            add_loss_list.append(frame_loss)
            # loss += frame_loss
        if self.config.get("loss_type", "sum") == "sum":
            for add_loss in add_loss_list:
                loss += add_loss
        elif self.config.get("loss_type", "sum") == "product":
            tmp = 1
            for add_loss in add_loss_list:
                tmp += add_loss
            loss *= tmp

        # report
        self.total_valid_loss["valid/loss"] += loss.item()
        self.total_valid_loss["valid/smi_loss"] += sim_loss
        _, predicted = torch.max(y_similarities, 1)
        self.total_valid_loss["valid/acc"] += (
            (predicted == y).double().mean().cpu().numpy()
        )
        if self.use_pitch_shift:
            self.total_valid_loss["valid/pitch_loss"] += pitch_loss.item()
        if self.use_frame_loss:
            self.total_valid_loss["valid/frame_loss"] += frame_loss.item()

    def _valid_epoch(self):
        """Validate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start valid data's evaluation.")
        # change mode
        self.model.eval()
        self.similar_fc.eval()
        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["valid"], desc="[valid]"), 1
        ):
            # eval one step
            self._valid_step(batch)
        # log
        logging.info(
            f"(Steps: {self.steps}) Finished valid data's evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )
        for key, value in self.total_valid_loss.items():
            value /= eval_steps_per_epoch + 1
            self.total_valid_loss[key] = value
            if key == "valid/loss":
                if self.best_score > value:
                    self.best_score = value
                    logging.info(f"Best loss score is updated.:{value:.6f}")
                    self.save_checkpoint(
                        os.path.join(self.config["outdir"], "best_loss.pkl",),
                        save_model_only=False,
                    )
            logging.info(f"(Epoch: {self.epochs}) {key} = {value:.6f}.")
        # self._write_to_tensorboard(self.total_valid_loss)
        self.save_result_log(self.total_valid_loss, mode="valid")

        # reset
        self.total_valid_loss = defaultdict(float)
        # restore mode
        self.model.train()
        self.similar_fc.train()

    # def _write_to_tensorboard(self, loss):
    #     """Write to tensorboard."""
    #     for key, value in loss.items():
    #         self.writer.add_scalar(key, value, self.steps)

    def save_result_log(self, loss, mode="train"):
        json_path = os.path.join(self.config["outdir"], f"{mode}.json")
        if (os.path.isfile(json_path)) & (
            self.steps != self.config["log_interval_steps"]
        ):
            with open(json_path, "r") as f:
                base_dict = json.load(f, object_pairs_hook=OrderedDict)
        else:
            base_dict = OrderedDict()
        base_dict[self.steps] = loss
        with open(json_path, "w") as f:
            json.dump(base_dict, f, indent=4, ensure_ascii=False)

    def _check_valid_interval(self):
        if self.steps % self.config["valid_interval_steps"] == 0:
            self._valid_epoch()

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config["outdir"], f"checkpoint-{self.steps}steps.pkl",
                ),
                save_model_only=False,
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            self.total_train_loss["train/lr"] = self.optimizer.param_groups[0]["lr"]
            for key in self.total_train_loss.keys():
                if "lr" not in key:
                    self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            # self._write_to_tensorboard(self.total_train_loss)
            self.save_result_log(self.total_train_loss, mode="train")
            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True
