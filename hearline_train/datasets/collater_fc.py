"""Collater function modules."""
import logging
import random
import librosa
import numpy as np
import torch


class FeatTrainCollater(object):
    """Customized collater for Pytorch DataLoader for feat form data in training."""

    def __init__(self, max_frames=96):
        """Initialize customized collater for PyTorch DataLoader.
        Args:
            max_frames (int): The max size of melspectrogram's frame.
        """
        self.max_frames = max_frames

    def __call__(self, batch):
        """Convert into batch tensors."""
        logmels = [b["feat"] for b in batch]
        anchor_batch = []
        positive_batch = []

        # select start point
        for logmel in logmels:
            l_spec = len(logmel)
            # for anchor_batch
            beginning = random.randrange(0, l_spec - self.max_frames)
            ending = beginning + self.max_frames
            if ending > l_spec:
                ending = l_spec
            beginning = ending - self.max_frames
            anchor_batch.append(logmel[beginning:ending].astype(np.float32))
            # for positive_batch
            beginning = random.randrange(0, l_spec - self.max_frames)
            ending = beginning + self.max_frames
            if ending > l_spec:
                ending = l_spec
            beginning = ending - self.max_frames
            positive_batch.append(logmel[beginning:ending].astype(np.float32))

        # convert to tensor, assume that each item in batch has the same length
        batch = {}
        # (B, max_frames, mel)
        batch["anchors"] = torch.tensor(anchor_batch, dtype=torch.float)
        batch["positives"] = torch.tensor(positive_batch, dtype=torch.float)
        return batch


class WaveTrainCollater(object):
    """Customized collater for Pytorch DataLoader for wave form data in training."""

    def __init__(
        self,
        sample_rate=16000,
        sec=0.960,
        hop_size=160,
        noise=0.001,
        pitch_shift=0.0,
        embed_time_frame=12,
    ):
        """Initialize customized collater for PyTorch DataLoader.
        Args:
            sample_rate (int): sample_rate.
        """
        self.sample_rate = sample_rate
        self.max_frames = int(sample_rate * sec) - hop_size
        self.embed_time_frame = embed_time_frame
        self.noise = noise
        self.pitch_shift = pitch_shift

    def __call__(self, batch):
        """Convert into batch tensors."""
        anchor_batch = []
        positive_batch = []
        if self.pitch_shift != 0:
            shifted_anchor_batch = []
            pitch_shift = []
            pitch_gen = np.random.default_rng()
        # select start point
        for wave in batch:
            l_wave = len(wave)
            # logging.info(f"l_wave:{l_wave}")
            # for anchor_batch
            beginning = random.randrange(0, l_wave - self.max_frames)
            ending = beginning + self.max_frames
            if ending > l_wave:
                ending = l_wave
            beginning = ending - self.max_frames
            anchor = wave[beginning:ending].astype(np.float32)
            anchor_batch.append(torch.tensor(anchor, dtype=torch.float))
            if self.pitch_shift != 0:
                pitch_factor = pitch_gen.uniform(
                    -self.pitch_shift, self.pitch_shift, 1
                )[0]
                shifted_anchor = librosa.effects.pitch_shift(
                    anchor, sr=self.sample_rate, n_steps=np.log2(1 + pitch_factor)
                )
                pitch_shift.append(np.ones((1, self.embed_time_frame)) * pitch_factor)
                shifted_anchor_batch.append(shifted_anchor)
            # for positive_batch
            beginning = random.randrange(0, len(wave) - self.max_frames)
            ending = beginning + self.max_frames
            if ending > l_wave:
                ending = l_wave
            beginning = ending - self.max_frames
            pos = (
                wave[beginning:ending].astype(np.float32)
                + np.random.randn(self.max_frames) * self.noise
            )
            positive_batch.append(pos)

        # convert to tensor, assume that each item in batch has the same length
        batch = {}
        # (B, max_frames)
        batch["anchors"] = torch.stack(anchor_batch)
        # (B, max_frames)
        batch["positives"] = torch.tensor(np.array(positive_batch), dtype=torch.float)
        if self.pitch_shift != 0:
            # (B, embed_time_frame)
            batch["pitch_shift"] = torch.tensor(
                np.concatenate(pitch_shift, axis=0), dtype=torch.float
            )
            # logging.info(batch["pitch_shift"].shape, batch["pitch_shift"])
            # (B, max_frames)
            batch["shifted_anchors"] = torch.tensor(
                np.array(shifted_anchor_batch), dtype=torch.float
            )
        return batch


class EpochWaveCollater(object):
    """Customized collater for Pytorch DataLoader for wave form data in training."""

    def __init__(self, use_pitch_shift, embed_time_frame=12):
        self.use_pitch_shift = use_pitch_shift
        self.embed_time_frame = embed_time_frame

    def __call__(self, batch):
        """Convert into batch tensors."""
        return_batch = {"anchors": [], "positives": []}
        if self.use_pitch_shift:
            return_batch["shifted_anchors"] = []
            return_batch["pitch_shift"] = []

        for item in batch:
            return_batch["anchors"].append(item["anchor"])
            return_batch["positives"].append(item["positive"])
            if self.use_pitch_shift:
                return_batch["shifted_anchors"].append(item["shifted_anchor"])
                return_batch["pitch_shift"].append(
                    item["pitch_shift"][:, : self.embed_time_frame]
                )
        for key, value in return_batch.items():
            if key == "pitch_shift":
                value = np.concatenate(value, axis=0)
            return_batch[key] = torch.tensor(value, dtype=torch.float)

        return return_batch
