"""EfficientNet."""
import logging
import torch
import torch.nn as nn
import torchaudio.transforms as T
from efficientnet_pytorch import EfficientNet


class EfficientNet_b0(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        n_embedding=512,
        use_pitch_shift=False,
        use_frame_embedding=False,
        parallel=False,
    ):

        super(self.__class__, self).__init__()
        self.spectrogram_extracter = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=2.0,
            n_mels=n_mels,
            normalized=True,
            f_min=60,
            f_max=7800,
        )
        self.efficientnet = EfficientNet.from_name(
            model_name="efficientnet-b0", in_channels=1
        )
        self.efficientnet._fc = nn.Sequential()
        self.efficientnet._bn1 = nn.Sequential()
        self.efficientnet._avg_pooling = nn.Sequential()
        self.efficientnet._dropout = nn.Sequential()
        self.fc1 = nn.Linear(1280, n_embedding, bias=True)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=n_embedding)
        self.fc2 = torch.nn.Linear(n_embedding, n_embedding, bias=False)
        self.sample_rate = sample_rate
        self.scene_embedding_size = n_embedding
        self.timestamp_embedding_size = n_embedding
        self.n_timestamp = None
        self.use_frame_embedding = use_frame_embedding
        if use_frame_embedding:
            self.frame_fc2 = torch.nn.Linear(n_embedding, n_embedding, bias=True)
            self.layer_norm1 = torch.nn.LayerNorm(normalized_shape=n_embedding)
        self.use_pitch_shift = use_pitch_shift
        self.parallel = parallel
        if use_pitch_shift:
            self.pitch_shift = nn.Linear(n_embedding, 1, bias=True)
            if parallel:
                self.layer_norm3 = torch.nn.LayerNorm(normalized_shape=n_embedding)

    def forward(self, X):
        """X: (batch_size, T', mels)"""
        # logging.info(f"X:{X.shape}")
        if len(X.shape) == 2:
            # X: (batch_size, wave_length)->(batch_size, T', mels)
            X = self.spectrogram_extracter(X).transpose(1, 2)
        x = X.unsqueeze(1)  # (B, 1, T', mels)
        x = self.efficientnet.extract_features(x)
        # logging.info(f"x:{x.shape}")
        x = x.max(dim=3)[0]
        # logging.info(f"x:{x.shape}")
        embedding_h = self.fc1(x.transpose(1, 2))
        self.n_timestamp = embedding_h.shape[1]
        # logging.info(f"embedding_h: {embedding_h.shape}")
        embedding_z = self.fc2(torch.tanh(self.layer_norm(embedding_h.max(dim=1)[0])))
        output_dict = {
            # (B, T', timestamp_embedding_size)
            "framewise_embedding": embedding_h,
            # (B, scene_embedding_size)
            "clipwise_embedding": embedding_h.max(dim=1)[0],
            "embedding_z": embedding_z,  # (B, n_embedding)
        }
        if self.use_frame_embedding or self.use_pitch_shift:
            # (B, T', n_embedding)
            frame_embedding_z = self.frame_fc2(
                torch.tanh(self.layer_norm1(embedding_h))
            )
            output_dict["frame_embedding_z"] = frame_embedding_z
        if self.use_pitch_shift:
            # (B, T)
            if self.parallel:
                output_dict["pitch_shift"] = self.pitch_shift(
                    torch.tanh(self.layer_norm3(embedding_h))
                ).squeeze()
            else:
                output_dict["pitch_shift"] = self.pitch_shift(
                    frame_embedding_z
                ).squeeze()

        return output_dict
