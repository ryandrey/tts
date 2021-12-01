import torch
import torch.nn as nn
from tts.collate_fn import Batch


class DurLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, batch):
        min_len = min(batch.durations.shape[-1], batch.durations_prediction.shape[-1])

        return self.loss(batch.durations[:, :min_len], batch.durations_prediction[:, :min_len])


class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, batch):
        min_len = min(batch.melspec.shape[-1], batch.melspec_prediction.shape[-2])
        return self.loss(
            batch.melspec[:, :, :min_len],
            batch.melspec_prediction.transpose(-1, -2)[:, :, :min_len]
        )



