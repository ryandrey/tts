import logging
from typing import Tuple, Dict, Optional, List, Union
from itertools import islice

import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    durations_prediction: Optional[torch.Tensor] = None
    melspec: Optional[torch.Tensor] = None
    melspec_prediciton: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        self.durations = self.durations.to(device)
        self.durations_prediction = self.durations_prediction.to(device)
        self.melspec = self.melspec.to(device)
        self.durations_prediction = self.melspec_prediciton.to(device)
        return self


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)