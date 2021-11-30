import random
import torchaudio
import torch


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, limit=None, *args, **kwargs):
        super().__init__(root=root, download=True)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self._index = list(range(super().__len__()))
        self.limit = limit
        if limit is not None:
            random.seed(42)
            random.shuffle(self._index)
            self._index = self._index[:limit]

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(self._index[index])
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def __len__(self):
        if self.limit is None:
            return super().__len__()
        else:
            return self.limit

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
