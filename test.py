import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from tts.model import FastSpeech
from tts.model import Vocoder

if __name__ == "__main__":
    texts = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
        "Give me some points please"
    ]

    device = torch.device('cuda:0')
    model = FastSpeech(
        ntoken=51,
        d_model=384,
        d_k=384,
        num_attn_layers=2,
        conv_d=1536,
        kernel_size=3,
        nlayers=6,
        num_mels=80,
        dropout=0.1
    )
    checkpoint = torch.load("tts_model.pth", device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    vocoder = Vocoder().to(device)
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    for i, text in enumerate(texts):
        tokens, token_lengths = tokenizer(text)
        output = model(tokens.to(device), None)
        pred_wav = vocoder.inference(output.transpose(-1, -2)).cpu()
        torchaudio.save(f"{i + 1}.wav", pred_wav, sample_rate=22050)

