import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tts.collate_fn import Batch
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    # pytorch tutorial
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Q_layer = nn.Linear(d_model, d_k)
        self.K_layer = nn.Linear(d_model, d_k)
        self.V_layer = nn.Linear(d_model, d_k)
        self.d_k = d_k

    def forward(self, x):
        Q = self.Q_layer(x)
        K = self.K_layer(x)
        V = self.V_layer(x)
        QK = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / self.d_k ** 0.5, dim=-1)
        return torch.matmul(QK, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, num_attn_layers):
        super().__init__()
        self.head = nn.ModuleList()
        for i in range(num_attn_layers):
            self.head.append(Attention(d_model, d_k))
        self.tail = nn.Linear(num_attn_layers * d_k, d_model)

    def forward(self, x):
        tmp = []
        for module in self.head:
            tmp.append(module(x))
        out = torch.cat(tmp, dim=-1)
        out = self.tail(out)
        return out


class Conv1D_FFT(nn.Module):
    def __init__(self, d_model, d_hid, kernel_size):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(d_model, d_hid, kernel_size, padding="same"))
        layers.append(nn.ReLU())
        layers.append(nn.Conv1d(d_hid, d_model, kernel_size, padding="same"))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x.transpose(1, 2))
        return x.transpose(1, 2)


class FFTBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_k,
                 num_attn_layers,
                 conv_d,
                 kernel_size,
                 dropout=0.2,
                 ):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, d_k, num_attn_layers)
        self.mha_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv = Conv1D_FFT(d_model, conv_d, kernel_size)

    def forward(self, x):
        out = self.multi_head_attention(x)
        x = self.mha_norm(x + out)
        x = self.dropout(x)

        out = self.conv(x)
        x = self.conv_norm(x + out)
        x = self.dropout(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d_model, conv_d, kernel_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            Conv1D_FFT(d_model, conv_d, kernel_size),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            Conv1D_FFT(d_model, conv_d, kernel_size),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        return self.net(x)


def create_alignment(base_mat, dur_preds):
    N, L = dur_preds.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(dur_preds[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + dur_preds[i][j]
    return base_mat


class Length_Regulator(nn.Module):
    def __init__(self, d_model, conv_d, kernel_size, dropout=0.1, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.dur_pred = DurationPredictor(d_model, conv_d, kernel_size, dropout)

    def LR(self, x, dur_preds, mel_max_len):
        expand_max_len = int(torch.max(
            torch.sum(dur_preds, -1), -1)[0])
        alignment = torch.zeros(dur_preds.size(0),
                                expand_max_len,
                                dur_preds.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     dur_preds.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_len:
            output = F.pad(
                output, (0, 0, 0, mel_max_len - output.size(1), 0, 0))
        return output

    def forward(self, x, target=None, mel_max_len=None):
        dur_preds = self.dur_pred(x).squeeze(-1)

        if target is not None:
            output = self.LR(x, target, mel_max_len)
            return output, dur_preds
        else:
            dur_preds = ((dur_preds + 0.5) * self.alpha).int()
            output = self.LR(x, dur_preds, mel_max_len)
            mel_pos = torch.stack(
                [torch.Tensor([i + 1 for i in range(output.size(1))])]).long().to(x.device)

            return output, mel_pos


class FastSpeech(nn.Module):
    def __init__(self, ntoken,
                 d_model,
                 d_k,
                 num_attn_layers,
                 conv_d,
                 kernel_size,
                 nlayers,
                 num_mels,
                 dropout=0.3,
                 phoneme_max=2000,
                 alpha=1.0,
                 mel_max=2000
                 ):
        super().__init__()
        self.mel_max = mel_max

        self.embedding = nn.Embedding(ntoken, d_model)
        self.phoneme_pos_enc = PositionalEncoding(d_model, dropout, phoneme_max)

        first_FFTblocks = []
        for i in range(nlayers):
            first_FFTblocks.append(FFTBlock(d_model, d_k, num_attn_layers, conv_d, kernel_size, dropout))
        self.encoder = nn.Sequential(*first_FFTblocks)

        self.length_regulator = Length_Regulator(d_model, d_k, kernel_size, dropout, alpha)

        self.mel_pos_enc = PositionalEncoding(d_model, dropout, mel_max)

        second_FFTblocks = []
        for i in range(nlayers):
            second_FFTblocks.append(FFTBlock(d_model, d_k, num_attn_layers, conv_d, kernel_size, dropout))
        self.decoder = nn.Sequential(*second_FFTblocks)

        self.head = nn.Linear(d_model, num_mels)

    def forward(self, x, target):
        output = self.embedding(x)
        output = self.phoneme_pos_enc(output)
        output = self.encoder(output)

        lr_output, dur_preds = self.length_regulator(output,
                                                     target=target,
                                                     mel_max_len=self.mel_max)
        output = self.mel_pos_enc(lr_output)
        output = self.decoder(output)
        output = self.head(output)

        if self.training:
            return output, dur_preds
        return output



