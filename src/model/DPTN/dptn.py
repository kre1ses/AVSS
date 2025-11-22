from torch import Tensor, nn

from src.model.DPTN.decoder import DPTNDecoder
from src.model.DPTN.encoder import DPTNEncoder
from src.model.DPTN.separator import DPTNSeparator


class DPTN(nn.Module):
    """
    Dual-Path Transformer Network (DPTN) for speech separation (https://arxiv.org/pdf/2007.13975)

    Args:
        N (int): number of filters in autoencoder
        L (int): length of the filters (in samples)
        K (int): chunk length
        H (int): hop size
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional
        R (int): number of DPTN blocks
        C (int): number of speakers

    Input: [batch, T]
    Output: dict, where s{i} -> [:, i, :] (i-th speaker audio)
    """

    def __init__(
        self,
        N: int = 64,
        L: int = 8,  # если плохо учится, то ставь 2
        K: int = 151,
        H: int = 72,
        nhead: int = 4,
        dropout: float = 0.1,
        lstm_dim: int = 128,
        bidirectional: bool = True,
        R: int = 6,
        C: int = 2,
    ):
        super().__init__()
        self.encoder = DPTNEncoder(N, L)
        self.separator = DPTNSeparator(
            K, H, nhead, dropout, lstm_dim, bidirectional, R, N, C
        )
        self.decoder = DPTNDecoder(N, L)

    def forward(self, mix_audio: Tensor, **batch) -> Tensor:
        mix_audio = mix_audio.unsqueeze(1)  # [batch, 1, T]
        mix_enc = self.encoder(mix_audio)  # [batch, N, T_new]
        masked = self.separator(mix_enc)  # [batch, C, N, T_new]
        masked = mix_enc.unsqueeze(1) + masked  # [batch, C, N, T_new]
        separated_audios = self.decoder(masked)  # [batch, C, T]

        return {
            "s1_pred": separated_audios[:, 0, :],
            "s2_pred": separated_audios[:, 1, :],
        }
