import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.utils import GlobalLayerNorm


class Segmentation(nn.Module):
    """
    Segmentation module in DPTN model

    Args:
        K (int): chunk length
        H (int): hop size

    Input: [batch, N, T_new]
    Output: [batch, N, K, num_chunks] (num_chunks = (T_new - K) // H + 1)
    """

    def __init__(self, K: int, H: int):
        super().__init__()
        self.K = K
        self.H = H

    def forward(self, x: Tensor) -> Tensor:
        B, N, T = x.shape
        x = x.view(B, N, T, 1) # [B, N, T, 1]
        chunks = F.unfold(x, kernel_size=(self.K, 1), stride=(self.H, 1)) # [B, N, K, num_chunks]
        chunks = chunks.view(B, N, self.K, -1)
        chunks = chunks.permute(
            0, 1, 3, 2
        ).contiguous()  # [B, N, num_chunks, K]

        return chunks


class DPTNTransformer(nn.Module):
    """
    Improved transformer module in DPTN model
    Args:
        N (int): number of filters in autoencoder
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional

    Input: [batch * num_chunks, K, N] - intra OR [batch * K, num_chunks, N] - inter
    Output: [batch * num_chunks, K, N] - intra OR [batch * K, num_chunks, N] - inter
    """

    def __init__(
        self,
        N: int,
        nhead: int,
        dropout: float,
        lstm_dim: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=N, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(N)
        self.norm2 = nn.LayerNorm(N)
        self.ffn_act = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=N,
            hidden_size=lstm_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=1,
        )
        lstm_output_dim = lstm_dim * 2 if bidirectional else lstm_dim
        self.linear = nn.Linear(lstm_output_dim, N)

    def forward(self, x: Tensor) -> Tensor:
        self.lstm.flatten_parameters()

        res = x
        x = self.attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + res)
        res = x
        x = self.lstm(x)[0]
        x = self.linear(self.ffn_act(x))
        x = self.norm2(x + res)

        return x


class DPTNBlock(nn.Module):
    """
    DPTN separation block

    Args:
        N (int): number of filteres in autoencoder
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional


    Input: [batch, N, num_chunks, K]
    Output: [batch, N, num_chunks, K]
    """

    def __init__(
        self,
        N: int,
        nhead: int,
        dropout: float,
        lstm_dim: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.intra_transformer = DPTNTransformer(
            N, nhead, dropout, lstm_dim, bidirectional=True
        )
        self.inter_transformer = DPTNTransformer(
            N, nhead, dropout, lstm_dim, bidirectional=bidirectional
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, num_chunks, K = x.shape

        x = (
            x.permute(0, 2, 3, 1).contiguous().view(B * num_chunks, K, N)
        )  # K - embed_dim
        x = self.intra_transformer(x)

        x = (
            x.contiguous()
            .view(B, num_chunks, K, N)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * K, num_chunks, N)
        )  # num_chunks - embed_dim
        x = self.inter_transformer(x)

        x = (
            x.contiguous()
            .view(B, K, num_chunks, N)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        return x


class OverlappAdd(nn.Module):
    """
    Overlap-add module in DPTN model

    Args:
        K (int): chunk length
        H (int): hop size

    Input: [batch * C, N, num_chunks, K]
    Output: [batch, N, T_new] (T_new = H * (num_chunks - 1) + K)
    """

    def __init__(self, K: int, H: int):
        super().__init__()
        self.K = K
        self.H = H

    def forward(self, x: Tensor) -> Tensor:
        BC, N, num_chunks, _ = x.shape
        T_new = self.H * (num_chunks - 1) + self.K

        x = (
            x.permute(0, 1, 3, 2)
            .contiguous()
            .view(BC, N * self.K, num_chunks)
        )

        output = F.fold(
            x, kernel_size=(self.K, 1), stride=(self.H, 1), output_size=(T_new, 1)
        )  # [B*C, N, T_new, 1]

        output = output.squeeze(3) # [B*C, N, T_new]

        return output


class MaskCreator(nn.Module):
    """
    Final layers after OverlapAdd module

    Args:
        N (int): number of filters in autoencoder
        C (int): number of speakers

    Input: [batch * C, N, T_new]
    Output: [batch, C, N, T_new]
    """

    def __init__(self, N: int, C: int):
        super().__init__()
        self.C = C
        self.N = N

        self.tanh = nn.Sequential(
            nn.Conv1d(in_channels=N, out_channels=N, kernel_size=1, bias=False),
            nn.Tanh(),
        )

        self.sigmoid = nn.Sequential(
            nn.Conv1d(in_channels=N, out_channels=N, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        B, C, N, T_new = x.shape

        x = x.view(B * C, N, T_new)  # [B*C, N, T_new]

        x_tanh = self.tanh(x)  # [B*C, N, T_new]
        x_sigmoid = self.sigmoid(x)  # [B*C, N, T_new]

        x = self.act(x_tanh * x_sigmoid)  # [B*C, N, T_new]

        masks = x.view(B, C, N, T_new)

        return masks


class DPTNSeparator(nn.Module):
    """
    Separation module in DPTN model

    Args:
        K (int): chunk length
        H (int): hop size
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional
        R (int): number of DPTN blocks
        N (int): number of filters in autoencoder
        C (int): number of speakers

    Input: [batch, N, T_new]
    Output: [batch, C, N, T_new]
    """

    def __init__(
        self,
        K: int,
        H: int,
        nhead: int,
        dropout: float,
        lstm_dim: int,
        bidirectional: bool,
        R: int,
        N: int,
        C: int,
    ):
        super().__init__()
        self.C = C
        self.N = N

        self.conv = nn.Conv1d(
            in_channels=N, out_channels=N, kernel_size=1
        )
        self.segmentation = Segmentation(K, H)
        # self.global_norm = GlobalLayerNorm(N)

        self.dptn_blocks = nn.Sequential()

        for _ in range(R):
            dptn_block = DPTNBlock(
                N=N,
                nhead=nhead,
                dropout=dropout,
                lstm_dim=lstm_dim,
                bidirectional=bidirectional,
            )
            self.dptn_blocks.append(dptn_block)

        self.act = nn.PReLU()
        self.conv2d = nn.Conv2d(
            in_channels=N,
            out_channels=N * C,
            kernel_size=1
        )
        self.overlap_add = OverlappAdd(K, H)
        # self.mask_creator = MaskCreator(N, C)

    def forward(self, x: Tensor) -> Tensor:

        segm_x = self.segmentation(x)  
        # x = self.global_norm(x)

        trans_x = self.dptn_blocks(segm_x)  # [batch, N, num_chunks, K]
        trans_x = self.conv2d(self.act(trans_x))  # [batch, C * N, num_chunks, K]

        B, _, _, _ = trans_x.shape  # [B, C * N, num_chunks, K]
        x = self.overlap_add(trans_x)  # [B, C*N, T_new]
        x = x.view(B * self.C, self.N, -1) # [B*C, N, T_new]
        x = self.conv(x)  # [B*C, N, T_new]
        masks = x.view(B, self.C, self.N, -1)  # [B, C, N, T_new]
        # masks = self.mask_creator(x)  # [B, C, N, T_new]

        return masks
