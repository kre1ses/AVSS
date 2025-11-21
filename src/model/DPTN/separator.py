import torch
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

    Input: [batch, feature_dim, T_new]
    Output: [batch, feature_dim, K, num_chunks] (num_chunks = (T_new - K) // H + 1)
    """

    def __init__(self, K: int, H: int):
        super().__init__()
        self.K = K
        self.H = H

    def forward(self, x: Tensor) -> Tensor:
        B, N, T = x.shape
        chunks = x.unfold(
            dimension=2, size=self.K, step=self.H
        ).contiguous()  # [B, feature_dim, K, num_chunks]
        chunks = chunks.view(B, N, self.K, -1)
        chunks = chunks.permute(
            0, 1, 3, 2
        ).contiguous()  # [B, feature_dim, num_chunks, K]

        return chunks


class DPTNTransformer(nn.Module):
    """
    Improved transformer module in DPTN model
    Args:
        embed_dim (int): dimension of model
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional

    Input: [B*num_chunks, K, feature_dim] - intra OR [B*K, num_chunks, feature_dim] - inter
    Output: [B*num_chunks, K, feature_dim] - intra OR [B*K, num_chunks, feature_dim] - inter
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dropout: float,
        lstm_dim: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn_act = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_output_dim = lstm_dim * 2 if bidirectional else lstm_dim
        self.linear = nn.Linear(lstm_output_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        self.lstm.flatten_parameters()

        res = x
        x = self.attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + res)
        res = x
        x = self.ffn_act(self.lstm(x)[0])
        x = self.linear(x)
        x = self.norm2(x + res)

        return x


class DPTNBlock(nn.Module):
    """
    DPTN separation block

    Args:
        feature_dim (int): number of features in separator
        nhead (int): number of heads in multi-head attention
        dropout (float): dropout in transformer
        lstm_dim (int): dimension of LSTM layer
        bidirectional (bool): whether LSTM is bidirectional


    Input: [batch, feature_dim, K, num_chunks]
    Output: [batch, feature_dim, K, num_chunks]
    """

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        dropout: float,
        lstm_dim: int,
        bidirectional: bool,
    ):
        super().__init__()

        self.intra_transformer = DPTNTransformer(
            feature_dim, nhead, dropout, lstm_dim, bidirectional
        )
        self.inter_transformer = DPTNTransformer(
            feature_dim, nhead, dropout, lstm_dim, bidirectional=bidirectional
        )

    def forward(self, x: Tensor) -> Tensor:
        B, feature_dim, num_chunks, K = x.shape

        x = (
            x.permute(0, 2, 3, 1).contiguous().view(B * num_chunks, K, feature_dim)
        )  # K - embed_dim
        x = self.intra_transformer(x)

        x = (
            x.contiguous()
            .view(B, num_chunks, K, feature_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * K, num_chunks, feature_dim)
        )  # num_chunks - embed_dim
        x = self.inter_transformer(x)

        x = (
            x.contiguous()
            .view(B, K, num_chunks, feature_dim)
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

    Input: [batch * C, feature_dim, K, num_chunks]
    Output: [batch * C, feature_dim, T_new] (T_new = H * (num_chunks - 1) + K)
    """

    def __init__(self, K: int, H: int):
        super().__init__()
        self.K = K
        self.H = H

    def forward(self, x: Tensor) -> Tensor:
        BC, feature_dim, num_chunks, _ = x.shape
        T_new = self.H * (num_chunks - 1) + self.K

        x = (
            x.permute(0, 1, 3, 2)
            .contiguous()
            .view(BC, feature_dim * self.K, num_chunks)
        )

        output = F.fold(
            x, kernel_size=(self.K, 1), stride=(self.H, 1), output_size=(T_new, 1)
        )  # -> (batch_size, num_features, T_new, 1)

        output = output.squeeze(3)

        return output


class MaskCreator(nn.Module):
    """
    Final layers after OverlapAdd module

    Args:
        feature_dim (int): number of features in separator
        N (int): number of filters in autoencoder
        C (int): number of speakers

    Input: [batch * C, feature_dim, T_new]
    Output: [batch, C, N, T_new]
    """

    def __init__(self, feature_dim: int, N: int, C: int):
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

        # x = [self.act(self.tanh(x[:, i, :, :]) * self.sigmoid(x[:, i, :, :])) for i in range(C)]

        # masks = torch.stack([i.unsqueeze(1) for i in x], dim=1)

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
        feature_dim (int): number of features in separator
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
        feature_dim: int,
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
        self.feature_dim = feature_dim

        self.feat_conv = nn.Conv1d(
            in_channels=N, out_channels=feature_dim, kernel_size=1, bias=False
        )
        self.segmentation = Segmentation(K, H)
        self.global_norm = GlobalLayerNorm(N)
        # self.global_norm = GlobalLayerNorm(feature_dim)

        self.dptn_blocks = nn.ModuleList(
            [
                DPTNBlock(feature_dim, nhead, dropout, lstm_dim, bidirectional)
                for _ in range(R)
            ]
        )

        self.act = nn.PReLU()
        self.conv2d = nn.Conv2d(
            in_channels=N,
            out_channels=N * C,
            kernel_size=1,
            bias=False,
        )
        self.overlap_add = OverlappAdd(K, H)
        self.mask_creator = MaskCreator(feature_dim, N, C)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.global_norm(x)  # [B, N, T_new]
        # x = self.feat_conv(x)  # [B, feature_dim, T_new]

        x = self.segmentation(x)  # [B, feature_dim, K, num_chunks]
        x = self.global_norm(x)

        for dptn_block in self.dptn_blocks:
            x = dptn_block(x)

        x = self.act(x)  # [B, feature_dim, num_chunks, K]
        x = self.conv2d(x)  # [B, C*feature_dim, num_chunks, K]

        B, _, num_chunks, K = x.shape  # [B, C*feature_dim, num_chunks, K]
        # x = x.view(
        #     B * self.C, self.N, num_chunks, K
        # )  # [B*C, feature_dim, K, num_chunks]
        x = self.overlap_add(x)  # [B, C*feature_dim, T_new]
        x = x.view(B, self.C, self.N, -1)
        masks = self.mask_creator(x)  # [B, C, N, T_new]

        return masks
