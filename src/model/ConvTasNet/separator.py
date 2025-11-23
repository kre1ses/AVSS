import torch
import torch.nn as nn
from torch import Tensor

from src.model.utils import GlobalLayerNorm


class ConvolutionBlock(nn.Module):
    """
    Convolution Block in ConvTasNet

    Args:
        B (int): number of channels in bottleneck and residual paths
        Sc (int): number of channels in skip-connection paths
        H (int): number of channels in conv blocks
        P (int): kernel size in conv block
        D (int): dilation in D-conv of the ConvBlock

    Input: [batch, B, T_new]
    Output:
        1) [batch, Sc, T_new] -> skip conn
        2) [batch, B, T_new] -> output
    """

    def __init__(self, B: int, Sc: int, H: int, P: int, D: int):
        super().__init__()
        assert P % 2 != 0, "kernel_size must be odd"

        self.conv1 = nn.Conv1d(in_channels=B, out_channels=H, kernel_size=1)
        self.act1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(H)
        self.conv2 = nn.Conv1d(
            in_channels=H,
            out_channels=H,
            kernel_size=P,
            groups=H,
            dilation=D,
            padding=D * (P - 1) // 2,
        )
        self.act2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(H)

        self.conv_out = nn.Conv1d(in_channels=H, out_channels=B, kernel_size=1)
        self.conv_skip = nn.Conv1d(in_channels=H, out_channels=Sc, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.norm1(self.act1(self.conv1(x)))
        x_2 = self.norm2(self.act2(self.conv2(x_1)))
        x_out = self.conv_out(x_2)
        x_skip = self.conv_skip(x_2)
        return x + x_out, x_skip


class Repeat(nn.Module):
    """
    Repeat is a block which is made from ConvBlocks with different dilation factors

    Args:
        B (int): number of channels in bottleneck and residual paths
        Sc (int): number of channels in skip-connection paths
        H (int): number of channels in conv blocks
        P (int): kernel size in conv block
        D (int): dilation in D-conv of the ConvBlock
        X (int): number of ConvBlocks in 1 Repeat

    Input: [batch, B, T_new]
    Output:
        1) [batch, Sc, T_new] -> skip conn
        2) [batch, B, T_new] -> output
    """

    def __init__(self, B: int, Sc: int, H: int, P: int, X: int):
        super().__init__()
        self.Sc = Sc
        self.blocks = nn.ModuleList(
            [ConvolutionBlock(B, Sc, H, P, D=2**i) for i in range(X)]
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, _, T = x.shape
        skip_sum = torch.zeros(batch, self.Sc, T, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip

        return x, skip_sum


class ConvTasNetSeparator(nn.Module):
    """
    Separation block in ConvTasNet model

    Args:
        N (int): number of filters in autoencoder
        B (int): number of channels in bottleneck and residual paths
        Sc (int): number of channels in skip-connection paths
        H (int): number of channels in conv blocks
        P (int): kernel size in conv block
        X (int): number of conv blocks in repeat
        R (int): number of repeats
        C (int): number of speakers

    Input: [batch, N, T_new]
    Output: [batch, C, N, T_new] -> masks
    """

    def __init__(self, N: int, B: int, Sc: int, H: int, P: int, X: int, R: int, C: int):
        super().__init__()
        self.C = C
        self.N = N
        self.B = B
        self.Sc = Sc

        self.norm = GlobalLayerNorm(N)
        self.conv_encoder = nn.Conv1d(
            in_channels=self.N, out_channels=self.B, kernel_size=1
        )

        self.repeats = nn.ModuleList([Repeat(B, Sc, H, P, X) for _ in range(R)])

        self.act = nn.PReLU()
        self.mask_conv = nn.Conv1d(in_channels=Sc, out_channels=C * N, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        batch, _, T_new = x.shape

        x = self.conv_encoder(self.norm(x))  # [batch, N, T_new] -> [batch, B, T_new]

        skip_sum = torch.zeros(batch, self.Sc, T_new, device=x.device, dtype=x.dtype)
        for repeat in self.repeats:
            x, skip = repeat(x)
            skip_sum += skip

        masks = self.sigmoid(
            self.mask_conv(self.act(skip_sum))
        )  # [batch, Sc, T_new] -> [batch, C*N, T_new]
        batch, _, T_new = masks.shape

        masks = masks.view(batch, self.C, self.N, T_new)  # [batch, C, N, T_new]

        return masks
