from src.model.ConvTasNet.encoder import ConvTasNetEncoder
from src.model.ConvTasNet.separator import ConvTasNetSeparator
from src.model.ConvTasNet.decoder import ConvTasNetDecoder
import torch.nn as nn


class AVConvTasNet(nn.Module):
    '''
    ConvTasNet model with Video Support (https://arxiv.org/pdf/1809.07454)

    Args:
        N (int): number of filters in autoencoder
        L (int): length of the filters (in samples)
        B (int): number of channels in bottleneck and residual paths
        Sc (int): number of channels in skip-connection paths
        H (int): number of channels in conv blocks
        P (int): kernel size in conv block
        X (int): number of conv blocks in repeat
        R (int): number of repeats
        C (int): number of speakers

    Input: [batch, 1, T]
    Output: [batch, C, T] -> [:, i, :] - speaker i
    '''

    def __init__(self, N: int, L: int, B: int, Sc: int, H: int, P: int, X: int, R: int, C: int):
        super().__init__()
        self.encoder = ConvTasNetEncoder(N, L)
        self.separator = ConvTasNetSeparator(N, B, Sc, H, P, X, R, C)
        self.decoder = ConvTasNetDecoder()