from src.model.ConvTasNet.encoder import ConvTasNetEncoder
from src.model.ConvTasNet.separator import ConvTasNetSeparator
from src.model.ConvTasNet.decoder import ConvTasNetDecoder
import torch.nn as nn
from torch import Tensor


class ConvTasNet(nn.Module):
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
    Output: dict, where s{i} -> [:, i, :] (i-th speaker audio)
    '''

    def __init__(self, 
            N: int = 512, 
            L: int = 16, 
            B: int = 128, 
            Sc: int = 128, 
            H: int = 512, 
            P: int = 3, 
            X: int = 8, 
            R: int = 3, 
            C: int = 2,
        ):
        super().__init__()
        self.encoder = ConvTasNetEncoder(N, L)
        self.separator = ConvTasNetSeparator(N, B, Sc, H, P, X, R, C)
        self.decoder = ConvTasNetDecoder(N, L)

    def forward(self, mix_audio: Tensor, **batch) -> dict:
        mix_enc = self.encoder(mix_audio) # [batch, N, T_new]
        masks = self.separator(mix_enc) # [batch, C, N, T_new]
        masked_audios = mix_enc.unsqueeze(1) * masks # [batch, C, N, T_new]
        separated_audios = self.decoder(masked_audios) # [batch, C, T]

        return {'s1_pred': separated_audios[:, 0, :], 's2_pred': separated_audios[:, 1, :]}
