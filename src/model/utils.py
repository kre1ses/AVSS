import torch.nn as nn
from torch import Tensor


class GlobalLayerNorm(nn.Module):
    '''
    Global Layer Normalization in ConvTasNet

    Args: 
        N (int): number of filters in autoencoder

    '''
    def __init__(self, N: int):
        super().__init__()
        self.normalizer = nn.GroupNorm(num_groups=1, num_channels=N)

    def forward(self, x: Tensor) -> Tensor:
        return self.normalizer(x)
    

