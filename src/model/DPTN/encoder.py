import torch.nn as nn
from torch import Tensor

class DPTNEncoder(nn.Module):
    '''
    Convolutional encoder in DPTN model

    Args:
        N (int): number of filters in autoencoder
        L (int): length of the filters (in samples)

    Input: [batch, 1, T]
    Output: [batch, N, T_new]
    '''

    def __init__(self, N: int, L: int):
        super().__init__()
        self.encoder = nn.Conv1d(in_channels=1, out_channels=N, kernel_size=L, stride = L // 2, bias=False)
        # self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
        # return self.act(self.encoder(x))