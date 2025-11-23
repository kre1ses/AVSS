import torch.nn as nn
from torch import Tensor

class DPTNDecoder(nn.Module):
    '''
    Convolutional decoder in DPTN model

    Args:
        N (int): number of filters in autoencoder
        L (int): length of the filters (in samples)

    Input: [batch, C, N, T_new]
    Output: [batch, C, T]
    '''

    def __init__(self, N: int, L: int):
        super().__init__()
        self.decoder = nn.ConvTranspose1d(in_channels=N, out_channels=1, kernel_size=L, stride = L // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, N, T_L = x.shape                   
        x = x.view(B * C, N, T_L)                  
        x = self.decoder(x)                        
        x = x.view(B, C, -1)           
        return x
    