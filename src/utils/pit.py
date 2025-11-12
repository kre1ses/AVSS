import torch
from torch import Tensor

def PIT(s1: Tensor, s2: Tensor, t1: Tensor, t2: Tensor, fn: function, minimize: bool) -> Tensor:
    '''
    Permutation invariant training
    s1, s2:  predicted waveforms [B, T]
    t1, t2: target waveforms [B, T]
    fn: loss or metric function
    minimize: True - for loss, False - for metric
    '''
    fn_1 = 0.5 * (fn(s1, t1) + fn(s2, t2))
    fn_2 = 0.5 * (fn(s1, t2) + fn(s2, t1))

    if minimize:
        final = torch.min(fn_1, fn_2)
        return torch.mean(final)
    
    final = torch.max(fn_1, fn_2)
    return torch.mean(final)
    
    