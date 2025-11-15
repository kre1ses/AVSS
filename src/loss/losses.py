import torch
from torch import Tensor
from src.loss.base_loss import BaseLoss
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

        
class SI_SNR_Loss(BaseLoss):
    '''
    Scale Invariant Sound to Noise Ratio Loss
    preds - predicted waveforms [B, T]
    targets - target waveforms [B, T]
    '''
    def __init__(self):
        super().__init__()
        self.metric = ScaleInvariantSignalNoiseRatio(reduction='none')

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss = -self.metric(preds, targets)
        return loss


class L1_Loss(BaseLoss):
    '''
    L1 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    '''
    def __init__(self):
        super().__init__()  

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        dims = tuple(range(1, preds.ndim))
        loss = torch.mean(torch.abs(preds - targets), dim=dims)
        return loss


class L2_Loss(BaseLoss):
    '''
    L2 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    '''
    def __init__(self):
        super().__init__()  

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        dims = tuple(range(1, preds.ndim))
        loss = torch.mean((preds - targets) ** 2, dim=dims)
        return loss

