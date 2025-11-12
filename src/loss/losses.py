import torch
from torch import Tensor
from src.metrics.si_snr import SI_SNR_Metric
        

def SI_SNR_Loss(preds: Tensor, targets: Tensor, **batch) -> Tensor:
    '''
    Scale Invariant Sound to Noise Ratio Loss
    preds - predicted waveforms [B, T]
    targets - target waveforms [B, T]
    '''
    metric = SI_SNR_Metric()
    si_snr_loss = -metric(preds, targets)
    loss = si_snr_loss.mean()

    return {"loss": loss}


def L1_Loss(preds: Tensor, targets: Tensor, **batch) -> Tensor:
    '''
    L1 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    '''
    l1_loss = torch.abs(preds - targets).mean()
    return {"loss": l1_loss}


def L2_Loss(preds: Tensor, targets: Tensor, **batch) -> Tensor:
    '''
    L2 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    '''
    l2_loss = torch.mean((preds-targets)**2)
    return {"loss": l2_loss}

