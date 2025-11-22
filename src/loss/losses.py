import torch
from torch import Tensor
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio

from src.loss.base_loss import BaseLoss


class SI_SNR_Loss(BaseLoss):
    """
    Scale Invariant Sound to Noise Ratio Loss
    preds - predicted waveforms [B, T]
    targets - target waveforms [B, T]
    """

    def __init__(self):
        super().__init__()

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss = -scale_invariant_signal_noise_ratio(preds, targets)
        return loss
    

class SI_SNR_Loss_by_hand(BaseLoss):
    """
    Scale Invariant Sound to Noise Ratio Loss
    preds - predicted waveforms [B, T]
    targets - target waveforms [B, T]
    """

    def __init__(self):
        super().__init__()

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        eps = 1e-8
        preds_zm = preds - preds.mean(dim=-1, keepdim=True)
        targets_zm = targets - targets.mean(dim=-1, keepdim=True)

        dot = torch.sum(preds_zm * targets_zm, dim=-1, keepdim=True)
        target_energy = torch.linalg.norm(targets_zm, ord=2, dim=-1, keepdim=True) ** 2 + eps
        s_target = dot * targets_zm / target_energy

        e_noise = preds_zm - s_target

        si_snr = -20 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + eps) /
            (torch.sum(e_noise ** 2, dim=-1) + eps)
        )

        return si_snr


class L1_Loss(BaseLoss):
    """
    L1 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    """

    def __init__(self):
        super().__init__()

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        dims = tuple(range(1, preds.ndim))
        loss = torch.mean(torch.abs(preds - targets), dim=dims)
        return loss


class L2_Loss(BaseLoss):
    """
    L2 loss
    preds - predicted [B, T] for waveforms / [B, F, T] for spectrogram
    targets - target [B, T] for wafeforms / [B, F, T] for spectrogram
    """

    def __init__(self):
        super().__init__()

    def calc_loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        dims = tuple(range(1, preds.ndim))
        loss = torch.mean((preds - targets) ** 2, dim=dims)
        return loss
