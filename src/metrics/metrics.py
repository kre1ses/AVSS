import torch
from torch import Tensor
from src.metrics.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class SI_SNR_Metric(BaseMetric):
    '''
    Scale-Invariant Signal-to-Noise Ratio calculation
    preds, targets: [B, T]
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, preds: Tensor, targets: Tensor, **batch) -> Tensor:

        preds = preds - preds.mean(dim=1)
        targets = targets - targets.mean(dim=1)

        dot = torch.sum(preds * targets, dim=1)
        target_energy = torch.sum(targets ** 2, dim=1)
        s_target = dot * targets / (target_energy + 1e-8) 
        e_noise = preds - s_target
        si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / torch.sum(e_noise ** 2, dim=1))

        return si_snr.mean()


class SDRi_Metric(BaseMetric):
    '''
    SI-SDR improvement in dB
    mix, preds, targers: [B, T]
    '''
    def __init__(self):
        super().__init__()
        self.metric = SI_SNR_Metric()

    def __call__(self, mix: Tensor, preds: Tensor, targets: Tensor, **batch) -> Tensor:
        improvement = self.metric(preds, targets) - self.metric(mix, targets)
        return improvement.mean()
    


class PESQ_Metric(BaseMetric):
    '''
    Perceptual Evaluation of Speech Quality
    Returns: score in [1.0, 4.5]
    '''
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__()
        self.pesq_metric = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, preds: Tensor, targets: Tensor, **batch) -> Tensor:
        pesq = self.pesq_metric(preds, targets)
        return pesq


class STOI_Metric(BaseMetric):
    '''
    Short-Time Objective Intelligibility
    Returns: score in [0.0, 1.0]
    '''
    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__()
        self.stoi_metric = ShortTimeObjectiveIntelligibility(fs, extended)

    def __call__(self, preds: Tensor, targets: Tensor, **batch) -> Tensor:
        stoi = self.stoi_metric(preds, targets)
        return stoi
