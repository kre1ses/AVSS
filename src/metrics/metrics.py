import torch
from torch import Tensor
from src.metrics.base_metric import BaseMetric
import torch.nn as nn
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.pesq import short_time_objective_intelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SI_SNR_Metric(BaseMetric):
    '''
    Scale-Invariant Signal-to-Noise Ratio calculation
    preds, targets: [B, T]
    '''
    def __init__(self):
        super().__init__()
        self.metric = ScaleInvariantSignalNoiseRatio(reduction='none')

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.metric(preds, targets)
    

class SI_SDR_Metric(BaseMetric):
    '''
    Scale-Invariant Signal-to-Distortion Ratio calculation
    preds, targets: [B, T]
    '''
    def __init__(self):
        super().__init__()
        self.metric = ScaleInvariantSignalDistortionRatio(reduction='none')

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        return self.metric(preds, targets)
    

class SNRi_Metric(nn.Module):
    '''
    SI-SNR improvement in dB
    Input: mix, preds, targers: [B, T]
    Output: metric 
    '''
    def __init__(self):
        super().__init__()
        self.si_snr_pit = SI_SNR_Metric()
        self.si_snr = ScaleInvariantSignalNoiseRatio(reduction='none')

    def forward(self, mix: Tensor, s1_pred: Tensor, s2_pred: Tensor, 
                 s1_audio: Tensor, s2_audio: Tensor, **batch) -> Tensor:
        
        predicted_snr = self.si_snr_pit(s1_pred, s2_pred, s1_audio, s2_audio) 

        m1 = self.si_snr(mix, s1_audio)  
        m2 = self.si_snr(mix, s2_audio)  
        mix_snr = 0.5 * (m1 + m2)          
        mix_snr = mix_snr.mean() 

        improvement = predicted_snr - mix_snr
        return improvement
    

class PESQ_Metric(BaseMetric):
    '''
    Perceptual Evaluation of Speech Quality
    Returns: score in [1.0, 4.5]
    '''
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.mode = mode

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        pesq = perceptual_evaluation_speech_quality(preds, targets, self.fs, self.mode)
        return pesq


class STOI_Metric(BaseMetric):
    '''
    Short-Time Objective Intelligibility
    Returns: score in [0.0, 1.0]
    '''
    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__()
        self.fs = fs
        self.extended = extended

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        stoi = short_time_objective_intelligibility(preds, targets, self.fs, self.extended)
        return stoi
