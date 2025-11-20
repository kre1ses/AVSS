import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.sdr import signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from src.metrics.base_metric import BaseMetric


class SI_SNR_Metric(BaseMetric):
    """
    Scale-Invariant Signal-to-Noise Ratio calculation
    preds, targets: [B, T]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        return scale_invariant_signal_noise_ratio(preds, targets)


class SDRi(nn.Module):
    """
    SDR improvement in dB
    Input: mix, preds, targers: [B, T]
    Output: metric
    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def forward(
        self,
        mix_audio: Tensor,
        s1_pred: Tensor,
        s2_pred: Tensor,
        s1_audio: Tensor,
        s2_audio: Tensor,
        **batch
    ) -> Tensor:
        predicted_sdr_s1 = signal_distortion_ratio(s1_pred, s1_audio)  # [B]
        predicted_sdr_s2 = signal_distortion_ratio(s2_pred, s2_audio)  # [B]
        mix_s1_sdr = signal_distortion_ratio(mix_audio, s1_audio)  # [B]
        mix_s2_sdr = signal_distortion_ratio(mix_audio, s2_audio)  # [B]

        improvement_s1 = predicted_sdr_s1 - mix_s1_sdr  # [B]
        improvement_s2 = predicted_sdr_s2 - mix_s2_sdr  # [B]
        print(improvement_s1.shape, improvement_s2.shape)
        improvement = torch.concat([improvement_s1, improvement_s2]).mean()
        return improvement


class SNRi_Metric(nn.Module):
    """
    SI-SNR improvement in dB
    Input: mix, preds, targers: [B, T]
    Output: metric
    """

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr_pit = SI_SNR_Metric()
        self.name = name

    def forward(
        self,
        mix_audio: Tensor,
        s1_pred: Tensor,
        s2_pred: Tensor,
        s1_audio: Tensor,
        s2_audio: Tensor,
        **batch
    ) -> Tensor:
        predicted_snr_s1 = scale_invariant_signal_noise_ratio(s1_pred, s1_audio)  # [B]
        predicted_snr_s2 = scale_invariant_signal_noise_ratio(s2_pred, s2_audio)  # [B]
        mix_s1_snr = scale_invariant_signal_noise_ratio(mix_audio, s1_audio)  # [B]
        mix_s2_snr = scale_invariant_signal_noise_ratio(mix_audio, s2_audio)  # [B]
        improvement_s1 = predicted_snr_s1 - mix_s1_snr  # [B]
        improvement_s2 = predicted_snr_s2 - mix_s2_snr  # [B]
        improvement = torch.concat(
            [improvement_s1.unsqueeze(0), improvement_s2.unsqueeze(0)], dim=0
        ).mean()
        return improvement


class PESQ_Metric(BaseMetric):
    """
    Perceptual Evaluation of Speech Quality
    Returns: score in [1.0, 4.5]
    """

    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.mode = mode

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        pesq = perceptual_evaluation_speech_quality(preds, targets, self.fs, self.mode)
        return pesq


class STOI_Metric(BaseMetric):
    """
    Short-Time Objective Intelligibility
    Returns: score in [0.0, 1.0]
    """

    def __init__(self, fs=16000, extended=False, *args, **kwargs):
        super().__init__()
        self.fs = fs
        self.extended = extended

    def calc_metric(self, preds: Tensor, targets: Tensor) -> Tensor:
        stoi = short_time_objective_intelligibility(
            preds, targets, self.fs, self.extended
        )
        return stoi
