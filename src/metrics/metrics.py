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


class SDRi_Metric(nn.Module):
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
        sdr_11 = signal_distortion_ratio(s1_pred, s1_audio)  # [B]
        sdr_22 = signal_distortion_ratio(s2_pred, s2_audio)  # [B]
        sdr_12 = signal_distortion_ratio(s1_pred, s2_audio)  # [B]
        sdr_21 = signal_distortion_ratio(s2_pred, s1_audio)  # [B]

        mix_sdr_1 = signal_distortion_ratio(mix_audio, s1_audio)  # [B]
        mix_sdr_2 = signal_distortion_ratio(mix_audio, s2_audio)  # [B]

        perm1_mean = 0.5 * (sdr_11 + sdr_22)
        perm2_mean = 0.5 * (sdr_12 + sdr_21)

        use_perm1 = perm1_mean >= perm2_mean

        impr1_s1 = sdr_11 - mix_sdr_1  # [B]
        impr1_s2 = sdr_22 - mix_sdr_2  # [B]

        impr2_s1 = sdr_21 - mix_sdr_1  # [B]
        impr2_s2 = sdr_12 - mix_sdr_2  # [B]

        final_impr_s1 = torch.where(use_perm1, impr1_s1, impr2_s1)  # [B]
        final_impr_s2 = torch.where(use_perm1, impr1_s2, impr2_s2)  # [B]

        improvement = torch.cat(
            [final_impr_s1.unsqueeze(0), final_impr_s2.unsqueeze(0)], dim=0
        ).mean()
        return improvement


class SI_SNRi_Metric(nn.Module):
    """
    SI-SNR improvement in dB
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
        si_snr_s11 = scale_invariant_signal_noise_ratio(s1_pred, s1_audio)  # [B]
        si_snr_s12 = scale_invariant_signal_noise_ratio(s1_pred, s2_audio)  # [B]
        si_snr_s21 = scale_invariant_signal_noise_ratio(s2_pred, s1_audio)  # [B]
        si_snr_s22 = scale_invariant_signal_noise_ratio(s2_pred, s2_audio)  # [B]

        mix_snr_1 = scale_invariant_signal_noise_ratio(mix_audio, s1_audio)  # [B]
        mix_snr_2 = scale_invariant_signal_noise_ratio(mix_audio, s2_audio)  # [B]

        perm1_mean = 0.5 * (si_snr_s11 + si_snr_s22)
        perm2_mean = 0.5 * (si_snr_s12 + si_snr_s21)

        use_perm1 = perm1_mean >= perm2_mean

        impr1_s1 = si_snr_s11 - mix_snr_1  # [B]
        impr1_s2 = si_snr_s22 - mix_snr_2  # [B]
        impr2_s1 = si_snr_s21 - mix_snr_1  # [B]
        impr2_s2 = si_snr_s12 - mix_snr_2  # [B]

        final_impr_s1 = torch.where(use_perm1, impr1_s1, impr2_s1)  # [B]
        final_impr_s2 = torch.where(use_perm1, impr1_s2, impr2_s2)  # [B]

        improvement = torch.cat(
            [final_impr_s1.unsqueeze(0), final_impr_s2.unsqueeze(0)], dim=0
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
