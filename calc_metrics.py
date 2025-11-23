import os
from pathlib import Path

import hydra
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.metrics.metrics import *
from src.utils.io_utils import ROOT_PATH


def load_audio(path, target_sr=16000):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor


@hydra.main(
    version_base=None, config_path="src/configs", config_name="calculate_metrics"
)
def main(config):
    si_snr = SI_SNR_Metric(name="SI_SNR_Metric")
    snri = SI_SNRi_Metric(name="SI_SNRi_Metric")
    pesq = PESQ_Metric(name="PESQ_Metric")
    stoi = STOI_Metric(name="STOI_Metric")
    sdri = SDRi_Metric(name="SDRi_Metric")

    si_snri_metrics = []
    si_snr_metrics = []
    pesq_metrics = []
    stoi_metrics = []
    sdri_metrics = []

    current_path = Path().absolute().resolve()

    predictions_path_s1 = current_path / Path(config.predictions_path) / "s1"
    predictions_path_s2 = current_path / Path(config.predictions_path) / "s2"
    groud_truth_path_s1 = current_path / Path(config.groud_truth_path) / "s1"
    groud_truth_path_s2 = current_path / Path(config.groud_truth_path) / "s2"
    mix_path = current_path / Path(config.mix_path) / "mix"

    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")

    mix_files = sorted(os.listdir(mix_path))

    for mix_file in mix_files:
        base_name = Path(mix_file).stem

        first_id, second_id = base_name.split("_")

        s1_gt = None
        s2_gt = None

        for ext in AUDIO_EXTENSIONS:
            s1_gt_cand = groud_truth_path_s1 / f"{base_name}{ext}"
            s2_gt_cand = groud_truth_path_s2 / f"{base_name}{ext}"

            if s1_gt_cand.exists() and s1_gt is None:
                s1_gt = s1_gt_cand

            if s2_gt_cand.exists() and s2_gt is None:
                s2_gt = s2_gt_cand

        s1_pred = predictions_path_s1 / f"{base_name}.wav"
        s2_pred = predictions_path_s2 / f"{base_name}.wav"

        mix_audio_path = mix_path / mix_file

        mix_audio = load_audio(mix_audio_path)
        s1_gt_audio = load_audio(s1_gt)
        s2_gt_audio = load_audio(s2_gt)
        s1_pred_audio = load_audio(s1_pred)
        s2_pred_audio = load_audio(s2_pred)

        si_snri_metrics.append(
            snri(mix_audio, s1_pred_audio, s2_pred_audio, s1_gt_audio, s2_gt_audio)
        )
        si_snr_metrics.append(
            si_snr(s1_pred_audio, s2_pred_audio, s1_gt_audio, s2_gt_audio)
        )
        pesq_metrics.append(
            pesq(s1_pred_audio, s2_pred_audio, s1_gt_audio, s2_gt_audio)
        )
        stoi_metrics.append(
            stoi(s1_pred_audio, s2_pred_audio, s1_gt_audio, s2_gt_audio)
        )
        sdri_metrics.append(
            sdri(mix_audio, s1_pred_audio, s2_pred_audio, s1_gt_audio, s2_gt_audio)
        )

    print("SI-SNRi: ", torch.mean(torch.tensor(si_snri_metrics)).item())
    print("SI-SNR: ", torch.mean(torch.tensor(si_snr_metrics)).item())
    print("PESQ: ", torch.mean(torch.tensor(pesq_metrics)).item())
    print("STOI: ", torch.mean(torch.tensor(stoi_metrics)).item())
    print("SDRi: ", torch.mean(torch.tensor(sdri_metrics)).item())


if __name__ == "__main__":
    main()
