from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    """
    Custom dataset for ASR inference or evaluation.

    Expected structure:
    dataset_dir/
      ├── audio/
      │     ├── file_001.wav
      │     ├── file_002.flac
      │     └── ...
      └── transcriptions/   (optional)
            ├── file_001.txt
            ├── file_002.txt
            └── ...

    """

    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")

    def __init__(self, dataset_dir: str, *args, **kwargs):
        dataset_dir = Path(dataset_dir)
        mix_audio_dir = dataset_dir / "audio" / "mix"
        s1_audio_dir = dataset_dir / "audio" / "s1"
        s2_audio_dir = dataset_dir / "audio" / "s2"
        mouths_root = dataset_dir / "mouths"
        self._embed_dir = ROOT_PATH / "src/data/embeddings"
        self.embed_exists = self._embed_dir.exists() and any(self._embed_dir.iterdir())
        self.AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")

        if not mix_audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {mix_audio_dir}")

        data = []
        for mix_file in sorted(mix_audio_dir.glob("*")):
            if mix_file.suffix.lower() not in self.AUDIO_EXTENSIONS:
                continue
            base_name = mix_file.stem
            first_id, second_id = base_name.split("_")

            s1_path = None
            s2_path = None
            for ext in self.AUDIO_EXTENSIONS:
                s1_candidate = s1_audio_dir / f"{base_name}{ext}"
                s2_candidate = s2_audio_dir / f"{base_name}{ext}"
                if s1_candidate.exists():
                    s1_path = s1_candidate
                if s2_candidate.exists():
                    s2_path = s2_candidate

            mouth1_path = mouths_root / f"{first_id}.npz"
            mouth2_path = mouths_root / f"{second_id}.npz"

            if self.embed_exists:
                s1_emb_path = self._embed_dir / f"{first_id}.npz"
                s2_emb_path = self._embed_dir / f"{second_id}.npz"
            else:
                s1_emb_path = None
                s2_emb_path = None

            t_info = torchaudio.info(str(mix_file))
            length = t_info.num_frames / t_info.sample_rate

            if not mix_file.exists():
                continue

            t_info = torchaudio.info(str(mix_file))
            length = t_info.num_frames / t_info.sample_rate

            entry = {
                "mix_path": str(mix_file.resolve()),
                "s1_path": str(s1_path.resolve()) if s1_path else None,
                "s2_path": str(s2_path.resolve()) if s2_path else None,
                "mouth1_path": str(mouth1_path.resolve())
                if mouth1_path.exists()
                else None,
                "mouth2_path": str(mouth2_path.resolve())
                if mouth2_path.exists()
                else None,
                "s1_emb_path": str(s1_emb_path.resolve())
                if (s1_emb_path is not None and s1_emb_path.exists())
                else None,
                "s2_emb_path": str(s2_emb_path.resolve())
                if (s2_emb_path is not None and s2_emb_path.exists())
                else None,
                "audio_len": length,
                "speakers": [first_id, second_id],
            }

            data.append(entry)

        if len(data) == 0:
            raise RuntimeError(f"No audio files found in {mix_audio_dir}")

        super().__init__(data, *args, **kwargs)
