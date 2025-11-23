import json
import os
import zipfile
from pathlib import Path

import requests
import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

public_url = "YOUR DATASET LINK"
api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_url}"
download_url = requests.get(api_url).json()["href"]

URL_LINKS = {
    "dataset": download_url,
}

ROOT_PATH = Path(ROOT_PATH)


class AVSSDataset(BaseDataset):
    """
    Датасет для задачи AVSS.
    Структура:
    dla_dataset/
      ├── audio/
      │   ├── {train|val|test}/
      │   │   ├── mix/
      │   │   ├── s1/
      │   │   └── s2/
      └── mouths/
          ├── {speaker_id}.npz
    """

    def __init__(self, part: str, data_dir=None, embed_dir=None, *args, **kwargs):
        assert part in [
            "train",
            "val",
            "test",
        ], "Аргумент part должен быть одним из ['train', 'val', 'test']"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "avss"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)

        if embed_dir is None:
            self._embed_dir = ROOT_PATH / "src/data/embeddings"
        else:
            self._embed_dir = ROOT_PATH / Path(embed_dir)

        self.embed_exists = self._embed_dir.exists() and any(self._embed_dir.iterdir())

        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _load_dataset_zip(self):
        zip_path = self._data_dir / "dla_dataset.zip"
        if not zip_path.exists():
            print("Скачиваем AVSS датасет...")

            url = URL_LINKS["dataset"]
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"Ошибка при скачивании: {response.status_code}")

            total = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading AVSS"
            ) as pbar:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print("AVSS датасет скачан...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                print("Распаковка архива...")
                zip_ref.extractall(self._data_dir)
        except zipfile.BadZipFile:
            raise RuntimeError(f"Файл {zip_path} повреждён или не является ZIP-архивом")

        print("AVSS датасет успешно распакован!")

    def _get_or_load_index(self, part: str):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                return json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
            return index

    def _create_index(self, part: str):
        dataset_root = self._data_dir / "dla_dataset"
        if not dataset_root.exists():
            self._load_dataset_zip()

        audio_root = dataset_root / "audio" / part
        mouths_root = dataset_root / "mouths"
        mix_dir = audio_root / "mix"
        s1_dir = audio_root / "s1"
        s2_dir = audio_root / "s2"

        index = []

        supported_exts = [".wav", ".flac", ".mp3"]
        print(f"Создание индекса для части '{part}'...")
        for mix_file in tqdm(
            sorted(mix_dir.glob("*")), desc=f"Создание индекса: {part}"
        ):
            if mix_file.suffix.lower() not in supported_exts:
                continue

            # имя файла = "FirstSpeakerID_SecondSpeakerID"
            base_name = mix_file.stem
            try:
                first_id, second_id = base_name.split("_")
            except ValueError:
                continue

            # ищем ground truth, если есть
            s1_path = None
            s2_path = None
            for ext in supported_exts:
                s1_candidate = s1_dir / f"{base_name}{ext}"
                s2_candidate = s2_dir / f"{base_name}{ext}"
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

            index.append(entry)

        return index
