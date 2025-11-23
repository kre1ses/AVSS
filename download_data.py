import argparse
import zipfile
from pathlib import Path

import hydra
import requests
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="src/configs", config_name="download_data")
def main(config):
    data_dir = config.data_dir
    public_url = config.public_url
    file_name = config.file_name
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / file_name

    print(public_url)
    api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_url}"

    download_url = requests.get(api_url).json()["href"]

    print("Dowload dataset.")
    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        desc="Progress",
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print("Unzip data")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset is ready")
    print(f"Location: {data_dir}")


if __name__ == "__main__":
    main()
