import zipfile
from pathlib import Path

import hydra
import requests
import yadisk
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="src/configs", config_name="download_data")
def main(config):
    data_dir = config.data_dir
    public_url = config.public_url
    file_name = config.file_name
    y = yadisk.YaDisk()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / file_name

    print(public_url)
    print("Dowload dataset.")
    y.download_public(public_url, file_name, file_path=data_dir)

    print("Unzip data")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset is ready")
    print(f"Location: {data_dir / 'dla_dataset'}")


if __name__ == "__main__":
    main()
