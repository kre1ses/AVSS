from pathlib import Path

import gdown
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

path = "models/conformer_best.pth"


@hydra.main(version_base=None, config_path="src/configs", config_name="download_model")
def main(config):
    model_path = Path("models").absolute().resolve()
    model_path.mkdir(exist_ok=True, parents=True)

    output_path = model_path / "model_best.pth"
    gdown.download(config.link, str(output_path), quiet=False)


if __name__ == "__main__":
    main()
