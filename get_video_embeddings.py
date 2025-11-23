import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.LipReading.lipreading.dataloaders import get_preprocessing_pipelines
from src.LipReading.lipreading.model import Lipreading
from src.LipReading.model_loader import load_json_model_parameters
from src.utils.init_utils import load_model_from_gdown
from src.utils.io_utils import ROOT_PATH


@hydra.main(
    version_base=None, config_path="src/configs", config_name="make_video_embeddings"
)
def main(config):
    """
    Loads video model and its hyperparameters using config_path and model_path,
    reads video from mouth_dir and stores video embeddings in embed_dir

    !!!!!! embed_dir must be the same in Hydra dataset config
    """
    url_link = "https://drive.google.com/uc?id=1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD"
    load_path = str(
        ROOT_PATH / "src" / "data" / "models" / "lrw_resnet18_mstcn_video.pth"
    )
    config_path = str(
        ROOT_PATH / "src" / "LipReading" / "configs" / "lrw_resnet18_mstcn.json"
    )
    embed_dir = str(ROOT_PATH / "src" / "data" / "embeddings")

    mouths_dir = str(ROOT_PATH / config.mouths_path)
    mouths_dir = os.path.normpath(mouths_dir).replace("\\", "/")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("slow inference incoming")

    if not os.path.exists(str(ROOT_PATH / "src" / "data" / "models")):
        os.makedirs(str(ROOT_PATH / "src" / "data" / "models"), exist_ok=True)

    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir, exist_ok=True)

    weights_path = load_model_from_gdown(url_link, load_path)
    lipreader = load_json_model_parameters(config_path, weights_path)
    lipreader.eval().to(device)

    preprocessing = get_preprocessing_pipelines(modality="video")["test"]

    for file in tqdm(os.listdir(mouths_dir), desc="extracting video embeddings"):
        if ".npz" in str(file):
            full_path = os.path.join(mouths_dir, file)
            processed_video = preprocessing(np.load(full_path)["data"])
            torch_video = torch.FloatTensor(processed_video)[None, None, :, :, :].to(
                device
            )  # [1, 1, T, H, W]
            emb = lipreader(torch_video, lengths=[50])  # [1, 50, 512]
            emb = emb.squeeze(0).transpose(0, 1)  # [512, 50]
            np_emb = emb.detach().cpu().numpy()
            emb_path = os.path.join(embed_dir, file)
            np.savez_compressed(emb_path, embedding=np_emb)

    print(f"Extraction complete. Embeddings saved at {embed_dir}.")


if __name__ == "__main__":
    main()
