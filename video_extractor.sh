#!/bin/bash

source .venv/bin/activate

if ! command -v gdown &> /dev/null
then
    echo "gdown not found"
    pip install gdown
fi

MOUTHS_DIR="${1:-mouths}"
EMBED_DIR="${2:-src/data/embeddings}"

export URL_LINK="https://drive.google.com/uc?id=1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD"
export LOAD_PATH="src/data/models/lrw_resnet18_mstcn_video.pth"
export CONFIG_PATH="src/LipReading/configs/lrw_resnet18_mstcn.json"
export MOUTHS_DIR
export EMBED_DIR

python3 get_video_embeddings.py