# Bridging Audio and Audio-Visual Source Separation: A Review and Experimental Evaluation of Modern Models

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains implementations of Audio-Visual Source Separation models based on the papers
[*Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*](https://arxiv.org/pdf/1809.07454)  by Yi Luo, Nima Mesgarani et al. (2019) and
[*Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation*](https://arxiv.org/pdf/2007.13975)  by Jingjing Chen, Qirong Mao, Dong Liu et al. (2020)

## Installation

Download all dependencies:
```bash
pip install -r ./requirements.txt
```

Install `pre-commit`:
```bash
pre-commit install
```

Set cometml api key locally
```bash
export COMET_API_KEY="YOUR-KEY"
```

Download one of the models:
```bash
python3 download_model.py --config-name download_model ++link='link for chosen model'
```

List of available models:

```python
available_links = {
    'av_convtasnet with gate fusion': "https://drive.google.com/uc?id=1JeGi10EHFrDIx2WjceL6Vcz-4qWoPv82&export=download",
    'dptn': "https://drive.google.com/uc?id=1qujtsl1wmv-zMMd3vm_6XJDtOs6UtFG3&export=download",
    'av_dptn with gate fusion': "https://drive.google.com/uc?id=1TlKEfqjZIV4kMVIbgGJd0y2fXw5J3efB&export=download",
    'av_dptn with attention fusion': "https://drive.google.com/uc?id=1jf-p_rH5S-s1y40THRqOVY1Fw_gyvFZj&export=download",
    'av_dptn with linear fusion': "https://drive.google.com/uc?id=1tdx4q6UgXRfaabTvwGQVfoEni-szkAg_&export=download",
    'fast av_dptn with linear fusion': "https://drive.google.com/uc?id=1-7aNkDMWpMfxjJhDIE8rtK4m0QkumyKV&export=download"
}
```

If you have GPU locally, run this:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## How To Use

(Optional) If you do not have your datab for inference locally, dowload it from Yandex Disk with script:

```bash
python3 download_data.py --config-name download_data ++data_dir="data_dir" ++public_url='Link to your dataset' ++file_name="your file name.zip"
```

Install model for video embeddings:

```bash
python3 get_video_embeddings.py -cn=make_video_embeddings mouths_path="data_dir\mouths"
```

To train a model, run the following command:

```bash
python3 train.py -cn=dptn
```

To run inference (evaluate the model or save predictions) with custom dataset:

```bash
python3 inference.py -cn=inference_av_dptn
```

Make sure, your custom dataset has following format

```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```


To calculate SI-SNR, SI-SNRi, STOI, PESQ and SDRi run:

```bash
python3 calc_metrics.py -cn=calculate_metrics ++predictions_path='"data/saved/result"' ++groud_truth_path='"data_dir/audio"' ++mix_path='"data_dir/audio"'
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
