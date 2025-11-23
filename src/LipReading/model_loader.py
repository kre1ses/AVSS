""" TCN for lipreading"""

import os
import numpy as np
import torch

from src.LipReading.lipreading.utils import load_json, load_model
from src.LipReading.lipreading.model import Lipreading
from src.LipReading.lipreading.dataloaders import get_preprocessing_pipelines


def extract_feats(model, mouths_path):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines(modality='video')['test']
    data = preprocessing_func(np.load(mouths_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].to('cpu'), lengths=[data.shape[0]])


def load_json_model_parameters(config_path: str, model_path: str):
    assert config_path.endswith('.json') and os.path.isfile(config_path), \
        f"'.json' config path does not exist. Path input: {config_path}"
    
    args_loaded = load_json(config_path)
    backbone_type = args_loaded['backbone_type']
    width_mult = args_loaded['width_mult']
    relu_type = args_loaded['relu_type']
    use_boundary = False

    if args_loaded.get('tcn_num_layers', ''):
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }
    else:
        tcn_options = {}
    if args_loaded.get('densetcn_block_config', ''):
        densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                            'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                            'reduced_size': args_loaded['densetcn_reduced_size'],
                            'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                            'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                            'squeeze_excitation': args_loaded['densetcn_se'],
                            'dropout': args_loaded['densetcn_dropout'],
                            }
    else:
        densetcn_options = {}

    model = Lipreading( modality='video',
                        num_classes=500,
                        tcn_options=tcn_options,
                        densetcn_options=densetcn_options,
                        backbone_type=backbone_type,
                        relu_type=relu_type,
                        width_mult=width_mult,
                        use_boundary=use_boundary,
                        extract_feats=True)
    
    model = load_model(model_path, model, allow_size_mismatch=True)

    return model


