import yaml
import os
import os.path
import pickle

import torch

def load_blunder_model_config(config_dir_path):
    with open(os.path.join(config_dir_path, 'config.yaml')) as f:
        config = yaml.safe_load(f.read())
    if config['engine'] == 'sklearn':
        weightsPath = os.path.join(config_dir_path, config['options']['weightsPath'])
        with open(weightsPath, 'rb') as f:
            model = pickle.load(f)
    elif config['engine'] == 'torch':
        weightsPath = os.path.join(config_dir_path, config['options']['weightsPath'])
        model = torch.load(weightsPath)
    else:
        raise NotImplementedError(f"{config['engine']} is not a known engine type")
    return model, config
