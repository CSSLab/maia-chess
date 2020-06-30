import yaml
import os
import os.path

from .tourney import RandomEngine, StockfishEngine, LC0Engine

def load_model_config(config_dir_path, lc0_depth = None, lc0Path = None, noise = False, temperature = 0, temp_decay = 0):
    with open(os.path.join(config_dir_path, 'config.yaml')) as f:
        config = yaml.safe_load(f.read())

    if config['engine'] == 'stockfish':
        model = StockfishEngine(**config['options'])
    elif config['engine'] == 'random':
        model = RandomEngine()
    elif config['engine'] == 'torch':
        raise NotImplementedError("torch engines aren't working yet")
    elif config['engine'] in ['lc0', 'lc0_23']:
        kwargs = config['options'].copy()
        if lc0_depth is not None:
            kwargs['nodes'] = lc0_depth
            kwargs['movetime'] *= lc0_depth / 10
        kwargs['weightsPath'] = os.path.join(config_dir_path, config['options']['weightsPath'])
        model = LC0Engine(lc0Path = config['engine'] if lc0Path is None else lc0Path, noise = noise, temperature = temperature, temp_decay = temp_decay, **kwargs)
    else:
        raise NotImplementedError(f"{config['engine']} is not a known engine type")

    return model, config
