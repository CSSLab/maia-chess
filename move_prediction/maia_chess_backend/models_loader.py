import os
import os.path
import yaml


class Trained_Model(object):
    def __init__(self, path):
        self.path = path
        try:
            with open(os.path.join(path, 'config.yaml')) as f:
                self.config = yaml.safe_load(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"No config file found in: {path}")

        self.weights = {int(e.name.split('-')[-1].split('.')[0]) :e.path for e in os.scandir(path) if e.name.endswith('.txt') or e.name.endswith('.pb.gz')}

    def getMostTrained(self):
        return self.weights[max(self.weights.keys())]
