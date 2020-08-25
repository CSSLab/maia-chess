#!/usr/bin/env python3
import argparse
import os
import yaml
import sys
import tensorflow as tf
from .tfprocess import TFProcess

START_FROM = 0

def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    root_dir = os.path.join(cfg['training']['path'], cfg['name'])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    tfprocess = TFProcess(cfg)
    tfprocess.init_net_v2()

    tfprocess.restore_v2()

    START_FROM = cmd.start

    tfprocess.global_step.assign(START_FROM)
    tfprocess.manager.save()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
    'Convert current checkpoint to new step count.')
    argparser.add_argument('--cfg', type=argparse.FileType('r'),
        help='yaml configuration with training parameters')
    argparser.add_argument('--start', type=int, default=0,
        help='Offset to set global_step to.')

    main(argparser.parse_args())
