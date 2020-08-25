#!/usr/bin/env python3
import argparse
import tensorflow as tf
import os
import yaml
from .tfprocess import TFProcess
from .net import Net

argparser = argparse.ArgumentParser(description='Convert net to model.')
argparser.add_argument('net', type=str,
    help='Net file to be converted to a model checkpoint.')
argparser.add_argument('--start', type=int, default=0,
    help='Offset to set global_step to.')
argparser.add_argument('--cfg', type=argparse.FileType('r'),
    help='yaml configuration with training parameters')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = args.start
net = Net()
net.parse_proto(args.net)

filters, blocks = net.filters(), net.blocks()
if cfg['model']['filters'] != filters:
    raise ValueError("Number of filters in YAML doesn't match the network")
if cfg['model']['residual_blocks'] != blocks:
    raise ValueError("Number of blocks in YAML doesn't match the network")
weights = net.get_weights()

tfp = TFProcess(cfg)
tfp.init_net_v2()
tfp.replace_weights_v2(weights)
tfp.global_step.assign(START_FROM)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
tfp.manager.save()
print("Wrote model to {}".format(tfp.manager.latest_checkpoint))
