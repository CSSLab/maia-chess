# Aligning Superhuman AI with Human Behavior: Chess as a Model System

## [paper](https://arxiv.org/abs/2006.01855)/[code](https://github.com/CSSLab/maia-chess)/[lichess](https://lichess.org/team/maia-bots)

A collection of chess engines that play like humans, from ELO 1100 to 1900.

![The accuracy of the different maias across ELO range](maia_lineplot.png)

In this repo is our 9 final maia models saved as Leela Chess neural networks, and the code to create more and reproduce our results.

You can also play against three of of our models on Lichess:

+ [`maia1`](https://lichess.org/@/maia1) is targeting ELO 1100
+ [`maia5`](https://lichess.org/@/maia5) is targeting ELO 1500
+ [`maia9`](https://lichess.org/@/maia9) is targeting ELO 1900

We also have a Lichess team, [_maia-bots_](https://lichess.org/team/maia-bots), that we will more bots to.

## Chess Engine

The models (`.pb.gz` files) work like any other Leela weights file. So to use them download or compile [`lc0`](http://lczero.org). If the version of `lc0` does not support the weights we have the exact version [here](https://github.com/CSSLab/lc0_23) to compile.

When using the model in `UCI` mode add `nodes 1` when querying as that disables the search.

## Datasets

As part of our analysis all the game on Lichess with stockfish analysis were processed into csv files. These can be found [here](http://csslab.cs.toronto.edu/datasets/chess/kdd2020/)

## Code

### Move Prediction

To create your own maia from a set of chess games in the PGN format:

1. Install the backend
   1. (optional) Install the `conda` environment, [`maia_env.yml`](maia_env.yml)
   2. Add the backend to env `python setup.py install`
2. Convert the PGN into the training format
   1. Add the [`pgn-extract`](https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/) tool to your path
   2. Add the [`trainingdata-tool`](https://github.com/DanielUranga/trainingdata-tool) to your path
   3. Run `move_prediction/0-pgn_to_trainingdata.sh PGN_FILE_PATH OUTPUT_PATH`
   4. The script will create a training and validation set, if you wish to train on the whole set copy the files from `OUTPUT_PATH/validation` to `OUTPUT_PATH/training`
3. Edit `move_prediction/maia_config.yml`
   1. Add  `OUTPUT_PATH/train/*/*` to `input_train`
   2. Add  `OUTPUT_PATH/validation/*/*` to `input_test`
   3. (optional) If you have multiple GPUS change the `gpu` filed to the one you are using
4. Run the training script `move_prediction/train_maia.py PATH_TO_CONFIG`
5. (optional) You cna use tensorboard to watch the training progress, the logs are in `runs/CONFIG_BASENAME/`
6. Once complete the final model will be in `models/CONFIG_BASENAME/` directory. It will be the one with the largest numbers

### Blunder Prediction

>>> IN PROGRESS
