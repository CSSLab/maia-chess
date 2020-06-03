# Aligning Superhuman AI and Human Behavior: Chess as a Model System

We are currently working on the full release of code, data and models (and our followup paper). The work will be presented at KDD '20 and the preprint is on arXiv.

You can play against three of of our models on Lichess right now:

+ [maia1](https://lichess.org/@/maia1) is targeting ELO 1100
+ [maia5](https://lichess.org/@/maia5) is targeting ELO 1500
+ [maia9](https://lichess.org/@/maia9) is targeting ELO 1900

## Model Files

The model weights we used in the paper are in `model_files` one for each of the 9 ELO ranges we trained. The `.pb.gz` files should work with [`lc0`](https://github.com/LeelaChessZero/lc0). The config files have the recommended options for use. Using `nodes 1` is important as the models are most human without any search.
