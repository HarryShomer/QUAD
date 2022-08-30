#!/bin/bash
### Reproduce our results on Wikipeople

# Non-Parallel
python src/run.py wikipeople --alpha .8  --qual-layers 2 --encoder-drop1 .1 --encoder-drop2 .1 \
                             --trans-dim 512 --lr-decay --lr-lambda .999 --mask --label-smooth 0.2 --device cuda

# Parallel
python src/run.py wikipeople --alpha .8  --qual-layers 2 --encoder-drop1 .1 --encoder-drop2 .1 \
                             --trans-dim 512 --lr-decay --lr-lambda .999 --mask --label-smooth 0.2 \
                             --parallel --parallel-drop .2 --device cuda
