#!/bin/bash
### Reproduce our results on WD50K

# Non-Parallel
python src/run.py wd50k --clean-data --alpha .7 --aux-ent --aux-weight .5 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                        --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --device cuda

# Parallel
python src/run.py wd50k --clean-data --alpha .7 --aux-ent --aux-weight .5 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                        --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --parallel --parallel-drop .2 --device cuda
