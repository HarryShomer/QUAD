#!/bin/bash
### Reproduce our results on JF17K

# Non-Parallel
python src/run.py jf17k --alpha .8 --aux-ent --aux-weight .25 --qual-layers 2 --encoder-drop1 .1 --encoder-drop2 .1 \
                        --trans-dim 768 --lr-decay --lr-lambda .999 --mask --label-smooth 0.6 \
                        --aux-ent-smooth 0.6 --device cuda

# Parallel
python src/run.py jf17k --alpha .8 --aux-ent --aux-weight .25 --qual-layers 2 --encoder-drop1 .1 --encoder-drop2 .1 \
                        --trans-dim 768 --lr-decay --lr-lambda .999 --mask --label-smooth 0.6 --aux-ent-smooth 0.6 \
                        --parallel --parallel-drop .2 --device cuda
