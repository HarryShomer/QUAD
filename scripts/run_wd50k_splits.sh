#1/bin/bash
### Reproduce our results on the WD50K splits

##### WD50K (33)#####

# Non-Parallel
python src/run.py wd50k_33 --clean-data --alpha .7 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                           --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --device cuda

# Parallel
python src/run.py wd50k_33 --clean-data --alpha .7 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                           --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --parallel --parallel-drop .3 --device cuda


##### WD50K (66)#####

# Non-Parallel
python src/run.py wd50k_66 --clean-data --alpha .7 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                           --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --device cuda

# Parallel
python src/run.py wd50k_66 --clean-data --alpha .7 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                           --trans-dim 768 --lr-decay --lr-lambda .9975 --mask --label-smooth 0.2 --aux-ent-smooth 0.2 --parallel --parallel-drop .3 --device cuda


##### WD50K (100)#####

# Non-Parallel
python src/run.py wd50k_100 --clean-data --alpha .6 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                            --batch-size 256 --trans-dim 768 --mask --label-smooth 0.1 --aux-ent-smooth 0.1 --device cuda

# Parallel
python src/run.py wd50k_100 --clean-data --alpha .6 --aux-ent --aux-weight 1 --qual-layers 1 --encoder-drop1 .2 --encoder-drop2 .2 \
                            --batch-size 256 --trans-dim 768 --mask --label-smooth 0.1 --aux-ent-smooth 0.1 --parallel --parallel-drop .3 --device cuda