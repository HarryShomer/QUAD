# QUAD

A PyTorch implementation of the paper ["Learning Representations for Hyper-Relational Knowledge Graphs"](https://arxiv.org/abs/2208.14322).

## Abstract

Knowledge graphs (KGs) have gained prominence for their ability to learn representations for uni-relational facts. Recently, research has focused on modeling hyper-relational facts, which move beyond the restriction of uni-relational facts and allow us to represent more complex and real-world information. However, existing approaches for learning representations on hyper-relational KGs majorly focus on enhancing the communication from qualifiers to base triples while overlooking the flow of information from base triple to qualifiers. This can lead to suboptimal qualifier representations, especially when a large amount of qualifiers are presented.  It motivates us to design a  framework that utilizes multiple aggregators to learn representations for hyper-relational facts: one from the perspective of the base triple and the other one from the perspective of the qualifiers. Experiments demonstrate the effectiveness of our framework for hyper-relational knowledge graph completion across multiple datasets. Furthermore, we conduct an ablation study that validates the importance of the various components in our framework. 

## Requirements

All experiments were conducted using python 3.8.5. 

For the required python packages, please see `requirements.txt`.

```
tqdm>=4.64.0
torch>=1.9.0
numpy>=1.19.12
torch_geometric>=2.0.2
pytorch_warmup>=0.0.4
```

## Running the Code

First clone our repository and install the required python packages.
```
git clone https://github.com/HarryShomer/QUAD.git
cd QUAD
pip install -r requirements.txt
```

To reproduce the results on each dataset you can run the following bash scripts. Each scripts contains the command to replicate both the parallel and non-parallel version of the model.
```
# Reproduce results on JF17K
bash scripts/run_jf17k.sh

# Reproduce results on WD50K
bash scripts/run_wd50k.sh

# Reproduce results on Wikipeople
bash scripts/run_wikipeople.sh

# Reproduce results on the WD50K splits - (33, 66, 100)
bash scripts/wd50k_splits.sh
```

By default the scripts try to run on cuda. To run on the cpu add the argument `--device cpu` 


## Acknowledgement

The code is modified from the StarE codebase (https://github.com/migalkin/StarE). We sincerely thank them for their contributions.


## Cite

```
@article{shomer2022learning,
  title={Learning Representations for Hyper-Relational Knowledge Graphs},
  author={Shomer, Harry and Jin, Wei and Li, Juanhui and Ma, Yao and Tang, Jiliang},
  journal={arXiv preprint arXiv:2208.14322},
  year={2022}
}
```