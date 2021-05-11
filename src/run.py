import os
os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import sys
import collections
import argparse

# Local imports
from data_loaders.data_manager import DataManager
from utils.utils import *
from utils.utils_mytorch import FancyDict, parse_args, BadParameters
from loops.evaluation import EvaluationBenchGNNMultiClass, evaluate_pointwise
from loops.evaluation import acc, mrr, mr, hits_at
from loops.corruption import Corruption
from loops.sampler import MultiClassSampler
from loops.loops import training_loop_gcn

# from models.hyper_gcn import *
from models.old_hyper_gcn import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# These shouldn't change
DEFAULT_CONFIG = {
    'GRAD_CLIPPING': True,
    'STATEMENT_LEN': -1,
    'SAMPLER_W_QUALIFIERS': True,
    'ENT_POS_FILTERED': True,
    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'hyper_kg',
    'CORRUPTION_POSITIONS': [0, 2]
}

# TODO: Add to Argparse
MODEL_CONFIG = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 200,
    'GCN_DROP': 0.1, 

    # Used *after* conv layers
    'ENCODER_DROP_1': 0.3,
    'ENCODER_DROP_2': 0.1,

    'BIAS': False,
    'OPN': 'corr',

    # For ConvE Only
    'FEAT_DROP': 0.3,
    'N_FILTERS': 200,
    'KERNEL_SZ': 7,
    'K_W': 10,
    'K_H': 20,

    # For Transformer
    'T_LAYERS': 2,
    'T_N_HEADS': 4,
    'T_HIDDEN': 512,
    'TRANSFORMER_DROP': .1,
    'POSITIONAL': True,
    'POS_OPTION': 'default',
    'TIME': False,
    'POOLING': 'avg'
}


### Cmd Line Args!
parser = argparse.ArgumentParser()

parser.add_argument("dataset", help="Dataset to run it on")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--epochs", help="Number of epochs to run", default=500, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--dim", help="Latent dimension of entities and relations", type=int, default=200)
parser.add_argument("--opn", help="Composition function", type=str, default="corr")
parser.add_argument("--val-every", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--label-smoothing", default=.1, type=float)
parser.add_argument("--lr-decay", action='store_true', default=False)
parser.add_argument("--early-stopping", help="Number of validation scores to wait for an increase before stopping", default=5, type=int)
parser.add_argument("--dropout", help="Dropout for encoder", default=.3, type=float)
parser.add_argument("--trans-dim", help="Transformer hidden dimension", type=int, default=512)

# Type of embedding to use
parser.add_argument("--emb-type", help="Choices -> ['same', 'project', base', diff']", type=str, default='same')

# should be false for WikiPeople and JF17K for their original data
parser.add_argument("--cleaned-dataset", action='store_true', default=False)

# Hyperparams controlling flow of qualifier info
parser.add_argument("--alpha", type=float, default=.5)
parser.add_argument("--beta", type=float, default=.5)

cmd_args = parser.parse_args()


DEFAULT_CONFIG['MODEL'] = MODEL_CONFIG
DEFAULT_CONFIG['BATCH_SIZE'] = cmd_args.batch_size
DEFAULT_CONFIG['DATASET'] = cmd_args.dataset.lower()
DEFAULT_CONFIG['DEVICE'] = torch.device(cmd_args.device)
DEFAULT_CONFIG['EMBEDDING_DIM'] = cmd_args.dim
DEFAULT_CONFIG['EPOCHS'] = cmd_args.epochs 
DEFAULT_CONFIG['EVAL_EVERY'] = cmd_args.val_every
DEFAULT_CONFIG['LEARNING_RATE'] = cmd_args.lr
DEFAULT_CONFIG['LABEL_SMOOTHING'] = cmd_args.label_smoothing
DEFAULT_CONFIG['CLEANED_DATASET'] = cmd_args.cleaned_dataset
DEFAULT_CONFIG['LR_SCHEDULER'] = cmd_args.lr_decay
DEFAULT_CONFIG['EARLY_STOPPING'] = cmd_args.early_stopping
DEFAULT_CONFIG['ALPHA'] = cmd_args.alpha
DEFAULT_CONFIG['BETA'] = cmd_args.beta
DEFAULT_CONFIG['MODEL']['EMB_TYPE'] = cmd_args.emb_type
DEFAULT_CONFIG['MODEL']['OPN'] = cmd_args.opn
DEFAULT_CONFIG['MODEL']['ENCODER_DROP_1'] = cmd_args.dropout
DEFAULT_CONFIG['MODEL']['T_HIDDEN'] = cmd_args.trans_dim



def get_data(config):
    """
    Get the data for the specific dataset requested
    """
    data = DataManager.load(config=config)()
    train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()


    # all_data = [train_data, valid_data, test_data]
    
    # main_ents = set([t[0] for d in all_data for t in d] + [t[2] for d in all_data for t in d])

    # qual_ents = []
    # for d in all_data:
    #     for t in d:
    #         qual_ents.extend(t[4::2])

    # qual_ents = set(qual_ents)
    # main_ents = set(main_ents)
    # ent_intersection = list(main_ents & qual_ents)

    # print("Num Entities:", n_entities)
    # print(f"Main Entities: {len(main_ents)}")
    # print(f"Qualifier Entities: {len(qual_ents)}")
    # print(f"Intersection Entities: {len(ent_intersection)}")

    # print(f"\n{(len(ent_intersection) / n_entities * 100):.2f}% of the main entities are found in the qualifiers")
    # exit()


    config['NUM_ENTITIES'] = n_entities
    config['NUM_RELATIONS'] = n_relations

    # Exclude entities which don't appear in the dataset. E.g. entity nr. 455 may never appear.
    # always off for wikipeople and jf17k
    if config['DATASET'] == 'jf17k' or config['DATASET'] == 'wikipeople':
        config['ENT_POS_FILTERED'] = False

    if config['ENT_POS_FILTERED']:
        ent_excluded_from_corr = DataManager.gather_missing_entities(
            data=train_data + valid_data + test_data,
            positions=config['CORRUPTION_POSITIONS'],
            n_ents=n_entities)
    else:
        ent_excluded_from_corr = [0]


    # -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
    # -> edge_type (n) array with [relation] corresponding to sub, obj above
    # -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals
    train_data_gcn = DataManager.get_alternative_graph_repr(train_data + valid_data, config)
    #train_data_gcn = DataManager.get_alternative_graph_repr(train_data, config)

    # add reciprocals to the train data
    reci = DataManager.add_reciprocals(train_data, config)
    train_data.extend(reci)
    reci_valid = DataManager.add_reciprocals(valid_data, config)
    valid_data.extend(reci_valid)
    reci_test = DataManager.add_reciprocals(test_data, config)
    test_data.extend(reci_test)

    return train_data_gcn, train_data, valid_data, test_data, ent_excluded_from_corr




def main():
    config = DEFAULT_CONFIG.copy()

    print("\nConfig\n------")
    print(config)
    print("\n")

    ## Specified Dataset splits
    train_data_gcn, train_data, valid_data, test_data, ent_excluded_from_corr = get_data(config)

    ## Model
    model = HypRelModel(train_data_gcn, config)
    model.to(config['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    

    """
    Prepare test benches.
    
        -  When computing train accuracy (`ev_tr_data`), we wish to use all the other data 
            to avoid generating true triples during corruption. 
        -  Similarly, when computing test accuracy, we index train and valid splits to avoid 
            generating negative triples.
    """
    ev_vl_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
    ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
    tr_data = {'train': combine(train_data, valid_data), 'valid': ev_vl_data['eval']}
    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]

    sampler = MultiClassSampler(data= tr_data['train'],
                                n_entities=config['NUM_ENTITIES'],
                                lbl_smooth=config['LABEL_SMOOTHING'],
                                bs=config['BATCH_SIZE'],
                                with_q=config['SAMPLER_W_QUALIFIERS'])
    evaluation_valid = EvaluationBenchGNNMultiClass(ev_vl_data, model, bs=config['BATCH_SIZE'], metrics=eval_metrics,
                                                    filtered=True, n_ents=config['NUM_ENTITIES'],
                                                    excluding_entities=ent_excluded_from_corr,
                                                    positions=config.get('CORRUPTION_POSITIONS', None), config=config)


    # The args to use if we're training w default stuff
    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "model": model,
        "neg_generator": Corruption(n=config['NUM_ENTITIES'], excluding=[0], position=list(range(0, config['MAX_QPAIRS'], 2))),
        "device": config['DEVICE'],
        "data_fn": sampler.reset,
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run,
        "trn_testbench": None,
        "eval_every": config['EVAL_EVERY'],
        "run_trn_testbench": False,
        "qualifier_aware": config['SAMPLER_W_QUALIFIERS'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "early_stopping": config['EARLY_STOPPING'],
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95) if config['LR_SCHEDULER'] else None
    }

    print(f"Training on {config['NUM_ENTITIES']} entities and {config['NUM_RELATIONS']} relations!\n")

    traces = training_loop_gcn(**args)
    save_model(model)
    


if __name__ == "__main__":
    main()
