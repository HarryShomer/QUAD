import random
import argparse
from functools import partial
import pytorch_warmup as warmup

from data_loaders.data_manager import DataManager
from utils.utils import *
from loops.evaluation import EvaluationBenchGNNMultiClass
from loops.evaluation import acc, mrr, mr, hits_at
from loops.sampler import MultiClassSampler
from loops.loops import training_loop_gcn
from models.hyper_gcn import HypRelModel


random.seed(42)
np.random.seed(42)
torch.manual_seed(132)
torch.cuda.manual_seed_all(132)


# These shouldn't change
DEFAULT_CONFIG = {
    'GRAD_CLIPPING': True,
    'STATEMENT_LEN': -1,
    'SAMPLER_W_QUALIFIERS': True,
    'ENT_POS_FILTERED': True,
    'MAX_QPAIRS': 15,
    'MODEL_NAME': 'hyper_kg',
    'CORRUPTION_POSITIONS': [0, 2]  # head and tail
}

# TODO: Add to Argparse
MODEL_CONFIG = {
    'GCN_DIM': 200,
    'POSITIONAL': True,
    #'POS_OPTION': 'default',
    'TIME': False
}

### Cmd Line Args!
parser = argparse.ArgumentParser()

parser.add_argument("dataset", help="Dataset to run it on")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--clean-data", help="Only use for WD50K datasets", action='store_true', default=False) # False for WikiPeople and JF17K for original data
parser.add_argument("--only-trips", help="Only do triplet aggregation", action='store_true', default=False)
parser.add_argument("--test-on-val", help="When true testing done on val set. Otherwise we train on train+val and test on test set.", action='store_true', default=False)
parser.add_argument("--val-every", help="Test on validation set every n epochs", type=int, default=5)
parser.add_argument("--early-stopping", help="Number of validation scores to wait for an increase before stopping", default=None, type=int)

parser.add_argument("--epochs", help="Number of epochs to run", default=500, type=int)
parser.add_argument("--batch-size", help="Batch size to use for training", default=128, type=int)
parser.add_argument("--lr", help="Learning rate to use while training", default=1e-4, type=float)
parser.add_argument("--lr-decay", action='store_true', default=False)
parser.add_argument("--lr-lambda", help="LR Decay", default=.9975, type=float)
parser.add_argument("--dim", help="Latent dimension of entities and relations", type=int, default=200)
parser.add_argument("--opn", help="Composition function", type=str, default="rotate")
parser.add_argument("--label-smooth", default=.1, type=float)
parser.add_argument("--warmup", action='store_true', default=False)

parser.add_argument("--gcn-drop", help="Dropout for GCN", type=float, default=.1)
parser.add_argument("--encoder-drop1", help="1st Dropout for encoder", default=.3, type=float)
parser.add_argument("--encoder-drop2", help="2nd Dropout for encoder", default=.3, type=float)

parser.add_argument("--fact-encoder", help="linear or transformer", type=str, default="linear")
parser.add_argument("--qual-layers", help="Number of Layers in Qual Encoder", default=2, type=int)
parser.add_argument("--trip-layers", help="Number of Layers in Trip Encoder", default=2, type=int)

parser.add_argument("--alpha", help="Parameter for controlling flow of qualifier info", type=float, default=.7)
parser.add_argument("--qual-comb", help="How to combine qual information in trip encoder. One of ['none', 'both', 'ent', 'rel', 'out']", type=str, default='out')

parser.add_argument("--trans-dim", help="Transformer hidden dimension", type=int, default=512)
parser.add_argument("--trans-drop", help="Dropout for transformer", default=.1, type=float)
parser.add_argument("--trans-pool", help="Pooling used for Transformer. Either 'avg' or 'concat'", type=str, default="concat")
parser.add_argument("--trans-layers", help="Transformer layers", type=int, default=2)
parser.add_argument("--trans-heads", help="Transformer layersheads", type=int, default=4)
parser.add_argument("--mask", help="Explicit masking in transformer", action='store_true', default=False)

parser.add_argument("--aux-ent", help="Include auxillary entity training task", action='store_true', default=False)
parser.add_argument("--aux-weight", help="Weight for aux loss", type=float, default=1)
parser.add_argument("--aux-ent-smooth", default=.1, type=float)

parser.add_argument("--parallel", action='store_true', default=False)
parser.add_argument("--parallel-drop", default=.3, type=float)

parser.add_argument("--max-qpairs", type=int, default=15)


cmd_args = parser.parse_args()
DEFAULT_CONFIG['MODEL'] = MODEL_CONFIG
DEFAULT_CONFIG['MAX_QPAIRS']= cmd_args.max_qpairs
DEFAULT_CONFIG['PARALLEL'] = cmd_args.parallel
DEFAULT_CONFIG['AUX_ENT'] = cmd_args.aux_ent
DEFAULT_CONFIG['LAMBDA'] = cmd_args.lr_lambda
DEFAULT_CONFIG['USE_TEST'] = not cmd_args.test_on_val
DEFAULT_CONFIG['BATCH_SIZE'] = cmd_args.batch_size
DEFAULT_CONFIG['DATASET'] = cmd_args.dataset.lower()
DEFAULT_CONFIG['DEVICE'] = torch.device(cmd_args.device)
DEFAULT_CONFIG['EMBEDDING_DIM'] = cmd_args.dim
DEFAULT_CONFIG['EPOCHS'] = cmd_args.epochs 
DEFAULT_CONFIG['EVAL_EVERY'] = cmd_args.val_every
DEFAULT_CONFIG['LEARNING_RATE'] = cmd_args.lr
DEFAULT_CONFIG['LABEL_SMOOTHING'] = cmd_args.label_smooth
DEFAULT_CONFIG['AUX_ENT_SMOOTH'] = cmd_args.aux_ent_smooth
DEFAULT_CONFIG['CLEANED_DATASET'] = cmd_args.clean_data
DEFAULT_CONFIG['LR_SCHEDULER'] = cmd_args.lr_decay
DEFAULT_CONFIG['EARLY_STOPPING'] = cmd_args.early_stopping
DEFAULT_CONFIG['ONLY-TRIPS'] = cmd_args.only_trips
DEFAULT_CONFIG['ALPHA'] = cmd_args.alpha
DEFAULT_CONFIG['AUX_WEIGHT'] = cmd_args.aux_weight
DEFAULT_CONFIG['FACT_ENCODER'] = cmd_args.fact_encoder.lower()
DEFAULT_CONFIG['MODEL']['OPN'] = cmd_args.opn
DEFAULT_CONFIG['MODEL']['ENCODER_DROP_1'] = cmd_args.encoder_drop1
DEFAULT_CONFIG['MODEL']['ENCODER_DROP_2'] = cmd_args.encoder_drop2
DEFAULT_CONFIG['MODEL']['T_HIDDEN'] = cmd_args.trans_dim
DEFAULT_CONFIG['MODEL']['GCN_DROP'] = cmd_args.gcn_drop
DEFAULT_CONFIG['MODEL']['QUAL_LAYERS'] = cmd_args.qual_layers
DEFAULT_CONFIG['MODEL']['TRIP_LAYERS'] = cmd_args.trip_layers
DEFAULT_CONFIG['MODEL']['TRANSFORMER_DROP'] = cmd_args.trans_drop
DEFAULT_CONFIG['MODEL']['POOLING'] = cmd_args.trans_pool.lower()
DEFAULT_CONFIG['MODEL']['SRC_MASK'] = cmd_args.mask
DEFAULT_CONFIG['MODEL']['QUAL_COMB'] = cmd_args.qual_comb.lower()
DEFAULT_CONFIG['MODEL']['T_LAYERS'] = cmd_args.trans_layers
DEFAULT_CONFIG['MODEL']['T_N_HEADS'] = cmd_args.trans_heads
DEFAULT_CONFIG['MODEL']['PARALLEL_DROP'] = cmd_args.parallel_drop


def get_data(config):
    """
    Get the data for the specific dataset requested
    """
    data = DataManager.load(config=config)()

    # _data hold indices of entites/relations
    train_data, valid_data, test_data, n_entities, n_relations, _, _ = data.values()

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

    # print("Total Stmts:", len(train_data) + len(valid_data) + len(test_data))
    # qual_stmts = [t for t in train_data + valid_data + test_data if len([i for i in t if i != 0]) > 3]
    # print("Qual%:", len(qual_stmts) / len(train_data + valid_data + test_data) * 100)

    # Inverses are added here in `get_alternative_graph_repr` !!!
    # -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
    # -> edge_type (n) array with [relation] corresponding to sub, obj above
    # -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals
    if config['USE_TEST']:
        train_data_gcn = DataManager.get_alternative_graph_repr(train_data + valid_data, config)
    else:
        train_data_gcn = DataManager.get_alternative_graph_repr(train_data, config)

    # from collections import Counter

    # num_quals = train_data_gcn['quals'][2].shape[0] // 2
    # num_rows = train_data_gcn['edge_type'].shape[0] // 2

    # row_quals = [0 for _ in range(num_rows)]

    # for x in train_data_gcn['quals'][2][:num_quals]:
    #     row_quals[x] += 1
    
    # qual_count = Counter(row_quals)
    
    # for i, c in qual_count.items():
    #     print(f"{i}: {round(c / num_rows * 100, 2)}")

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

    print("\nConfig\n------", flush=True)
    print(config, flush=True)
    print("\n", flush=True)

    ## Specified Dataset splits
    train_data_gcn, train_data, valid_data, test_data, ent_excluded_from_corr = get_data(config)

    ## Model
    model = HypRelModel(train_data_gcn, config)
    model.to(config['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    if config['USE_TEST']:
        ev_val_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        tr_data     = {'train': combine(train_data, valid_data), 'valid': ev_val_data['eval']}
    else:
        ev_val_data = {'index': combine(train_data, test_data), 'eval': combine(valid_data)}
        tr_data     = {'train': combine(train_data), 'valid': ev_val_data['eval']}


    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3), partial(hits_at, k=5), partial(hits_at, k=10)]

    sampler = MultiClassSampler(
                    data= tr_data['train'], 
                    n_entities=config['NUM_ENTITIES'], 
                    n_rel=config['NUM_RELATIONS'], 
                    lbl_smooth=config['LABEL_SMOOTHING'], 
                    bs=config['BATCH_SIZE'], 
                    aux_ent=config['AUX_ENT'], 
                    aux_ent_smooth=config['AUX_ENT_SMOOTH'], 
                    max_pairs = config['MAX_QPAIRS'] - 3 
            )

    evaluation_valid = EvaluationBenchGNNMultiClass(
                            ev_val_data, model, 
                            bs=config['BATCH_SIZE'], 
                            metrics=eval_metrics,
                            filtered=True, 
                            n_ents=config['NUM_ENTITIES'],
                            excluding_entities=ent_excluded_from_corr,
                            positions=config.get('CORRUPTION_POSITIONS', None), 
                            config=config
                        )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95) if config['LR_SCHEDULER'] else None
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: config['LAMBDA']**e) if config['LR_SCHEDULER'] else None

    # First 10% are warmup
    warmup_sched = warmup.LinearWarmup(optimizer, warmup_period=int(config['EPOCHS']*.1)) if cmd_args.warmup else None

    # The args to use if we're training w default stuff
    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "model": model,
        "device": config['DEVICE'],
        "data_fn": sampler.reset,
        "val_testbench": evaluation_valid.run,
        "eval_every": config['EVAL_EVERY'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "early_stopping": config['EARLY_STOPPING'],
        "scheduler": lr_scheduler,
        "warmup": warmup_sched,
        "aux_ent": config['AUX_ENT'],
        "aux_weight": config['AUX_WEIGHT'],
        "max_qpairs": config['MAX_QPAIRS'] - 3
    }

    print(f"Dataset has {config['NUM_ENTITIES']} entities and {config['NUM_RELATIONS']} relations!\n", flush=True)

    training_loop_gcn(**args)
    save_model(model, config['DATASET'])
    

if __name__ == "__main__":
    main()
