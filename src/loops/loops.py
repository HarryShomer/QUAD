"""
Original source can be found here -> https://github.com/migalkin/StarE/blob/master/loops/loops.py
"""
import sys
from typing import Callable
from tqdm.autonotebook import tqdm

from utils.utils_mytorch import *
from .corruption import Corruption


def training_loop_gcn(
        epochs: int,
        data: dict,
        opt: torch.optim,
        model: Callable,
        device: torch.device = torch.device('cpu'),
        data_fn: Callable = SimplestSampler,
        val_testbench: Callable = default_eval,
        eval_every: int = 1,
        grad_clipping: bool = True,
        early_stopping: int = 3,
        scheduler: Callable = None,
        aux_ent: bool = False,
        aux_rel: bool = False,
        aux_weight: float = 0.5
    ):
    """
    A fn which can be used to train a language model.

    The model doesn't need to be an nn.Module,
        but have an eval (optional), a train and a predict function.

    Data should be a dict like so:
        {"train":{"x":np.arr, "y":np.arr}, "val":{"x":np.arr, "y":np.arr} }

    model must return both loss and y_pred

    :param epochs: integer number of epochs
    :param data: a dictionary which looks like {'train': train data}
    :param opt: torch optimizer
    :param model: a fn which is/can call forward of a nn module
    :param device: torch.device for making tensors
    :param data_fn: Something that can make iterators out of training data (think mytorch samplers)
    :param val_testbench: Function call to see generate all negs for all pos and get metrics in valid set
    :param eval_every: int which dictates after how many epochs should run testbenches
    :param scheduler: LR scheduler (torch.optim.lr_scheduler)
    :param aux_train: Whether to include auxillary training task
    """
    train_loss = []
    train_acc = []
    valid_acc = []
    valid_mrr = []
    valid_mr = []
    valid_hits_3, valid_hits_5, valid_hits_10 = [], [], []
    train_acc_bnchmk = []
    train_mrr_bnchmk = []
    train_hits_3_bnchmk, train_hits_5_bnchmk, train_hits_10_bnchmk = [], [], []

    # Epoch level
    for e in range(1, epochs + 1):
        per_epoch_loss = []

        with Timer() as timer:

            # Make data
            trn_dl = data_fn(data['train'])
            model.train()

            for batch in tqdm(trn_dl):
                opt.zero_grad()

                if not any([aux_ent, aux_rel]):
                    triples, obj_labels = batch
                elif aux_ent and not aux_rel:
                    triples, obj_labels, aux_ent_stmts, aux_ent_labels = batch
                elif aux_rel and not aux_ent:
                    sys.exit("TODO: Implement aux_rel and no aux_ent")
                else:
                    triples, obj_labels, aux_ent_stmts, aux_ent_labels, aux_rel_stmts, aux_rel_labels = batch

                # Standard batch data
                _sub, _rel, _quals, _obj_labels = process_triplets(triples, obj_labels, device)

                # Do same if aux ent and rel are specified
                if aux_ent and len(aux_ent_stmts.shape) > 1:
                    aux_ent_dict, _aux_ent_labels = process_aux_ent(aux_ent_stmts, aux_ent_labels, device)
                if aux_rel and len(aux_rel_stmts.shape) > 1:
                    aux_rel_dict, _aux_rel_labels = process_aux_rel(aux_rel_stmts, aux_rel_labels, device)


                # for i in range(len(aux_rel_dict['base_sub_ix'])):
                #     if aux_rel_dict['base_sub_ix'][i].item() == 15395:
                #         print(aux_rel_dict['base_sub_ix'][i].item(), end=" ")
                #         print(aux_rel_dict['base_rel_ix'][i].item(), end=" ")
                #         print(aux_rel_dict['base_obj_ix'][i].item(), end=" ")
                #         print([j.item() for j in aux_rel_dict['quals'][i]], end=" ")
                #         print(aux_rel_dict['mask'][i].item())
                #         print(aux_rel_labels[i].argmax())
                # exit()

                    
                # If no quals and aux_ent = True then dim of `aux_ent_stmts = 1`
                if aux_ent and not aux_rel and len(aux_ent_stmts.shape) > 1:
                    obj_preds, aux_preds, _ = model(_sub, _rel, _quals, aux_ent=aux_ent_dict)
                    loss = model.loss(obj_preds, _obj_labels) + aux_weight * model.loss(aux_preds, _aux_ent_labels)
                
                elif aux_rel and not aux_ent and len(aux_rel_stmts.shape) > 1:
                    obj_preds, _, aux_preds = model(_sub, _rel, _quals, aux_rel=aux_rel_dict)
                    loss = model.loss(obj_preds, _obj_labels) + aux_weight * model.loss(aux_preds, _aux_rel_labels)
                
                elif aux_rel and aux_ent and len(aux_ent_stmts.shape) > 1 and len(aux_rel_stmts.shape) > 1:
                    obj_preds, aux_ent_preds, aux_rel_preds = model(_sub, _rel, _quals, aux_ent=aux_ent_dict, aux_rel=aux_rel_dict)
                    # .Take mean of aux loss so loss isn't too high
                    a = model.loss(aux_ent_preds, _aux_ent_labels)
                    b = model.loss(aux_rel_preds, _aux_rel_labels)

                    # print(f"Entity Loss: {a}  | Relation Loss: {b}")

                    loss = model.loss(obj_preds, _obj_labels) + .5 * aux_weight * (a + b)
                
                else:
                    pred, _, _ = model(_sub, _rel, _quals)
                    loss = model.loss(pred, _obj_labels)
                
                per_epoch_loss.append(loss.item())    
                loss.backward()

                if grad_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                opt.step()


        print(f"[Epoch: {e} ] Loss: {np.mean(per_epoch_loss)}", flush=True)
        train_loss.append(np.mean(per_epoch_loss))

        # Detailed metrics every n epochs
        if e % eval_every == 0 and e >= 1:
            with torch.no_grad():
                summary_val = val_testbench()
                per_epoch_vl_acc = summary_val['metrics']['hits_at 1']
                per_epoch_vl_mrr = summary_val['metrics']['mrr']
                per_epoch_vl_mr = summary_val['metrics']['mr']
                per_epoch_vl_hits_3 = summary_val['metrics']['hits_at 3']
                per_epoch_vl_hits_5 = summary_val['metrics']['hits_at 5']
                per_epoch_vl_hits_10 = summary_val['metrics']['hits_at 10']

                valid_acc.append(per_epoch_vl_acc)
                valid_mrr.append(per_epoch_vl_mrr)
                valid_mr.append(per_epoch_vl_mr)
                valid_hits_3.append(per_epoch_vl_hits_3)
                valid_hits_5.append(per_epoch_vl_hits_5)
                valid_hits_10.append(per_epoch_vl_hits_10)

                print("Epoch: %(epo)03d | Loss: %(loss).5f | "
                      "Vl_c: %(vlacc)0.5f | Vl_mrr: %(vlmrr)0.5f | Vl_mr: %(vlmr)0.5f | "
                      "Vl_h3: %(vlh3)0.5f | Vl_h5: %(vlh5)0.5f | Vl_h10: %(vlh10)0.5f | "
                      "time_trn: %(time).3f min"
                      % {'epo': e,
                         'loss': float(np.mean(per_epoch_loss)),
                         'vlacc': float(per_epoch_vl_acc),
                         'vlmrr': float(per_epoch_vl_mrr),
                         'vlmr': float(per_epoch_vl_mr),
                         'vlh3': float(per_epoch_vl_hits_3),
                         'vlh5': float(per_epoch_vl_hits_5),
                         'vlh10': float(per_epoch_vl_hits_10),
                         'time': timer.interval / 60.0
                         }, 
                         flush=True)

                ## Early stopping
                ## When not improve in last n validation stop
                if early_stopping is not None and len(valid_mrr) >= early_stopping and np.argmax(valid_mrr[-early_stopping:]) == 0:
                    print("Perforamance has not improved! Stopping now!")
                    break
        else:
            # No test benches this time around
            print("Epoch: %(epo)03d | Loss: %(loss).5f | Time_Train: %(time).3f min" 
                  % {'epo': e, 'loss': float(np.mean(per_epoch_loss)), 'time': timer.interval / 60.0}, flush=True)
    
        if scheduler is not None:
            scheduler.step()

    return train_acc, train_loss, \
           train_acc_bnchmk, train_mrr_bnchmk, \
           train_hits_3_bnchmk, train_hits_5_bnchmk, train_hits_10_bnchmk, \
           valid_acc, valid_mrr, \
           valid_hits_3, valid_hits_5, valid_hits_10


def process_triplets(triples, obj_labels, device):
    """
    Extract and convert to torch objects
    """
    sub, rel, quals = triples[:, 0], triples[:, 1], triples[:, 2:]

    _sub        = torch.tensor(sub, dtype=torch.long, device=device)
    _rel        = torch.tensor(rel, dtype=torch.long, device=device)
    _quals      = torch.tensor(quals, dtype=torch.long, device=device)
    _obj_labels = torch.tensor(obj_labels, dtype=torch.float, device=device)

    return _sub, _rel, _quals, _obj_labels



def process_aux_ent(aux_ent_stmts, aux_ent_labels, device):
    """
    Extract information from aux ent batch, convert to tensors, and place in dict
    """
    sub_q, rel_q, obj, qual_ix, quals_q = aux_ent_stmts[:, 0], aux_ent_stmts[:, 1], aux_ent_stmts[:, 2], aux_ent_stmts[:, 3], aux_ent_stmts[:, 4:16]

    aux_ent_dict = {
        "base_sub_ix": torch.tensor(sub_q.astype(int), dtype=torch.long, device=device),
        "base_rel_ix": torch.tensor(rel_q.astype(int), dtype=torch.long, device=device),
        "base_obj_ix": torch.tensor(obj.astype(int), dtype=torch.long, device=device),
        "quals": torch.tensor(quals_q.astype(int), dtype=torch.long, device=device),

        # Position in sequence of qual entity. +3 is to account for base trip indices
        "mask": torch.tensor(qual_ix.astype(int), dtype=torch.long, device=device) + 3
    }

    _aux_ent_labels = torch.tensor(aux_ent_labels, dtype=torch.float, device=device)

    return aux_ent_dict, _aux_ent_labels


def process_aux_rel(aux_rel_stmts, aux_rel_labels, device):
    """
    Extract information from aux rel batch, convert to tensors, and place in dict
    """
    sub_q, rel_q, obj, rel_ix, quals_q = aux_rel_stmts[:, 0], aux_rel_stmts[:, 1], aux_rel_stmts[:, 2], aux_rel_stmts[:, 3], aux_rel_stmts[:, 4:]

    aux_rel_dict = {
        "base_sub_ix": torch.tensor(sub_q.astype(int), dtype=torch.long, device=device),
        "base_rel_ix": torch.tensor(rel_q.astype(int), dtype=torch.long, device=device),
        "base_obj_ix": torch.tensor(obj.astype(int), dtype=torch.long, device=device),
        "quals": torch.tensor(quals_q.astype(int), dtype=torch.long, device=device),

        # No need for +3. Handled in sampler
        "mask": torch.tensor(rel_ix.astype(int), dtype=torch.long, device=device)
    }

    _aux_rel_labels = torch.tensor(aux_rel_labels, dtype=torch.float, device=device)

    return aux_rel_dict, _aux_rel_labels