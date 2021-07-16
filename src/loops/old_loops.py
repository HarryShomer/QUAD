"""
Original source can be found here -> https://github.com/migalkin/StarE/blob/master/loops/loops.py
"""

from tqdm.autonotebook import tqdm
from typing import Callable

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
        aux_train: bool = False,
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

                if aux_train:
                    triples, obj_labels, qual_stmts, qual_labels = batch
                else:
                    triples, obj_labels = batch

                # Standard batch info
                sub, rel, quals = triples[:, 0], triples[:, 1], triples[:, 2:]
                _sub = torch.tensor(sub, dtype=torch.long, device=device)
                _rel = torch.tensor(rel, dtype=torch.long, device=device)
                _quals = torch.tensor(quals, dtype=torch.long, device=device)
                _obj_labels = torch.tensor(obj_labels, dtype=torch.float, device=device)

                if aux_train:
                    sub_q, rel_q, obj, qual_ix, quals_q = qual_stmts[:, 0], qual_stmts[:, 1], qual_stmts[:, 2], qual_stmts[:, 3], qual_stmts[:, 4:16]

                    _sub_q = torch.tensor(sub_q.astype(int), dtype=torch.long, device=device)
                    _rel_q = torch.tensor(rel_q.astype(int), dtype=torch.long, device=device)
                    _obj = torch.tensor(obj.astype(int), dtype=torch.long, device=device)

                    _quals_q = torch.tensor(quals_q.astype(int), dtype=torch.long, device=device)

                    # Position in sequence of qual entity. +3 is to account for base trip indices
                    _qual_ix = torch.tensor(qual_ix.astype(int), dtype=torch.long, device=device) + 3

                    _qual_labels = torch.tensor(qual_labels, dtype=torch.float, device=device)
                

                # with open("wd50k_100_trip_batch.txt", "w") as f:
                #     for i in range(len(_sub)):
                #         f.write(f"{_sub[i] : <8}  |  {_rel[i] : <8}  | {', '.join(str(e) for e in _quals[i].tolist()) : <60}  | {[j[0] for j in _obj_labels[i].nonzero().tolist()]}")
                #         f.write("\n")

                # with open("wd50k_100_qual_batch.txt", "w") as f:
                #     for i in range(len(_sub_q)):
                #         f.write(f"{_sub_q[i] : <8}  |  {_rel_q[i] : <8}  |  {_obj[i] : < 8}  |  {', '.join(str(e) for e in _quals_q[i].tolist()) : <60}  | {_qual_ix[i] : < 5}  |  {[j[0] for j in _qual_labels[i].nonzero().tolist()]}")
                #         f.write("\n")
                
                
                if aux_train:
                    obj_preds, qual_preds = model(_sub, _rel, _quals, _sub_q, _rel_q, _obj, _quals_q, _qual_ix)
                    loss =  model.loss(obj_preds, _obj_labels) + aux_weight * model.loss(qual_preds, _qual_labels)
                else:
                    pred = model(_sub, _rel, _quals)
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
