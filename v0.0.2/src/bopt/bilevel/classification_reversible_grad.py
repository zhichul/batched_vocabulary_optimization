import code
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Any

import higher
import torch
from tqdm import tqdm

from bopt.bilevel import ClassificationBilevelTrainingSetup, InnerLoopOutput
from bopt.bilevel.utils import snap_to_initial_tokenizer, extract_tokenizer
from bopt.modeling.classifier import ClassifierOutput
from bopt.training import TrainingState
from bopt.bilevel.classification_eval_loop import eval_classification
from bopt.training.utils import load_forever
from experiments.utils.reversible_operations import f_div_rat, to_rational, f_cast, f_mul_alpha, f_add, f_sub, f_uncast, \
    f_set


def accumulate_gradient(setup: ClassificationBilevelTrainingSetup):
    num_batches = max(min(math.ceil(setup.args.train_batch_size/setup.args.gpu_batch_size), len(setup.train_outer_dataloader)), 1)
    losses = 0
    for i, batch in enumerate(setup.train_outer_dataloader):
        if i >= num_batches: break  # only do one training batch max
        # training
        ids, sentences, labels = batch

        # run model
        output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_outer")

        # compute loss (this will accumulate both direct tokenizer gradients and the initial bit of indirect gradients (accumulated into the last iteration parameters)
        (output.task_loss / num_batches / setup.args.train_trajectory_inner / setup.args.random_restarts).backward(retain_graph=i < (num_batches-1))
        losses += output.task_loss.item() / num_batches / setup.args.train_trajectory_inner / setup.args.random_restarts
    return losses

def reversible_grad(setup: ClassificationBilevelTrainingSetup, rollout:InnerLoopOutput):
    step = len(rollout.buffers_first) - 1 # init
    w_t = f_cast(setup.classifier.model.parameters()) # init
    v_tm1 = rollout.last_step # this is already cast
    w_tm1 = f_sub(w_t, f_mul_alpha(v_tm1, alpha=setup.args.task_model_learning_rate))
    f_set(setup.classifier.parameters(), f_uncast(w_tm1))
    dv = tuple(torch.zeros_like(p.data) for p in setup.classifier.model.parameters())
    dtok = tuple(torch.zeros_like(p.data) for p in setup.classifier.input_tokenizer.parameters())

    outer_loss = accumulate_gradient(setup)
    dw = [p.grad.detach().clone() for p in setup.classifier.model.parameters()]
    bar = tqdm(reversed(list(enumerate(rollout.batch_ids))), total=len(rollout.batch_ids))
    n, d = to_rational(setup.args.momentum_coefficient, d=setup.args.momentum_coefficient_precision)
    for step, batch_id in bar:
        # zero_grad
        g_tm1 = [torch.zeros_like(p.data) for p in setup.classifier.model.parameters()]
        setup.classifier.model.zero_grad()
        for ids in batch_id:
            ids, sentences, labels = setup.train_inner_dataloader.collate_fn([setup.train_inner_dataloader.dataset[id] for id in ids])
            output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_inner")

            # reverse sgd
            loss = output.task_loss * (setup.args.gpu_batch_size_inner / setup.args.train_batch_size_inner)
            for i, pg in enumerate(torch.autograd.grad(loss, setup.classifier.model.parameters(), create_graph=True, retain_graph=True)):
                g_tm1[i] = g_tm1[i] + pg # line 7
            # loss.backward()
            # maybe reverse optimizer
        v_tm2_numerator = f_add(v_tm1, f_mul_alpha(f_cast(g_tm1), alpha=(1 - setup.args.momentum_coefficient)))
        v_tm2 = f_div_rat(v_tm2_numerator, n, d, rollout.buffers_first[step])
        dv = torch._foreach_add(dv, dw, alpha=setup.args.task_model_learning_rate) # line 9
        sodw = torch.autograd.grad(g_tm1, setup.classifier.model.parameters(), grad_outputs=dv, retain_graph=True) # line 11
        torch._foreach_add_(dw, sodw, alpha=-(1-setup.args.momentum_coefficient)) # line 11
        sodtok = torch.autograd.grad(g_tm1, setup.classifier.input_tokenizer.parameters(), grad_outputs=dv, retain_graph=False) # line 12
        torch._foreach_add_(dtok, sodtok, alpha=-(1-setup.args.momentum_coefficient))
        dv = torch._foreach_mul(dv, setup.args.momentum_coefficient)
        # lcs = locals()
        #
        # lcs["f_cast"] = f_cast
        # lcs["f_uncast"] = f_uncast
        # lcs["g_tm1"] = f_uncast(f_cast(g_tm1))
        # code.interact(local=lcs)

        if step > 0:
            v_tm1 = v_tm2
            w_tm1 = f_sub(w_tm1, f_mul_alpha(v_tm2, alpha=setup.args.task_model_learning_rate))  # line 6
            f_set(setup.classifier.model.parameters(), f_uncast(w_tm1))


    if not step == 0: code.interact(local=locals())
    for p, pgrad in zip(setup.classifier.input_tokenizer.parameters(), dtok):
        if p.grad is None:
            p.grad = pgrad
        else:
            p.grad += pgrad
    return outer_loss