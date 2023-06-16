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
from bopt.bilevel.classification_reversible_inner import reversible_inner
from bopt.bilevel.utils import snap_to_initial_tokenizer, extract_tokenizer
from bopt.modeling.classifier import ClassifierOutput
from bopt.training import TrainingState
from bopt.bilevel.classification_eval_loop import eval_classification
from bopt.training.utils import load_forever




def train_classification_inner(setup: ClassificationBilevelTrainingSetup, outer_step: int, warmup=False):
    if setup.args.bilevel_optimization_scheme == "reversible-learning":
        return reversible_inner(setup, outer_step, warmup=warmup)
    if setup.args.bilevel_optimization_scheme == "ift":
        with open(os.path.join(setup.args.output_directory, f"log-inner-of-{outer_step if not warmup else 'warmup'}.json"), "wt") as f:
            pass

    step = 0
    windowed_loss = []
    windowed_loss_avg = math.inf
    grad_norm = math.inf

    # for unroll mode
    cumloss = 0
    params = None
    one_step_params = None
    one_step_loss = None
    init_tokenizer_params = None
    model = None

    inner_optimizer, _, inner_scheduler, _ = setup.optimizer_builder()
    print(id(inner_optimizer))
    setup.inner_optimizer = inner_optimizer
    setup.inner_scheduler = inner_scheduler
    with higher.innerloop_ctx(setup.classifier, setup.inner_optimizer) as (fmodel, diffopt):
        bar = tqdm(enumerate(load_forever(setup.train_inner_dataloader)),total=(setup.args.train_steps_inner if not warmup else setup.args.train_steps_warmup) * setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner)
        for raw_step, (epoch, batch) in bar:
            # training state
            bar.set_description_str(f"Inner Epoch={epoch} Step={step} InnerLoss={sum(windowed_loss) / len(windowed_loss) if len(windowed_loss) else windowed_loss_avg:.2f}({len(windowed_loss) * setup.args.gpu_batch_size_inner:>4d} exs)")
            state = TrainingState(step, epoch, bar.format_dict['elapsed'])

            if raw_step >= ((setup.args.train_steps_inner if not warmup else setup.args.train_steps_warmup) * (setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner)) or grad_norm <= setup.args.inner_threshold:
                break
            # training
            ids, sentences, labels = batch

            # run model
            if setup.args.bilevel_optimization_scheme == "ift" or warmup:
                # ift and warmpup iterations don't require diff-through-opt, so just do normal backprop
                output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_inner")
                loss = output.task_loss * (setup.args.gpu_batch_size_inner / setup.args.train_batch_size_inner)
                loss.backward()
                loss = loss.item()
            elif setup.args.bilevel_optimization_scheme == "unroll":
                # unroll requires diff-through-opt so do that instead
                # code.interact(local=locals())

                output: ClassifierOutput = fmodel(setup, ids, sentences, labels, "train_inner")#, snap_to_initial_tokenizer(setup.classifier.tokenizer_parameter_mask, fmodel.init_fast_params, fmodel.fast_params))
                loss = output.task_loss * (setup.args.gpu_batch_size_inner / setup.args.train_batch_size_inner)
                cumloss += loss
                loss = loss.item()

            else:
                raise ValueError(f"{setup.args.bilevel_optimization_scheme} is unknown bilevel scheme")

            # maybe step optimizer
            if (raw_step + 1) %  (setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner) == 0:
                if setup.args.bilevel_optimization_scheme == "ift" or warmup:
                    grad_norm = math.sqrt(sum(
                        p.grad.norm() ** 2 for p in setup.classifier.model.parameters() if p.grad is not None).item())
                    setup.inner_optimizer.step()
                    setup.classifier.model.zero_grad()
                elif setup.args.bilevel_optimization_scheme == "unroll":
                    grad_norm = math.inf #TODO: implement grad norm tracking for unroll (actually maybe its unnecessary since with unroll we usually only do a few steps anyways)
                    # print("before step")
                    # code.interact(local=locals())
                    diffopt.step(cumloss)
                    # print("after step")
                    if one_step_loss is None:
                        one_step_loss = cumloss.item()
                    cumloss = 0
                else:
                    raise ValueError(f"{setup.args.bilevel_optimization_scheme} is unknown bilevel scheme")
                step += 1

            # maybe step scheduler, only for ift currently
            windowed_loss.append(loss)
            if len(windowed_loss) >= (setup.args.lr_adjustment_window_size // setup.args.gpu_batch_size_inner):
                windowed_loss_avg = sum(windowed_loss) / len(windowed_loss)
                windowed_loss = []
                if setup.args.bilevel_optimization_scheme == "ift":
                    if setup.args.annealing > 0 and setup.args.annealing_start_steps <= step <= setup.args.annealing_end_steps:
                        setup.inner_scheduler._reset()
                        # reset so never step down due to increasing entropy loss factor, and continue with new baseline after
                        # the annealing factor is maxed out
                    setup.inner_scheduler.step(windowed_loss_avg)
            # code.interact(local=locals())
        if setup.args.bilevel_optimization_scheme == "unroll" and not warmup:
            params = fmodel.fast_params
            one_step_params = snap_to_initial_tokenizer(setup.classifier.tokenizer_parameter_mask, fmodel.init_fast_params, fmodel.parameters(time=1))
            init_tokenizer_params = extract_tokenizer(setup.classifier.tokenizer_parameter_mask, fmodel.init_fast_params)
            model=fmodel
            # for time in range(setup.args.train_steps_inner):
            #     for param in extract_tokenizer(setup.classifier.tokenizer_parameter_mask, fmodel.parameters(time=time)):
            #         param.requires_grad = True

    if setup.args.bilevel_optimization_scheme == "ift":
        setup.classifier.eval()
        with torch.no_grad():
            eval_metrics = eval_classification(setup, state, save_prediction=False)
        setup.classifier.train()
        # save log
        with open(os.path.join(setup.args.output_directory, f"log-inner-of-{outer_step}.json"), "at") as f:
            logline = {
                "step": step,
                "epoch": epoch,
                "elapsed": state.elapsed,
                "train_loss": windowed_loss_avg,
                "model_lr": setup.inner_optimizer.named_param_groups["model_decay"]["lr"]
            }
            logline.update(eval_metrics)
            print(json.dumps(logline), file=f)
            print("\t" + str(logline))
    return InnerLoopOutput(params=params, one_step_params=one_step_params, one_step_loss=one_step_loss, init_tokenizer_params=init_tokenizer_params, fmodel=model)

