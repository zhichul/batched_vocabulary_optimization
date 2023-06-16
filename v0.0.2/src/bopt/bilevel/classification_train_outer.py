import code
import json
import math
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from bopt.bilevel import ClassificationBilevelTrainingSetup
from bopt.bilevel.classification_loss_inner import train_loss_inner
from bopt.bilevel.classification_train_inner import train_classification_inner, InnerLoopOutput
from bopt.bilevel.classification_trajectory_inner import train_trajectory_inner
from bopt.bilevel.ift import hyper_step
from bopt.bilevel.utils import snap_to_initial_model
from bopt.modeling.classifier import ClassifierOutput
from bopt.training import TrainingState
from bopt.bilevel.classification_eval_loop import eval_classification
from bopt.training.saving import save_classification_checkpoint
from bopt.training.utils import load_forever
from experiments.utils.optimization import set_grad
from experiments.utils.seeding import seed


def maybe_random_restart(setup, override):
    if not setup.args.fix_transformer_initialization:
        setup.classifier.model.init_weights()
    else:
        for p, init_p in zip(setup.classifier.model.parameters(), override):
            p.data = init_p.detach().clone()


def train_classification_outer(setup: ClassificationBilevelTrainingSetup):
    with open(os.path.join(setup.args.output_directory, "log.json"), "wt") as f:
        pass
    step = 0
    windowed_loss = []
    best_dev_acc = -1
    windowed_loss_avg = math.inf
    igrad = None
    dgrad = None
    tgrad = None
    initial_model_parameters = [param.data.detach().clone() for param in setup.classifier.model.parameters()]

    bar = tqdm(enumerate(load_forever(setup.train_outer_dataloader)),total=setup.args.train_steps * setup.args.train_batch_size // setup.args.gpu_batch_size)

    for raw_step, (epoch, batch) in bar:
        # training state
        bar.set_description_str(f"Outer Epoch={epoch} Step={step} OuterLoss={sum(windowed_loss) / len(windowed_loss) if len(windowed_loss) else windowed_loss_avg:.2f}({len(windowed_loss) * setup.args.gpu_batch_size:>4d} exs) Dgrad={dgrad} Igrad={igrad} grad={tgrad}")
        state = TrainingState(step, epoch, bar.format_dict['elapsed'])
        seed(setup.args.seed)
        # maybe evaluate
        if ((raw_step) % (setup.args.train_batch_size // setup.args.gpu_batch_size) == 0) and (step % setup.args.eval_steps == 0):
            eval_metrics = defaultdict(float)
            for i in range(setup.args.eval_random_restarts):
                maybe_random_restart(setup, initial_model_parameters)
                setup.classifier.model.zero_grad()

                # warmup inner problem
                set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=False)
                # train_classification_inner(setup, step, warmup=True)
                #
                # # train trajectory steps
                # warmup_steps = setup.args.train_steps_warmup
                # setup.args.train_steps_warmup = setup.args.train_trajectory_inner + setup.args.train_steps_inner - 1
                # train_classification_inner(setup, step, warmup=True)
                # setup.args.train_steps_warmup = warmup_steps

                warmup_steps = setup.args.train_steps_warmup
                setup.args.train_steps_warmup = setup.args.eval_train_steps
                train_classification_inner(setup, step, warmup=True)
                setup.args.train_steps_warmup = warmup_steps
                set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=True)

                setup.classifier.eval()
                with torch.no_grad():
                    eval_metrics_i = eval_classification(setup, state)
                setup.classifier.train()
                for metric in eval_metrics_i:
                    eval_metrics[metric] += eval_metrics_i[metric] / setup.args.eval_random_restarts
                print(eval_metrics_i)
            # early stopping
            if eval_metrics["dev_accuracy"] > best_dev_acc:
                best_dev_acc = eval_metrics["dev_accuracy"]
                save_classification_checkpoint(setup.args.output_directory, "checkpoint-early-stopping", state, setup.classifier, optimizer=setup.outer_optimizer)

            # save log
            with open(os.path.join(setup.args.output_directory, "log.json"), "at") as f:
                logline = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed": state.elapsed,
                    "train_loss": windowed_loss_avg,
                    "tokenizer_lr": setup.outer_optimizer.named_param_groups["tokenizer"]["lr"],
                    "tokenizer_direct_gradient": dgrad,
                    "tokenizer_indirect_gradient": igrad,
                    "tokenizer_gradient": tgrad,
                }
                logline.update(eval_metrics)
                print(json.dumps(logline), file=f)
                print(logline)


        if step >= setup.args.train_steps:
            break

        if (raw_step) % (setup.args.train_batch_size // setup.args.gpu_batch_size) == 0:
            print()
            task_loss = 0
            tokenizer_params = list(setup.classifier.parameters())[-1].detach()
            if setup.args.bilevel_optimization_scheme == "ift":
                # run inner loop with outer parameter fixed
                set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=False)
                train_classification_inner(setup, step)
                set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=True)
            elif setup.args.bilevel_optimization_scheme == "unroll" or setup.args.bilevel_optimization_scheme == "reversible-learning":
                cumgrad = 0
                tokenizer_params_ref = next(setup.classifier.input_tokenizer.parameters())
                for rr in tqdm(range(setup.args.random_restarts), desc="Random restarts"):
                    # code.interact(local=locals())
                    # reset model back to where the trajectory started
                    maybe_random_restart(setup, initial_model_parameters)
                    setup.classifier.model.zero_grad()

                    # warmup inner problem
                    set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=False)
                    train_classification_inner(setup, step, warmup=True)
                    set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=True)

                    if any(p.grad is not None and p.grad.norm() > 0 for p in setup.classifier.model.parameters()):
                        code.interact(local=locals())
                    if not (tokenizer_params_ref.data == tokenizer_params).all():
                        code.interact(local=locals())
                        raise AssertionError
                    # if not (tokenizer_params == 0).all():
                    #     code.interact(local=locals())
                    # clf_params = list(setup.classifier.parameters())
                    # print(clf_params[0].norm().item())
                    # for np, p in zip(snap_to_initial_model(setup.classifier.tokenizer_parameter_mask, init_params, setup.classifier.parameters()), clf_params):
                    #     p.data = np.data
                    # print(clf_params[0].norm())
                    # if raw_step > 0 and (not (init_params[0] ==clf_params[0]).all() or (init_params[-1] == clf_params[-1]).all()):
                    #     code.interact(local=locals())
                    task_loss += train_trajectory_inner(setup, step)

                    rr_grad = tokenizer_params_ref.grad - cumgrad
                    cumgrad = tokenizer_params_ref.grad.detach().clone()
                    torch.save(rr_grad, os.path.join(setup.args.output_directory, f"step-{step}-gradient-{rr}.pt"))

                igrad = math.sqrt(sum((param.grad ** 2).sum().item() for param in setup.classifier.input_tokenizer.parameters() if param.grad is not None))

        # training
        ids, sentences, labels = batch
        # run model
        output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_outer")

        if setup.args.bilevel_optimization_scheme == "ift":

            task_loss = output.task_loss

            if setup.args.indirect_gradient_only: set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=False)
            (task_loss * (setup.args.gpu_batch_size / setup.args.train_batch_size)).backward(retain_graph=True)
            if setup.args.indirect_gradient_only: set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=True)

        # compute other losses
        reg_loss = 0
        if setup.args.L1 > 0: reg_loss += setup.args.L1 * output.regularizers.l1
        if setup.args.annealing > 0:
            reg_loss += setup.annealing_scheduler(step) * output.regularizers.entropy

        # do backward
        (reg_loss * (setup.args.gpu_batch_size / setup.args.train_batch_size)).backward()

        if setup.args.bilevel_optimization_scheme == "unroll" or setup.args.bilevel_optimization_scheme == "reversible-learning":
            tgrad = math.sqrt(sum((param.grad ** 2).sum().item() for param in setup.classifier.input_tokenizer.parameters() if param.grad is not None))

        # maybe step optimizer
        if (raw_step + 1) %  (setup.args.train_batch_size // setup.args.gpu_batch_size) == 0:
            if setup.args.bilevel_optimization_scheme == "ift":
                # hypergradient computation
                hyper_meta = hyper_step(setup)
                igrad = hyper_meta["indirect_gradient_norm"]
                dgrad = hyper_meta["direct_gradient_norm"]
                tgrad = hyper_meta["total_gradient_norm"]
            setup.outer_optimizer.step()
            setup.classifier.model.zero_grad()
            setup.classifier.input_tokenizer.zero_grad()
            setup.classifier.input_tokenizer.clamp_weights()
            step += 1

            # save the learned vocab
            checkpointdir = os.path.join(setup.args.output_directory, f"checkpoint-{step}")
            os.makedirs(checkpointdir, exist_ok=True)
            setup.classifier.input_tokenizer.save_to_folder(checkpointdir)


        # maybe step scheduler
        if setup.args.bilevel_optimization_scheme == "ift": windowed_loss.append(task_loss.item() + reg_loss.item())
        elif setup.args.bilevel_optimization_scheme == "unroll" or setup.args.bilevel_optimization_scheme == "reversible-learning": windowed_loss.append(task_loss + reg_loss.item())
        if len(windowed_loss) >= (setup.args.lr_adjustment_window_size // setup.args.gpu_batch_size):
            windowed_loss_avg = sum(windowed_loss) / len(windowed_loss)
            windowed_loss = []
            if setup.args.annealing > 0 and setup.args.annealing_start_steps <= step <= setup.args.annealing_end_steps:
                setup.outer_scheduler._reset()
                # reset so never step down due to increasing entropy loss factor, and continue with new baseline after
                # the annealing factor is maxed out
            setup.outer_scheduler.step(windowed_loss_avg)

    save_classification_checkpoint(setup.args.output_directory, "checkpoint-final", state, setup.classifier, optimizer=setup.outer_optimizer)