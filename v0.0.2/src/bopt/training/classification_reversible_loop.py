import code
import gc
import json
import math
import os
import time
from collections import deque

import torch
from tqdm import tqdm

from bopt.training.classification_eval_loop import eval_classification
from bopt.modeling.classifier import ClassifierOutput
from bopt.training import TrainingState, ClassificationTrainingSetup
from bopt.training.saving import save_classification_checkpoint
from bopt.training.utils import load_forever
from experiments.utils.reversible_operations import f_cast, f_uncast, f_mul_rat, f_add, to_rational, f_mul_alpha, f_set, f_sub


def reversible_classification(setup: ClassificationTrainingSetup):
    step = 0
    windowed_loss = []
    windowed_loss_avg = math.inf
    best_dev_acc = -1
    v_t_model = f_cast(tuple(torch.zeros_like(p.data) for p in setup.classifier.model.parameters()))
    v_t_tokenizer = f_cast(tuple(torch.zeros_like(p.data) for p in setup.classifier.input_tokenizer.parameters()))
    f_set(setup.classifier.parameters(), f_uncast(f_cast(setup.classifier.parameters())))

    bar = tqdm(enumerate(load_forever(setup.train_dataloader)),total=setup.args.train_steps * setup.args.train_batch_size // setup.args.gpu_batch_size)
    for raw_step, (epoch, batch) in bar:

        # training state
        bar.set_description_str(f"Epoch={epoch} Step={step} Loss={sum(windowed_loss) / len(windowed_loss) if len(windowed_loss) else windowed_loss_avg:.2f}({len(windowed_loss) * setup.args.gpu_batch_size:>4d} exs)")
        state = TrainingState(step, epoch, bar.format_dict['elapsed'])

        if ((raw_step) % (setup.args.train_batch_size // setup.args.gpu_batch_size) == 0) and (
                step % setup.args.eval_steps == 0):
            setup.classifier.eval()
            with torch.no_grad():
                eval_metrics = eval_classification(setup, state)
            setup.classifier.train()

            # early stopping
            if eval_metrics["dev_accuracy"] > best_dev_acc:
                best_dev_acc = eval_metrics["dev_accuracy"]
                save_classification_checkpoint(setup.args.output_directory, "checkpoint-early-stopping", state, setup.classifier)

            # save log
            with open(os.path.join(setup.args.output_directory, "log.json"), "at") as f:
                logline = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed": state.elapsed,
                    "train_loss": windowed_loss_avg,
                    "model_lr": setup.optimizer.named_param_groups["model_decay"]["lr"]
                }
                if setup.args.input_tokenizer_model in ["unigram", "nulm"] and setup.args.input_tokenizer_learning_rate: # only do this if training
                    logline["tokenizer_lr"] = setup.optimizer.named_param_groups["tokenizer"]["lr"]
                logline.update(eval_metrics)
                print(json.dumps(logline), file=f)
                print(logline)

        if raw_step >= ((setup.args.train_steps) * (setup.args.train_batch_size // setup.args.gpu_batch_size)):
            break
        # training
        ids, sentences, labels = batch

        # run model
        output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train")
        loss = output.task_loss * (setup.args.gpu_batch_size / setup.args.train_batch_size)
        loss.backward()

        # maybe step optimizer
        if (raw_step + 1) % (setup.args.train_batch_size // setup.args.gpu_batch_size) == 0:
            if setup.args.task_model_learning_rate != 0:
                # grad
                g_t_model = f_cast(p.grad for p in setup.classifier.model.parameters())
                # new direction
                v_t_first, buffer_t_first = f_mul_rat(v_t_model, *to_rational(setup.args.momentum_coefficient, d=setup.args.momentum_coefficient_precision))
                v_t_model = f_sub(v_t_first, f_mul_alpha(g_t_model, (1-setup.args.momentum_coefficient)))
                w_tp1 = f_add(f_cast(setup.classifier.model.parameters()), f_mul_alpha(v_t_model, setup.args.task_model_learning_rate))
                f_set(setup.classifier.model.parameters(), f_uncast(w_tp1))
            if setup.args.input_tokenizer_learning_rate != 0:
                # grad
                g_t_tokenizer = f_cast(p.grad for p in setup.classifier.input_tokenizer.parameters())
                # new direction
                v_t_first, buffer_t_first = f_mul_rat(g_t_tokenizer, *to_rational(setup.args.momentum_coefficient, d=setup.args.momentum_coefficient_precision))
                v_t_tokenizer = f_sub(v_t_first, f_mul_alpha(g_t_tokenizer, (1-setup.args.momentum_coefficient)))
                w_tp1 = f_add(f_cast(setup.classifier.input_tokenizer.parameters()), f_mul_alpha(v_t_tokenizer, setup.args.input_tokenizer_learning_rate))
                f_set(setup.classifier.input_tokenizer.parameters(), f_uncast(w_tp1))

            step += 1
            setup.classifier.zero_grad()

        windowed_loss.append(loss.item())
        if len(windowed_loss) >= (setup.args.lr_adjustment_window_size // setup.args.gpu_batch_size):
            windowed_loss_avg = sum(windowed_loss) / len(windowed_loss)
            windowed_loss = []

    save_classification_checkpoint(setup.args.output_directory, "checkpoint-final", state, setup.classifier)