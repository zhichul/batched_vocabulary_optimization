import code
import math

import higher
from tqdm import tqdm

from bopt.bilevel import ClassificationBilevelTrainingSetup, TrainingState
from bopt.bilevel.classification_reversible_inner import reversible_inner
from bopt.bilevel.classification_reversible_grad import reversible_grad
from bopt.bilevel.classification_train_inner import train_classification_inner, InnerLoopOutput
from bopt.bilevel.utils import stop_tokenizer_grad, extract_tokenizer
from bopt.modeling.classifier import ClassifierOutput
from bopt.training.utils import load_forever
from experiments.utils.optimization import set_grad
from experiments.utils.reversible_operations import f_uncast, f_cast


def accumulate_rollout_gradient(rollout_output: InnerLoopOutput, setup: ClassificationBilevelTrainingSetup):
    fmodel = higher.monkeypatch(setup.classifier, track_higher_grads=False)
    num_batches = max(min(math.ceil(setup.args.train_batch_size/setup.args.gpu_batch_size), len(setup.train_outer_dataloader)), 1)
    params = stop_tokenizer_grad(setup.classifier.tokenizer_parameter_mask, rollout_output.params) if setup.args.indirect_gradient_only else rollout_output.params
    losses = 0
    for i, batch in enumerate(setup.train_outer_dataloader):
        if i >= setup.args.train_batch_size // setup.args.gpu_batch_size: break  # only do one training batch max
        # training
        ids, sentences, labels = batch

        # run model
        output: ClassifierOutput = fmodel(setup, ids, sentences, labels, "train_outer", params=params)
        #
        # code.interact(local=locals())
        # compute loss
        (output.task_loss / num_batches / setup.args.train_trajectory_inner / setup.args.random_restarts).backward(retain_graph=i < (num_batches-1))
        losses += output.task_loss.item() / num_batches / setup.args.train_trajectory_inner / setup.args.random_restarts
    tokenizer_params = extract_tokenizer(setup.classifier.tokenizer_parameter_mask, setup.classifier.parameters())
    if not (len(tokenizer_params) == len(rollout_output.init_tokenizer_params)):
        code.interact(local=locals())
    for p, np in zip(tokenizer_params, rollout_output.init_tokenizer_params):
        if p.grad is None:
            p.grad = np.grad
        else:
            p.grad += np.grad
    return losses

def train_trajectory_inner(setup: ClassificationBilevelTrainingSetup, outer_step):
    windowed_loss = []
    windowed_loss_avg = math.inf
    bar = tqdm(range(setup.args.train_trajectory_inner))
    outer_loss = 0
    initial_params = f_uncast(f_cast(p.detach().clone() for p in setup.classifier.model.parameters()))
    for step in bar:
        bar.set_description_str(f"Traj Epoch={(step * setup.args.train_batch_size_inner) / len(setup.train_inner_dataloader.dataset) if step > 0 else 0} Step={step} InnerLoss={sum(windowed_loss) / len(windowed_loss) if len(windowed_loss) else windowed_loss_avg:.2f}({len(windowed_loss) * setup.args.train_batch_size_inner:>4d} exs)")
        # unroll requires diff-through-opt so do that instead
        if setup.args.bilevel_optimization_scheme == "reversible-learning":
            set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=False)
        rollout_output = train_classification_inner(setup, outer_step)
        if setup.args.bilevel_optimization_scheme == "reversible-learning":
            set_grad(setup.classifier.input_tokenizer.parameters(), requires_grad=True)
        if setup.args.bilevel_optimization_scheme == "unroll":
            # manually update model parameters with a single step
            for np, p in zip(rollout_output.one_step_params, setup.classifier.parameters()):
                p.data = np.data
        if setup.args.bilevel_optimization_scheme == "reversible-learning":
            outer_loss += reversible_grad(setup, rollout_output)
            if not all((p1 == p2).all() for p1, p2 in zip(initial_params, setup.classifier.model.parameters())):
                print("Did not reverse properly : (")
                code.interact(local=locals())
        else:
            outer_loss += accumulate_rollout_gradient(rollout_output, setup)
        # maybe step scheduler
        loss = rollout_output.one_step_loss
        windowed_loss.append(loss)
        if len(windowed_loss) >= (setup.args.lr_adjustment_window_size // setup.args.train_batch_size_inner):
            windowed_loss_avg = sum(windowed_loss) / len(windowed_loss)
            setup.inner_scheduler.step(windowed_loss_avg)
            windowed_loss = []
    return outer_loss
