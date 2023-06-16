import code
import math

import torch
from tqdm import tqdm

from bopt.bilevel import ClassificationBilevelTrainingSetup, InnerLoopOutput
from bopt.bilevel.classification_eval_loop import eval_classification
from bopt.modeling.classifier import ClassifierOutput
from bopt.training import TrainingState
from bopt.training.utils import load_forever
from experiments.utils.reversible_operations import f_cast, f_uncast, f_mul_rat, f_add, to_rational, f_mul_alpha, f_set, \
    f_sub


def reversible_inner(setup: ClassificationBilevelTrainingSetup, outer_step: int, warmup=False):

    step = 0
    windowed_loss = []
    windowed_loss_avg = math.inf
    grad_norm = math.inf
    buffers_first = []
    buffers_second = []
    batch_ids = []
    batch_id = []
    v_t  = f_cast(tuple(torch.zeros_like(p.data) for p in setup.classifier.model.parameters()))
    zero_iterate  = f_cast(setup.classifier.model.parameters())
    f_set(setup.classifier.model.parameters(), f_uncast(zero_iterate))
    second_iterate = None
    first_iterate = None
    first_step = None
    third_iterate = None
    g_1 = None
    g_2 = None
    g_t = None
    second_step = None
    bar = tqdm(enumerate(load_forever(setup.train_inner_dataloader)),total=(setup.args.train_steps_inner if not warmup else setup.args.train_steps_warmup) * setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner)
    for raw_step, (epoch, batch) in bar:
        # training state
        bar.set_description_str(f"Inner Epoch={epoch} Step={step} InnerLoss={sum(windowed_loss) / len(windowed_loss) if len(windowed_loss) else windowed_loss_avg:.2f}({len(windowed_loss) * setup.args.gpu_batch_size_inner:>4d} exs)")
        state = TrainingState(step, epoch, bar.format_dict['elapsed'])

        if raw_step >= ((setup.args.train_steps_inner if not warmup else setup.args.train_steps_warmup) * (setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner)) or grad_norm <= setup.args.inner_threshold:
            break
        # training
        ids, sentences, labels = batch
        batch_id.append(ids)

        # run model
        output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_inner")
        loss = output.task_loss * (setup.args.gpu_batch_size_inner / setup.args.train_batch_size_inner)
        loss.backward()
        loss = loss.item()
        # print(loss)
        # maybe step optimizer
        if (raw_step + 1) % (setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner) == 0:
            grad_norm = math.sqrt(sum(p.grad.norm() ** 2 for p in setup.classifier.model.parameters() if p.grad is not None).item())
            # grad
            g_t = f_cast(p.grad for p in setup.classifier.model.parameters())
            # new direction
            v_t_first, buffer_t_first = f_mul_rat(v_t, *to_rational(setup.args.momentum_coefficient, d=setup.args.momentum_coefficient_precision))
            # v_tp1_second = [mult_by_ratio(grad, setup.args.momentum_dampening[1]-setup.args.momentum_dampening[0], setup.args.momentum_dampening[1]) for grad in g_t]
            # buffer_t_second = [(buffer.detach().cpu(), dexp.detach().cpu()) for _, buffer, dexp in v_tp1_second]
            buffers_first.append(buffer_t_first)
            # buffers_second.append(buffer_t_second)
            # v_tp1 = torch._foreach_add([tup[0] for tup in v_tp1_first], [tup[0] for tup in v_tp1_second], alpha=-1)
            # v_t = torch._foreach_add(v_t_first, g_t, alpha=-(1-setup.args.momentum_dampening[0]/setup.args.momentum_dampening[1]))
            v_t = f_sub(v_t_first, f_mul_alpha(g_t, (1-setup.args.momentum_coefficient))) # this one not invertible
            w_tp1 = f_add(f_cast(setup.classifier.model.parameters()), f_mul_alpha(v_t, setup.args.task_model_learning_rate))

            f_set(setup.classifier.model.parameters(), f_uncast(w_tp1))

            # torch._foreach_add_(tuple(setup.classifier.model.parameters()), v_t, alpha=setup.args.task_model_learning_rate)  # NO NEGATIVE SIGN ON THE LR!!!! negative sign is introduced in the v_tp1 lines

            # torch._foreach_add_(tuple(setup.classifier.model.parameters()), v_t, alpha=setup.args.task_model_learning_rate)  # NO NEGATIVE SIGN ON THE LR!!!! negative sign is introduced in the v_tp1 lines
            # torch._foreach_add_(tuple(setup.classifier.model.parameters()), g_t, alpha=-setup.args.task_model_learning_rate) # NO NEGATIVE SIGN ON THE LR!!!! negative sign is introduced in the v_tp1 lines

            # step_norm = math.sqrt(sum((v ** 2).sum() for v in f_uncast(v_t)).item())

            # loggibg for debugging
            # if step_norm > 100000000:
            #     code.interact(local=locals())
            #     print(step_norm)
            # if any(v.isnan().any() for v in setup.classifier.model.parameters()):
            #     print("NOOoOOOO")
            #     code.interact(local=locals())
            # if step == 0:
            #     g_1 = g_t
            #     first_step = v_t
            #     first_iterate = f_cast(setup.classifier.model.parameters())
            # if step == 1:
            #     g_2 = g_t
            #     second_step = v_t
            #     second_iterate = f_cast(setup.classifier.model.parameters())
            # if step == 2:
            #     third_iterate = f_cast(setup.classifier.model.parameters())

            step += 1
            batch_ids.append(batch_id)
            batch_id = []
            setup.classifier.model.zero_grad()
            # code.interact(local=locals())
            # code.interact(local=locals())

        # maybe step scheduler, only for ift currently
        windowed_loss.append(loss)
        if len(windowed_loss) >= (setup.args.lr_adjustment_window_size // setup.args.gpu_batch_size_inner):
            windowed_loss_avg = sum(windowed_loss) / len(windowed_loss)
            windowed_loss = []
            # if setup.args.annealing > 0 and setup.args.annealing_start_steps <= step <= setup.args.annealing_end_steps:
            #     setup.inner_scheduler._reset()
            #     # reset so never step down due to increasing entropy loss factor, and continue with new baseline after
            #     # the annealing factor is maxed out
            # setup.inner_scheduler.step(windowed_loss_avg)
    # if setup.args.bilevel_optimization_scheme == "ift":
    #     setup.classifier.eval()
    #     with torch.no_grad():
    #         eval_metrics = eval_classification(setup, state, save_prediction=False)
    #     setup.classifier.train()
    #     # save log
    #     logline = {
    #         "step": step,
    #         "epoch": epoch,
    #         "elapsed": state.elapsed,
    #         "train_loss": windowed_loss_avg,
    #         "model_lr": setup.inner_optimizer.named_param_groups["model_decay"]["lr"]
    #     }
    #     logline.update(eval_metrics)
    #     print("\t" + str(logline))
    # code.interact(local=locals())
    return InnerLoopOutput(buffers_first=buffers_first, buffers_second=buffers_second, batch_ids=batch_ids, last_step=v_t, last_grad=f_uncast(g_t) if g_t is not None else None,zero_iterate=zero_iterate,  first_iterate=first_iterate, g_1=f_uncast(g_1)if g_1 is not None else None, g_2=f_uncast(g_2)if g_2 is not None else None, second_iterate=second_iterate, third_iterate=third_iterate, second_step=second_step, first_step=first_step)

