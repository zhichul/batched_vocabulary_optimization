import os

import torch


def save_learning_dynamics_log(setup, step, weights_dynamics, attention_dynamics, attention_std_dynamics, conditional_marginal_dynamics, weights_grad_dynamics, attention_grad_dynamics, attention_grad_std_dynamics, conditional_marginal_grad_dynamics):
    torch.save({
        "weights":weights_dynamics,
        "attentions":attention_dynamics,
        "attentions_std":attention_std_dynamics,
        "conditional_marginals":conditional_marginal_dynamics,
        "weights_grad":weights_grad_dynamics,
        "attentions_grad":attention_grad_dynamics,
        "attentions_grad_std":attention_grad_std_dynamics,
        "conditional_marginals_grad":conditional_marginal_grad_dynamics,
    }, os.path.join(setup.args.output_directory, f"{step}-dynamics.pt"))