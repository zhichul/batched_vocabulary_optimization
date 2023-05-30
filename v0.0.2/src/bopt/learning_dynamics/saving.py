import os

import torch


def save_learning_dynamics_log(setup, step, raw_step, **kwargs):
    kwargs["step"] = step
    kwargs["raw_step"] = raw_step
    kwargs["args"] = setup.args
    torch.save(kwargs, os.path.join(setup.args.output_directory, f"{step}-{raw_step}-dynamics.pt"))