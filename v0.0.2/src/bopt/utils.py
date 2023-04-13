from collections import OrderedDict

import torch
from torch import autograd

from bopt.integerize import Integerizer

from torch import logaddexp as logaddexp_old

def increasing_roll_left(mat: torch.Tensor, padding_value):
    """
    This function rolls the rows of the input matrix (or tensor) by increasing
    amounts to the left. This is not exactly the inverse to increasing_roll_right
    but close. The reason it is not an inverse is that increasing_roll_right
    throws away elements of the matrix and pads. Rolling left moves those
    thrown away elements back into position but their value is replaced by
    padding.

    For example: the following matrix was produced by increasing_roll_right

        [[1,2,3,4],
        [-1,5,6,7],
        [-1,-1,9,10]]

    would be rolled back into

        [[1,2,3,4],
        [5,6,7,0],
        [9,10,0,0]]

    where 0 is the padding value.

    Usually it is used in conjunction with increasing_roll_right with the same
    padding value.
    """
    size = mat.size()
    if not len(size) >= 2:
        raise ValueError(mat.size())
    rows, cols = size[-2:]
    size_prefix = size[:-2]
    padding1 = torch.zeros(size[:-2] + (rows,rows-1), dtype=mat.dtype, device=mat.device)
    padding1.fill_(padding_value)
    padding2 = torch.zeros(size[:-2] + (rows,), dtype=mat.dtype, device=mat.device)
    padding2.fill_(padding_value)
    block = torch.cat([mat, padding1], dim=-1).reshape(*(size_prefix + (rows * (cols + rows-1),))).reshape(*(size_prefix + (-1,)))
    rolled = torch.cat([block, padding2], dim=-1).reshape(*(size_prefix+ (rows,cols+rows)))[...,:cols]
    return rolled

def increasing_roll_right(mat: torch.Tensor, padding_value):
    """
    This function rolls the rows of the input matrix (or tensor) by increasing
    amounts to the right.

    For example:

        [[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]]

    would be rolled into

        [[1,2,3,4],
        [0,5,6,7],
        [0,0,9,10]]

    where 0 is the padding value.

    """
    size = mat.size()
    if not len(size) >= 2:
        raise ValueError(mat.size())
    rows, cols = size[-2:]
    size_prefix = size[:-2]
    padding = torch.zeros(size[:-2] + (rows,rows-1), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat, padding], dim=-1).reshape(*(size_prefix + (rows * (cols + rows-1),)))[...,:-(rows)].reshape(*(size_prefix+ (rows,cols+rows-2)))[...,:cols]
    return rolled

def col_shift(mat, amount:int, padding_value):
    """
    Shift matrice columns to the left or right
    """
    size = mat.size()
    if not len(size) >= 2:
        raise ValueError(mat.size())
    if not amount != 0:
        raise ValueError(amount)
    rows, cols = size[-2:]
    padding = torch.zeros(size[:-2] + (rows, abs(amount)), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    if amount > 0:
        return torch.cat([padding, mat], dim=-1)[...,:cols]
    if amount < 0:
        return torch.cat([mat, padding], dim=-1)[...,-cols:]

def load_vocab(file):
    units = []
    with open(file, "rt") as f:
        for line in f:
            line = line.rstrip()
            unit = line.split("\t")[0]
            units.append(unit)
    return Integerizer(units)

def load_scalar_weights(file):
    weights = []
    with open(file, "rt") as f:
        for line in f:
            line = line.rstrip()
            ws = line.split("\t")[1:]
            weights.append([float(w) for w in ws])
    return torch.tensor(weights)


class LogAddExpSafe(torch.autograd.Function):
    """Implemented by Jason Eisner 2020 adapted by Brian Lu 2023.
    Implements a torch function that is exactly like logaddexp,
    but is willing to zero out nans on the backward pass."""

    @staticmethod
    def forward(ctx, input, other):  # type: ignore
        with torch.enable_grad():
            output = logaddexp_old(input, other)  # internal copy of output
        ctx.save_for_backward(input, other, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        input, other, output = ctx.saved_tensors
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        zeros = grad_output.new_zeros(grad_output.size())
        if input.requires_grad and other.requires_grad:
            grad_input, grad_other = autograd.grad(output, (input, other), grad_output, only_inputs=True)
            g1, g2 = torch.where(grad_output == 0, zeros, grad_input), torch.where(grad_output == 0, zeros, grad_other)
        elif input.requires_grad:
            grad_input, = autograd.grad(output, (input,), grad_output, only_inputs=True)
            g1, g2 = torch.where(grad_output == 0, zeros, grad_input), None
        elif other.requires_grad:
            grad_other, = autograd.grad(output, (other,), grad_output, only_inputs=True)
            g1, g2 = None, torch.where(grad_output == 0, zeros, grad_other)
        else:
            g1, g2 = None, None
        torch.set_anomaly_enabled(enabled)
        return g1, g2

logaddexp_safe = lambda x, y: LogAddExpSafe.apply(x, y)