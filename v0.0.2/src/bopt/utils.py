import code
from collections import OrderedDict

import torch
from torch import autograd

from bopt.integerize import Integerizer

from torch import logaddexp as logaddexp_old
from torch import log as log_old

DEBUG=False
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


class LogAddExpGradSafe(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, input, other, grad_output, create_graph):
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        # print("forward call in the logaddexp grad function")
        r0 = input.requires_grad
        r1 = other.requires_grad
        grad0 = grad1 = None
        # make a copy of the local graph for logaddexp that is detached from the rest of the graph
        input, other, grad_output = input.detach(), other.detach(), grad_output.detach()
        input.requires_grad = True
        other.requires_grad = True
        grad_output.requires_grad = True
        with torch.enable_grad():
            output = logaddexp_old(input, other)
        with torch.set_grad_enabled(create_graph):
            grad = autograd.grad(output, [t for t, rg in zip([input, other],[r0, r1]) if rg], grad_output, create_graph=create_graph, retain_graph=True, allow_unused=True)
        if r0 and r1:
            grad0, grad1 = grad
        elif r0:
            grad0 = grad[0]
        elif r1:
            grad1 = grad[0]
        ctx.save_for_backward(grad0, grad1, output, input, other, grad_output)
        torch.set_anomaly_enabled(enabled)
        return grad0.detach() if grad0 is not None else None, grad1.detach() if grad1 is not None else None

    @staticmethod
    def backward(ctx, grad_output0, grad_output1):
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        # print("backward call in the logaddexp grad function")
        grad0, grad1, output, input, other, grad_output = ctx.saved_tensors
        gradable = [v for v in [output, input, other, grad_output] if v is not None and isinstance(v, torch.Tensor) and v.requires_grad] #left out output
        grad_from0 = None
        grad_from1 = None
        zeros = grad_output.new_zeros(grad_output.size())
        # if not DEBUG:
        #     code.interact(local=locals())
        if grad0 is not None:
            # backprop through the local copy so that we don't end up recursing all the way
            grad_from0 = autograd.grad((grad0,), gradable, grad_outputs=(grad_output0,), retain_graph=True, allow_unused=True)
            grad_from0 = [torch.where((grad_output == 0) | (input == -torch.inf) | (input < -20), zeros, v) if v is not None else None for v in grad_from0]
        if grad1 is not None:
            # backprop through the local copy so that we don't end up recursing all the way
            grad_from1 = autograd.grad((grad1,), gradable, grad_outputs=(grad_output1,), retain_graph=True, allow_unused=True)
            grad_from1 = [torch.where((grad_output == 0) | (other == -torch.inf) | (other < -20), zeros, v)  if v is not None else None for v in grad_from1]
        outputs = []
        i = 0
        for v in [output, input, other, grad_output]:
            if v is None or not isinstance(v, torch.Tensor) or not v.requires_grad:
                outputs.append(None)
            else:
                if (grad_from0 is None and grad_from1 is None):
                    outputs.append(None)
                else:
                    grad = 0
                    if grad_from0 is not None and grad_from0[i] is not None:
                        grad += grad_from0[i]
                    if grad_from1 is not None and grad_from1[i] is not None:
                        grad += grad_from1[i]
                    if isinstance(grad, int):
                        outputs.append(None)
                    else:
                        outputs.append(grad)
                i += 1
        torch.set_anomaly_enabled(enabled)
        # code.interact(local=locals())
        if any(o.isnan().any() for o in outputs if o is not None):
            print("NAAAHNNNNNNN")
            code.interact(local=locals())
        return tuple(outputs) + (None,) # last one is for create_graph

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
        create_graph = torch.is_grad_enabled()
        grad_input, grad_other = LogAddExpGradSafe.apply(output, input, other, grad_output, create_graph)
        # if not DEBUG:
        #     code.interact(local=locals())
        if input.requires_grad and other.requires_grad:
            g1, g2 = torch.where(grad_output == 0, zeros, grad_input), torch.where(grad_output == 0, zeros, grad_other)
        elif input.requires_grad:
            g1, g2 = torch.where(grad_output == 0, zeros, grad_input), None
        elif other.requires_grad:
            g1, g2 = None, torch.where(grad_output == 0, zeros, grad_other)
        else:
            g1, g2 = None, None
        torch.set_anomaly_enabled(enabled)
        # if g1 is not None and g1.isnan().any() or g2 is not None and g2.isnan().any():
        #     code.interact(local=locals())
        return g1, g2

logaddexp_safe = lambda x, y: LogAddExpSafe.apply(x, y)


class LogGradSafe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, input, grad_output, create_graph):  # type: ignore
        # grad_input, =
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        r0 = input.requires_grad
        grad0 = None
        # make a copy of the local graph for logaddexp that is detached from the rest of the graph
        input, grad_output = input.detach(), grad_output.detach()
        input.requires_grad = True
        grad_output.requires_grad = True
        with torch.enable_grad():
            output = log_old(input)
        with torch.set_grad_enabled(create_graph):
            grad, = autograd.grad(output, (input), grad_output, retain_graph=True, create_graph=create_graph, allow_unused=True)
        if r0:
            grad0 = grad
        ctx.save_for_backward(grad0, output, input, grad_output)
        torch.set_anomaly_enabled(enabled)
        return grad0.detach() if grad0 is not None else None

    @staticmethod
    def backward(ctx, grad_output0):  # type: ignore
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        grad0, output, input, grad_output = ctx.saved_tensors
        gradable = [v for v in [output, input, grad_output] if v is not None and isinstance(v, torch.Tensor) and v.requires_grad]  # left out output
        grad_from0 = None
        zeros = grad_output.new_zeros(grad_output.size())
        if grad0 is not None:
            # backprop through the local copy so that we don't end up recursing all the way
            grad_from0 = autograd.grad((grad0,), gradable, grad_outputs=(grad_output0,), retain_graph=True, allow_unused=True)
            grad_from0 = [torch.where((grad_output == 0) | (input == 0), zeros, v) if v is not None else None for v in grad_from0]
        outputs = []
        i = 0
        for v in [output, input, grad_output]:
            if v is None or not isinstance(v, torch.Tensor) or not v.requires_grad:
                outputs.append(None)
            else:
                if grad_from0 is None:
                    outputs.append(None)
                else:
                    if grad_from0[i] is not None:
                        outputs.append(grad_from0[i])
                    else:
                        outputs.append(None)
                i += 1
        torch.set_anomaly_enabled(enabled)
        # code.interact(local=locals())
        if any(o.isnan().any() for o in outputs if o is not None):
            print("NAAAHNNNNNNN")
            code.interact(local=locals())
        return tuple(outputs) + (None,)  # last one is for create_graph
class LogSafe(torch.autograd.Function):
    """Implemented by Jason Eisner 2020 adapted by Brian Lu 2023.
    Implements a torch function that is exactly like logaddexp,
    but is willing to zero out nans on the backward pass."""

    @staticmethod
    def forward(ctx, input):  # type: ignore
        with torch.enable_grad():
            output = log_old(input)  # internal copy of output
        ctx.save_for_backward(input, output)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        # print("############ logsafe backward")
        input, output = ctx.saved_tensors
        enabled = torch.is_anomaly_enabled()
        torch.set_anomaly_enabled(False)
        zeros = grad_output.new_zeros(grad_output.size())
        if input.requires_grad:
            create_graph = torch.is_grad_enabled()
            grad_input = LogGradSafe.apply(output, input, grad_output, create_graph)

            # grad_input, = autograd.grad(output, (input), grad_output, retain_graph=True, create_graph=torch.is_grad_enabled())
            g1 = torch.where(grad_output == 0, zeros, grad_input)
            # if g1.isnan().any():
            #     code.interact(local=locals())
        else:
            g1 = None
        torch.set_anomaly_enabled(enabled)
        return g1

log_safe = lambda x: LogSafe.apply(x)

def product(l):
    if len(l) == 0:
        raise AssertionError
    out = 1
    for item in l:
        out *= item
    return out