import code
import math
from collections import defaultdict

import torch
from tqdm import tqdm

INF=1e9

def increasing_roll_left(mat: torch.Tensor, padding_value, shift=False):
    size = mat.size()
    if not len(size) > 2:
        raise ValueError(mat.size())
    R, C = size[-2:]
    padding = torch.zeros(size[:-2] + (R,), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat.reshape(*(size[:-2] + (-1,))), padding], dim=-1).reshape(*(size[:-2] + (R, C + 1)))
    out = rolled[..., :C]
    return out

def increasing_roll_right(mat: torch.Tensor, padding_value):
    size = mat.size()
    if not len(size) >= 2:
        raise ValueError(mat.size())
    R, C = size[-2:]
    padding = torch.zeros(size[:-2] + (R,1), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat, padding], dim=-1).reshape(*(size[:-2] + (R * (C + 1),)))[...,:-R].reshape(*(size[:-2] + (R,C)))
    return rolled

def forever_generator():
    while True:
        yield None


def length_normalized_initialization(model, tokenizer, **kwargs):
    char_bias = find_bias(tokenizer, **kwargs)
    bias = [tokenizer.len_type(unit) * char_bias for unit in tokenizer.vocab]
    model.cls.predictions.bias.data[:len(bias)] = torch.tensor(bias, device = model.cls.predictions.bias.data.device)
    model.cls.predictions.bias.data[len(bias):] = -INF
def find_bias(tokenizer, eps = 1e-3, tol=1e-3, lr=1e-4):
    length_counts = defaultdict(int)
    for unit in tokenizer.vocab:
        length_counts[tokenizer.len_type(unit)] += 1

    x = torch.nn.Parameter(torch.tensor(math.log(1/sum(length_counts.values()))))
    loss = torch.tensor([math.inf])
    bar = tqdm(forever_generator())
    for iter in bar:
        if loss.item() <= tol:
            break
        values = []
        for length, count in length_counts.items():
            values.extend([x * length for _ in range(count)])
        loss = (torch.logsumexp(torch.stack(values), -1) - 1) ** 2
        loss.backward()
        x.data -= lr * x.grad
        x.data = torch.clamp(x.data, max=-eps)
        bar.desc = f"loss = {loss}, x = {x.item()}"
    return x.item()
