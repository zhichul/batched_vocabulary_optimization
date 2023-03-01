import code
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from bopt.core.integerize import Integerizer
from collections import OrderedDict

from bopt.core.modeling_bert import BertForMaskedLM
from bopt.core.modeling_bert import BertConfig


def apply(func, iterable):
    for x in iterable:
        func(x)

def load_vocab(file: Path):
    units = []
    with open(file, "rt") as f:
        for line in f:
            line = line.rstrip()
            unit = line.split("\t")[0]
            units.append(unit)
    return Integerizer(units)

def load_weights(file: Path, tensor=False):
    # returns Dict[str, float(tensor)] if single weight, Dict[str, list[float](tensor)]
    weights = OrderedDict()
    with open(file, "rt") as f:
        for line in f:
            line = line.rstrip()
            items = line.split("\t")
            unit = items[0]
            ws = items[1:]
            weights[unit] = [float(w) for w in ws] if len(ws) > 1 else float(ws[0])
            if tensor:
                weights[unit] = torch.tensor(weights[unit])
    return weights

def save_weights(weights, file: Path, unit_only=False):
    with open(file, "wt") as f:
        for v,w in weights.items():
            if unit_only:
                print(v, file=f)
            else:
                if isinstance(w, torch.Tensor):
                    if w.reshape(-1).size()[-1] == 1:
                        print(f"{v}\t{w.item()}", file=f)
                    else:
                        weght_str = '\t'.join([f'{i}' for i in w.reshape(-1).tolist()])
                        print(f"{v}\t{weght_str}", file=f)
                elif isinstance(w, float):
                    print(f"{v}\t{w}", file=f)
                elif isinstance(w, list) and isinstance(w[0], float):
                    weght_str = '\t'.join([f'{i}' for i in w])
                    print(f"{v}\t{weght_str}", file=f)
                else:
                    print(type(w))
                    raise ValueError(w)

def load_labels(file: Path):
    label_list = list()
    with open(file, "rt") as f:
        for line in f:
            label_list.append(line.strip())
    return label_list

def constant_initializer(vocab: Integerizer, constant=0.0, mixture_count=1):
    # remember this is in log space, so it corresponds to 1.0 in real space
    # this initialization gives weight 1 to all trees, thus having highest entropy
    if mixture_count == 1:
        return {k: constant for k in vocab}
    else:
        return {k: [constant] * mixture_count for k in vocab}

def gaussian_initializer(vocab: Integerizer, mean=0.0, sigma=1.0):
    # remember this is in log space, so it corresponds to 1.0 in real space
    # this initialization gives weight 1 to all trees, thus having highest entropy
    weights = np.random.lognormal(mean, sigma, size=len(vocab))
    return {k:weight for k, weight in zip(vocab, weights)}


def load_forever(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def load_model(model_name, device):
    config = BertConfig.from_json_file(os.path.join(model_name, "config.json"))
    model = BertForMaskedLM.from_pretrained(model_name)
    return model, config
