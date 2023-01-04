import code
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from bopt.core.tokenizer.attention import LatticeAttentionMixin
from argparse import ArgumentParser

from bopt.core.utils import increasing_roll_right

parser = ArgumentParser()
parser.add_argument("log_file")
parser.add_argument("out_dir")
args = parser.parse_args()

# np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

cp = []
with open(args.log_file) as f:
    for line in f:
        d = json.loads(line)
        unit = np.array(d["unit"])
        marginal = np.exp(np.array(d["marginal"]))
        _, _, E = marginal.shape
        log_prob = np.exp(np.array(d["log_prob"]))
        B, BL, M, L = log_prob.shape
        start = increasing_roll_right((torch.arange(L)[None, None, None, ...] + (torch.arange(BL) * L)[None, ...,None, None]).expand(B, BL, M, L), padding_value=-1).numpy()
        length = (torch.arange(M)[None, None, ..., None] + 1 ).expand(B, BL, M, L).numpy()

        triu_ones = torch.triu(torch.ones(M, L, dtype=torch.bool), diagonal=0)
        index = triu_ones[None, None, ...].expand(B, BL, -1, -1).numpy()
        permutation = LatticeAttentionMixin.permutation(L, M)
        unitE = unit[index].reshape(B, BL, E)[..., permutation]
        startE = start[index].reshape(B, BL, E)[..., permutation]
        lengthE = length[index].reshape(B, BL, E)[..., permutation]
        log_probE = log_prob[index].reshape(B, BL, E)[..., permutation]
        marginalE = marginal

        newrow = {"key": d["key"], "unit": unitE, "marginal": marginalE, "log_prob": log_probE, "start": startE, "length":lengthE}
        if "lm_marginal" in d:
            lm_marginal = np.exp(np.array(d["lm_marginal"]))
            lm_marginalE = lm_marginal
            newrow["lm_marginal"] = lm_marginalE
        cp.append(newrow)

for ckpt in cp:
    folder = os.path.join(args.out_dir, ckpt["key"])
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    for i, unit in enumerate(ckpt["unit"]):
        plf = []
        nodes = defaultdict(list)
        for b, block in enumerate(unit):
            for e, E in enumerate(block):
                if E == "[PAD]":
                    continue

                start = ckpt["start"][i,b,e]
                length = ckpt["length"][i,b,e]
                log_prob = ckpt["log_prob"][i,b,e]
                marginal = ckpt["marginal"][i,b,e]
                unit = ckpt["unit"][i,b,e]
                node_rep = {"unit": unit, "marginal": marginal, "start": int(start), "log_prob": log_prob, "length": int(length)}
                if "lm_marginal" in ckpt:
                    node_rep["lm_marginal"] = ckpt["lm_marginal"][i,b,e]
                nodes[start].append(node_rep)

        with open(os.path.join(folder, f"{i}.json"), "wt") as f:
            print(json.dumps([nodes[i] for i in sorted(list(nodes.keys()))]), file=f)