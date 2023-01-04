import code
import math
import sys
import json
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict

import numpy as np

parser = ArgumentParser()
parser.add_argument("field", nargs="+")
args = parser.parse_args()

def batchify(x):
    try:
        x = np.array(x)
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        x = x.tolist()
    except:
        print(x, file=sys.stderr)
        print(x.shape, file=sys.stderr)
    return x

def group_lasso(counts, indices):
    d = defaultdict(list)
    for batch in range(len(counts)):
        for idx in range(len(counts[batch])):
            d[indices[batch][idx]].append(counts[batch][idx])
    gl = 0
    for unit in d:
        p = len(d[unit])
        norm = np.linalg.norm(np.array(d[unit]))
        gl += math.sqrt(p) * norm
    return gl

AGGREGATORS = {
    "ent": lambda x: np.array(x).mean(),
    "marginal": batchify,
    "log_prob": batchify,
    "unit": batchify,
    "lm_marginal": batchify,
}
checkpoints = OrderedDict()
for line in sys.stdin:
    row = json.loads(line)
    key = row["key"]
    if key not in checkpoints:
        checkpoints[key] = []
    checkpoints[key].append(row)

for key, data in checkpoints.items():
    d = dict()
    d["key"] = key
    for field in args.field:
        if field == "group_lasso":
            counts = [row["marginal_count"] for row in data]
            indices = [row["token"] for row in data]
            value = group_lasso(counts, indices)
        else:
            values = [row[field] for row in data]
            value = AGGREGATORS[field](values)
        d[field] = value
    print(json.dumps(d))

