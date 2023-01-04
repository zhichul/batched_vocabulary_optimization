import code
import json
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLORS = ["orange", "blue"]
DEFAULT_LINESTYLES = ["solid", "dashed"]
parser = ArgumentParser()
parser.add_argument("files", nargs="+", type=str)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--field", nargs="+", type=str)
parser.add_argument("--field_pretty", nargs="+", type=str)
parser.add_argument("--colors", nargs="+", type=str)
parser.add_argument("--linestyles", nargs="+", type=str)
parser.add_argument("--names", nargs="+", type=str)
parser.add_argument("--title")
parser.add_argument("--ylabel")
parser.add_argument("--xlabel")
args = parser.parse_args()

if args.field is None:
    print("No field specified, retruning.")
    exit(0)

assert args.names is None or len(args.names) == len(args.files)
assert args.colors is None or len(args.colors) == len(args.files)
assert args.linestyles is None or len(args.linestyles) == len(args.field)
assert args.field_pretty is None or len(args.field_pretty) == len(args.field)



if args.names is None:
    args.names = args.files

data = OrderedDict()
for name, file in zip(args.names, args.files):
    with open(file, "rt") as f:
        rows = [json.loads(line) for line in f]
    data[name] = rows

fig, ax = plt.subplots()

for i, (name, run) in enumerate(data.items()):
    color = args.colors[i] if args.colors else DEFAULT_COLORS[i]
    for j, field in enumerate(args.field):
        field_pretty = args.field_pretty[j] if args.field_pretty else field
        linestyle = args.linestyles[j] if args.linestyles else DEFAULT_LINESTYLES[j]
        xs = np.arange(len(run)) + 1
        ys = [ckpt[field] for ckpt in run]
        ax.plot(xs, ys, color=color, linestyle=linestyle, label=f"{name}/{field_pretty}")

if args.xlabel:
    ax.set_xlabel(args.xlabel)

if args.ylabel:
    ax.set_ylabel(args.ylabel)

if args.title:
    ax.set_title(args.title)

ax.legend()
plt.savefig(args.out, dpi=300)