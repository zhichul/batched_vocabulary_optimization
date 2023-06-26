import code
import json
import os
from itertools import product

from matplotlib import pyplot as plt

outer_steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,30,45,60,75,90,105,120,135,150]
inner_steps = [5, 10, 15, 20, 25, 50, 75, 100]
datas = [100, 500]
xaxes = ["reference_log_prob", "boundary_f1"]
yaxes = ["accuracy", "loss"]
eval_datas = ["train.inner", "train.outer", "dev", "test"]
inits = ["train", "eval"]
dir_template = "/export/a02/artifacts/boptv2/syn4_small/exp21-9-reversible-3/42/768/0.00/{0}/100/1/mult_then_renorm/0/combined/6.25e-3/0.1/5/3/safe/"

for data in datas:
    dir = dir_template.format(data)
    for init_class, inner_step, eval_data, xaxis, yaxis in product(inits, inner_steps, eval_datas, xaxes, yaxes):
        xs = []
        ys = []
        alphas = []
        for outer_step in outer_steps:
            with open(os.path.join(dir, f"checkpoint-{outer_step}", f"{eval_data}.1best.tokenizations.f1.json"), "rt") as f:
                tokenization_performance = json.load(f)
            with open(os.path.join(dir, f"step-{outer_step-(1 if init_class == 'train' else 0)}-{init_class}-init-0-log.json"), "rt") as f:
                lines = [json.loads(line) for line in f.readlines()]
            for line in filter(lambda l: l["step"] == inner_step, lines):
                xs.append(tokenization_performance[xaxis])
                ys.append(line[f"{'_'.join(eval_data.split('.'))}_{yaxis}"])
                alphas.append(outer_step / max(outer_steps))
        plt.scatter(xs, ys, alpha=alphas)
        plt.xlabel("_".join([eval_data, xaxis]))
        plt.ylabel(yaxis)
        if yaxis == "accuracy":
            plt.ylim((0, 1.0))
        elif yaxis == "loss":
            plt.ylim((0, 5.0))
        plt.title(f"{data}-data-{init_class}-init-{eval_data}-{yaxis}-vs-{xaxis}-inner-{inner_step}")
        os.makedirs(f"scatters/{data}/{eval_data}", exist_ok=True)
        plt.savefig(f"scatters/{data}/{eval_data}/{data}-data-{init_class}-init-{eval_data}-{yaxis}-vs-{xaxis}-inner-{inner_step}.png", dpi=300)
        # if init_class == "eval":
        #     code.interact(local=locals())
        plt.close()