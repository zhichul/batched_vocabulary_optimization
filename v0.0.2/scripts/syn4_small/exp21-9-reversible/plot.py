import code

import torch
from tqdm import tqdm
from experiments.plotting.violin import violin_plot

N = 150
data1 = []
data2 = []
canonical_xs = list(range(N))
colors = ["blue"]
names = ["mult_then_renorm"]
RANDTRAIN=5
for name, color in tqdm(list(zip(names, colors))):
    dot_xs = []
    dot_ys = []
    norm_xs = []
    norm_ys = []
    for step in range(N):
        grads = []
        for initialization in range(RANDTRAIN):
            grads.append(torch.load(f"/export/a02/artifacts/boptv2/syn4_small/exp21-9-reversible/42/768/0.00/100/100/1/{name}/0/combined/6.25e-3/0.1/5/3/step-{step}-gradient-{initialization}.pt"))
        for i in range(RANDTRAIN):
            norm_xs.append(step)
            norm_ys.append(grads[i].norm().log10().item())
            for j in range(i+1,RANDTRAIN):
                dot_xs.append(step)
                dot_ys.append(((grads[i] * grads[j]).sum() / (grads[i].norm() * grads[j].norm())).item())
    data1.append({"name":name,
                 "xs": dot_xs,
                 "ys": dot_ys,
                 "color": color})
    data2.append({"name": name,
                 "xs": norm_xs,
                 "ys": norm_ys,
                 "color": color})

violin_plot(data=data1, ndata=len(names), width=0.9, canonical_xs=canonical_xs, dpi=600, xlabel="outer step", ylabel="cosine similarity", output="cosine.png", title="cosine similarity of outer gradient pairs (5choose2 = 10 pairs)\nfrom optimizing the endpoint after 100 inner steps of SGD")
violin_plot(data=data2, ndata=len(names), width=0.9, canonical_xs=canonical_xs, dpi=600, xlabel="outer step", ylabel="log(10) norm", output="norm.png", title="log norm of gradients (5 random restarts)\nfrom optimizing the endpoint after 100 inner steps of SGD")