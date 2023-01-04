import code

import numpy as np

from bopt.data.utils import load_model
import sys
import matplotlib.pyplot as plt

model_name = sys.argv[1]

output_name = sys.argv[2]
first_n = int(sys.argv[3])

model, config = load_model(model_name, "cpu")
weights = model.bert.embeddings.position_embeddings.weight
N, D = weights.shape

logits = weights @ weights.t()
norm = weights.norm(p=2, dim=-1, keepdim=True)
probs = logits.softmax(dim=-1)
cosine = logits / (norm * norm.t())

logits = logits.detach().numpy()
probs = probs.detach().numpy()
cosine = cosine.detach().numpy()[:first_n,:first_n]
# import pickle
# pickle.dump(logits, open("logits.pkl", "wb"))
# pickle.dump(probs, open("probs.pkl", "wb"))
# code.interact(local=locals())
# fig, ax = plt.subplots()
# im = ax.imshow(logits)
# # for i in range(N):
# #     for j in range(N):
# #         text = ax.text(j, i, round(logits[i,j],2),
# #                        ha="center", va="center", color="w")
# plt.savefig(f"{output_name}-logits.png", dpi=600)
# fig, ax = plt.subplots()
# im = ax.imshow(probs)
# # for i in range(N):
# #     for j in range(N):
# #         text = ax.text(j, i, probs[i,j],
# #                        ha="center", va="center", color="w")
# plt.savefig(f"{output_name}-probs.png", dpi=600)
fig, ax = plt.subplots()
im = ax.imshow(cosine)
plt.savefig(f"{output_name}-cosine.png", dpi=600)