import json
import sys
from itertools import product

label_file = sys.argv[1]
prediction_file = sys.argv[2]

label_tokenizations = []
predicted_tokenizations = []
with open(label_file, "rt") as f:
    for line in f:
        label_tokenizations.append(json.loads(line))

with open(prediction_file, "rt") as f:
    for line in f:
        predicted_tokenizations.append(json.loads(line))

assert len(label_tokenizations) == len(predicted_tokenizations)

token_tp = 0
token_ap = 0
token_at = 0
boundary_tp = 0
boundary_ap = 0
boundary_at = 0
for gold_line, predicted_line in zip(label_tokenizations, predicted_tokenizations):
    for i, j in product(range(len(gold_line["weights"])), range(len(predicted_line["weights"]))):
        weight = gold_line["weights"][i] * predicted_line["weights"][j]

        gold_boundaries = set([b for _, b in gold_line["tokenizations"][i][:-1]]) # skip last boundary
        gold_tokens = set([(b, len(t)) for t, b in gold_line["tokenizations"][i]]) # special characters should alawys be correct so we don't need len_c here

        predicted_boundaries = [b for _, b in predicted_line["tokenizations"][j][:-1]]
        predicted_tokens = [(b, len(t)) for t, b in predicted_line["tokenizations"][j]]

        token_tp += sum(token in gold_tokens for token in predicted_tokens) * weight
        token_ap += len(predicted_tokens) * weight
        token_at += len(gold_tokens) * weight

        boundary_tp += sum(boundary in gold_boundaries for boundary in predicted_boundaries) * weight
        boundary_ap += len(predicted_boundaries) * weight
        boundary_at += len(gold_boundaries) * weight

def f1(tp, ap, at):
    p, r = tp / ap, tp / at
    return p, r, 2 * p * r / (p + r)

token_precision, token_recall, token_f1 = f1(token_tp, token_ap, token_at)
boundary_precision, boundary_recall, boundary_f1 = f1(boundary_tp, boundary_ap, boundary_at)
print(json.dumps({"token_precision": token_precision, "token_recall": token_recall, "token_f1": token_f1, "boundary_precision": boundary_precision, "boundary_recall": boundary_recall, "boundary_f1":boundary_f1}))





