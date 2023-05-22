import code
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product

def lboundary(length, rboundary):
    return rboundary-length

def f1(tp, ap, at):
    if ap == 0:
        p = 0
    else:
        p = tp / ap
    r = tp / at
    return p, r, 2 * p * r / (p + r) if (p + r) > 0 else 0


def main():
    parser = ArgumentParser()
    parser.add_argument("label_file")
    parser.add_argument("prediction_file")
    parser.add_argument("--categories_file", default=None)
    args = parser.parse_args()

    label_tokenizations = []
    predicted_tokenizations = []
    with open(args.label_file, "rt") as f:
        for line in f:
            label_tokenizations.append(json.loads(line))

    with open(args.prediction_file, "rt") as f:
        for line in f:
            predicted_tokenizations.append(json.loads(line))

    assert len(label_tokenizations) == len(predicted_tokenizations)

    if args.categories_file:
        label_tokenization_categories = []
        with open(args.categories_file, "rt") as f:
            for line in f:
                label_tokenization_categories.append(json.loads(line))
        assert len(label_tokenizations) == len(label_tokenization_categories)

    token_tp = 0
    token_ap = 0
    token_at = 0
    boundary_tp = 0
    boundary_ap = 0
    boundary_at = 0
    unique_predicted_tokens = set()
    unique_gold_tokens = set()
    if args.categories_file:
        category_token_tp = defaultdict(float)
        category_token_ap = defaultdict(float)
        category_token_at = defaultdict(float)
        category_boundary_tp = defaultdict(float)
        category_boundary_ap = defaultdict(float)
        category_boundary_at = defaultdict(float)
        category_unique_gold_tokens = defaultdict(set)

    for k in range(len(label_tokenizations)):
        gold_line, predicted_line = label_tokenizations[k], predicted_tokenizations[k]
        if args.categories_file:
            category_line = label_tokenization_categories[k]
        for i, j in product(range(len(gold_line["weights"])), range(len(predicted_line["weights"]))):
            weight = gold_line["weights"][i] * predicted_line["weights"][j]

            gold_boundaries = [b for _, b in gold_line["tokenizations"][i][:-1]] # skip last boundary
            gold_tokens = [(b, len(t)) for t, b in gold_line["tokenizations"][i]] # special characters should alawys be correct so we don't need len_c here

            predicted_boundaries = set([b for _, b in predicted_line["tokenizations"][j][:-1]])
            predicted_tokens = set([(b, len(t)) for t, b in predicted_line["tokenizations"][j]]) # the representation of a token is rboundary + length of token
            for rb, length in predicted_tokens:
                unique_predicted_tokens.add(predicted_line["text"][lboundary(length, rb):rb])
            for rb, length in gold_tokens:
                unique_gold_tokens.add(gold_line["text"][lboundary(length, rb):rb])
            token_tp += sum(token in predicted_tokens for token in gold_tokens) * weight
            token_ap += len(predicted_tokens) * weight
            token_at += len(gold_tokens) * weight

            boundary_tp += sum(boundary in predicted_boundaries for boundary in gold_boundaries) * weight
            boundary_ap += len(predicted_boundaries) * weight
            boundary_at += len(gold_boundaries) * weight

            if args.categories_file:
                gold_categories = category_line["tokenizations"][i]
                assert len(gold_tokens) == len(gold_categories)
                for l, (token, category) in enumerate(zip(gold_tokens, gold_categories)):
                    rb, length = token
                    lb = lboundary(length, rb)
                    category_token_tp[category] += float(token in predicted_tokens) * weight
                    category_token_at[category] += weight
                    category_token_ap[category] += (len([b for b in predicted_boundaries if lb < b < rb]) + 1) * weight # num predicted tokens is (strictly) internal boundaries +1
                    if l < len(gold_tokens) - 1:
                        category_boundary_tp[category] += float(rb in predicted_boundaries) * weight
                    if l > 0 :
                        category_boundary_tp[category] += float(lb in predicted_boundaries) * weight
                    category_boundary_at[category] += 1 if (l == 0 or l == len(gold_tokens) - 1) else 2
                    category_boundary_ap[category] += len([b for b in predicted_boundaries if lb <= b <= rb]) * weight # num predicted boundarys is (not-strictly) internal boundaries
                    category_unique_gold_tokens[category].add(gold_line["text"][lb:rb])

    token_precision, token_recall, token_f1 = f1(token_tp, token_ap, token_at)
    boundary_precision, boundary_recall, boundary_f1 = f1(boundary_tp, boundary_ap, boundary_at)
    output = {"token_precision": token_precision, "token_recall": token_recall, "token_f1": token_f1, "boundary_precision": boundary_precision, "boundary_recall": boundary_recall, "boundary_f1":boundary_f1, "unique_predicted_tokens": len(unique_predicted_tokens), "unique_gold_tokens": len(unique_gold_tokens)}
    if args.categories_file:
        for category in category_token_at.keys():
            category_token_precision, category_token_recall, category_token_f1 = f1(category_token_tp[category], category_token_ap[category], category_token_at[category])
            category_boundary_precision, category_boundary_recall, category_boundary_f1 = f1(category_boundary_tp[category], category_boundary_ap[category], category_boundary_at[category])
            output[f"{category}_token_precision"] = category_token_precision
            output[f"{category}_token_recall"] = category_token_recall
            output[f"{category}_token_f1"] = category_token_f1
            output[f"{category}_boundary_precision"] = category_boundary_precision
            output[f"{category}_boundary_recall"] = category_boundary_recall
            output[f"{category}_boundary_f1"] = category_boundary_f1
            output[f"{category}_unique_gold_tokens"] = len(category_unique_gold_tokens[category])
    print(json.dumps(output))





if __name__ == "__main__":
    main()