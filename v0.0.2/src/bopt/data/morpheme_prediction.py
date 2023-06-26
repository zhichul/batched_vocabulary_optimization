from tqdm import tqdm

from bopt.utils import load_vocab
from experiments.utils.datasets import IDExampleDataset
from typing import List
import csv

def preprocess_morpheme_prediction_dataset(file, args):
    examples = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["id", "label", "text", "features", "segmentation"])
        for i, row in enumerate(tqdm(reader)):
            text = " ".join(["[SP1]", "[SP2]", "[SP3]", row["text"]])
            labels = row["features"].split("-")
            examples.append([text, labels])
    return IDExampleDataset(examples)

def preprocess_morpheme_prediction_gold_dataset(file, args):
    vocab = load_vocab(args.input_vocab)
    examples = []
    special = {"[SP1]", "[SP2]", "[SP3]"}
    ids = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["id", "label", "text", "features", "segmentation"])
        for i, row in enumerate(tqdm(reader)):
            text = " ".join(["[SP1]", "[SP2]", "[SP3]", row["text"]])
            labels = row["features"].split("-")
            segmentation = ["[SP1]", "[SP2]", "[SP3]"] + [s for s in row["segmentation"].split("-") if s not in special]
            ids.append( [vocab.index(s) for s in segmentation])
            examples.append([text, labels])
    gold_n = int(args.gold_percentage * len(examples))
    new_examples = []
    for i, ((text, labels), id) in enumerate(zip(examples, ids)):
        if i < gold_n:
            new_examples.append([[text, id], labels])
        else:
            new_examples.append([[text, None], labels])
    return IDExampleDataset(new_examples)

