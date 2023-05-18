from tqdm import tqdm
from experiments.utils.datasets import IDExampleDataset
from typing import List
import csv

def preprocess_superbizarre_prediction_dataset(file, args):
    examples = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["text", "score", "label"])
        for i, row in enumerate(tqdm(reader)):
            if i == 0: continue # skip the header line
            if args.input_tokenizer_model in ["unigram", "nulm"]:
                if (len(row["text"]) > args.max_block_length - 2): continue # skip lines that are longer than threshold, -2 is to make space for the [CLS] and for the added ▁
                text = "▁".join(["[CLS]", row["text"]])
            else:
                text = row["text"]
            labels = [row["label"]]
            examples.append([text, labels])
    return IDExampleDataset(examples)

