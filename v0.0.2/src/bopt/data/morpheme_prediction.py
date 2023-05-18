from tqdm import tqdm
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

