import code
import json
import os
from typing import Union

from tqdm import tqdm

from bopt.inference import ClassificationInferenceSetup
from bopt.training import ClassificationTrainingSetup

import torch

from experiments.metrics.accuracy import accuracy
from experiments.predictions.classification import save_classification_predictions


def evaluate(setup, dataloader, mode):
    # setup needs to have a classifier field
    predictions = []
    labels = []
    entropy = 0
    characters = 0
    loss = 0
    for batch in tqdm(dataloader):
        ids, sentences, lbs = batch
        output = setup.classifier(setup, ids, sentences, lbs, mode=mode)
        loss += output.task_loss.item()
        predictions.append(output.predictions)
        labels.append(output.labels)
        if output.regularizers.entropy:  # when a bert tokenizer is used entropy is not defined
            entropy += output.regularizers.entropy.item() * output.regularizers.nchars
            characters += output.regularizers.nchars
    loss = loss / len(dataloader)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    entropy = entropy / characters if characters > 0 else 0
    return predictions, labels, entropy, characters, loss

def eval_classification(setup: ClassificationTrainingSetup, output_name: str):
    predictions, labels, entropy, characters, loss = evaluate(setup, setup.dataloader, None)

    # calculate accuracy, save model if bested, log predictions, log to the global log
    acc = accuracy(predictions, labels)

    save_classification_predictions(predictions, f"{output_name}.predictions.tsv")
    save_classification_predictions(labels, f"{output_name}.labels.tsv")
    with open(f"{output_name}.results.json", "wt") as f:
        print(json.dumps({"accuracy": acc, "entropy":entropy, "loss": loss}), file=f)
