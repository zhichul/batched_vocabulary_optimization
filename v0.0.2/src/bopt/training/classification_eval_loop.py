import code
import os

from tqdm import tqdm

from bopt.training import ClassificationSetup, TrainingState

import torch

from experiments.metrics.accuracy import accuracy
from experiments.predictions.classification import save_classification_predictions


def eval_classification(setup: ClassificationSetup, state: TrainingState):
    dev_predictions = []
    dev_labels = []
    dev_entropy = 0
    dev_characters = 0
    for batch in tqdm(setup.dev_dataloader):
        ids, sentences, labels = batch
        output = setup.classifier(setup, ids, sentences, labels, mode="dev")
        dev_predictions.append(output.predictions)
        dev_labels.append(output.labels)
        dev_entropy += output.regularizers.entropy.item() * output.regularizers.nchars
        dev_characters += output.regularizers.nchars
    dev_predictions = torch.cat(dev_predictions, dim=0)
    dev_labels = torch.cat(dev_labels, dim=0)
    dev_entropy = dev_entropy / dev_characters

    test_predictions = []
    test_labels = []
    test_entropy = 0
    test_characters = 0
    for batch in tqdm(setup.test_dataloader):
        ids, sentences, labels = batch
        output = setup.classifier(setup, ids, sentences, labels, mode="test")
        test_predictions.append(output.predictions)
        test_labels.append(output.labels)
        test_entropy += output.regularizers.entropy.item() * output.regularizers.nchars
        test_characters += output.regularizers.nchars
    test_predictions = torch.cat(test_predictions, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_entropy = test_entropy / test_characters


    # calculate accuracy, save model if bested, log predictions, log to the global log
    dev_acc = accuracy(dev_predictions, dev_labels)
    test_acc= accuracy(test_predictions, test_labels)

    save_classification_predictions(dev_predictions, f"{os.path.join(setup.args.output_directory, f'dev-predictions-{state.step}.tsv')}")
    save_classification_predictions(dev_labels, f"{os.path.join(setup.args.output_directory, f'dev-labels-{state.step}.tsv')}")
    save_classification_predictions(test_predictions, f"{os.path.join(setup.args.output_directory, f'test-predictions-{state.step}.tsv')}")
    save_classification_predictions(test_labels, f"{os.path.join(setup.args.output_directory, f'test-labels-{state.step}.tsv')}")

    return {"dev_accuracy": dev_acc, "test_accuracy": test_acc, "dev_entroy": dev_entropy, "test_entropy": test_entropy}