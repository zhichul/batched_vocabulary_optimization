import code
import os

from tqdm import tqdm

from bopt.training import ClassificationSetup, TrainingState

import torch

from experiments.metrics.accuracy import accuracy
from experiments.predictions.classification import save_classification_predictions


def eval_classification(setup: ClassificationSetup, state: TrainingState):
    def eval(dataloader, mode):
        predictions = []
        labels = []
        entropy = 0
        characters = 0
        for batch in tqdm(dataloader):
            ids, sentences, lbs = batch
            output = setup.classifier(setup, ids, sentences, lbs, mode=mode)
            predictions.append(output.predictions)
            labels.append(output.labels)
            if output.regularizers.entropy:  # when a bert tokenizer is used entropy is not defined
                entropy += output.regularizers.entropy.item() * output.regularizers.nchars
                characters += output.regularizers.nchars
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        entropy = entropy / characters if characters > 0 else 0
        return predictions, labels, entropy, characters

    train_predictions, train_labels, train_entropy, train_characters = eval(setup.train_monitor_dataloader, "train")
    dev_predictions, dev_labels, dev_entropy, dev_characters = eval(setup.dev_dataloader, "dev")
    test_predictions, test_labels, test_entropy, test_characters = eval(setup.test_dataloader, "test")

    # calculate accuracy, save model if bested, log predictions, log to the global log
    train_monitor_acc = accuracy(train_predictions, train_labels)
    dev_acc = accuracy(dev_predictions, dev_labels)
    test_acc= accuracy(test_predictions, test_labels)

    save_classification_predictions(train_predictions, f"{os.path.join(setup.args.output_directory, f'train-predictions-{state.step}.tsv')}")
    save_classification_predictions(train_labels, f"{os.path.join(setup.args.output_directory, f'train-labels-{state.step}.tsv')}")
    save_classification_predictions(dev_predictions, f"{os.path.join(setup.args.output_directory, f'dev-predictions-{state.step}.tsv')}")
    save_classification_predictions(dev_labels, f"{os.path.join(setup.args.output_directory, f'dev-labels-{state.step}.tsv')}")
    save_classification_predictions(test_predictions, f"{os.path.join(setup.args.output_directory, f'test-predictions-{state.step}.tsv')}")
    save_classification_predictions(test_labels, f"{os.path.join(setup.args.output_directory, f'test-labels-{state.step}.tsv')}")

    return {"train_accuracy": train_monitor_acc, "dev_accuracy": dev_acc, "train_entropy": train_entropy, "test_accuracy": test_acc, "dev_entroy": dev_entropy, "test_entropy": test_entropy}