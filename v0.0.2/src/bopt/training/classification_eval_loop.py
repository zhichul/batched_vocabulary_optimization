import code
import os

from tqdm import tqdm

from bopt.training import ClassificationTrainingSetup, TrainingState
from bopt.inference.classification_eval_loop import evaluate

import torch

from experiments.metrics.accuracy import accuracy
from experiments.predictions.classification import save_classification_predictions


def eval_classification(setup: ClassificationTrainingSetup, state: TrainingState):

    train_predictions, train_labels, train_entropy, train_characters, train_loss = evaluate(setup, setup.train_monitor_dataloader, "train")
    dev_predictions, dev_labels, dev_entropy, dev_characters, dev_loss,  = evaluate(setup, setup.dev_dataloader, "dev")
    test_predictions, test_labels, test_entropy, test_characters, test_loss = evaluate(setup, setup.test_dataloader, "test")

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

    return {"train_accuracy": train_monitor_acc,
            "dev_accuracy": dev_acc,
            "test_accuracy": test_acc,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "train_entropy": train_entropy,
            "dev_entroy": dev_entropy,
            "test_entropy": test_entropy}