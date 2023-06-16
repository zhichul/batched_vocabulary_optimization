import code
import os

from tqdm import tqdm

from bopt.bilevel import ClassificationBilevelTrainingSetup
from bopt.training import TrainingState
from bopt.inference.classification_eval_loop import evaluate

import torch

from experiments.metrics.accuracy import accuracy
from experiments.predictions.classification import save_classification_predictions


def eval_classification(setup: ClassificationBilevelTrainingSetup, state: TrainingState, save_prediction=True):

    train_inner_predictions, train_inner_labels, train_inner_entropy, train_inner_characters, train_inner_loss = evaluate(setup, setup.train_inner_monitor_dataloader, "train_inner")
    train_outer_predictions, train_outer_labels, train_outer_entropy, train_outer_characters, train_outer_loss = evaluate(setup, setup.train_outer_monitor_dataloader, "train_outer")
    dev_predictions, dev_labels, dev_entropy, dev_characters, dev_loss = evaluate(setup, setup.dev_dataloader, "dev")
    test_predictions, test_labels, test_entropy, test_characters, test_loss = evaluate(setup, setup.test_dataloader, "test")

    # calculate accuracy, save model if bested, log predictions, log to the global log
    train_inner_monitor_acc = accuracy(train_inner_predictions, train_inner_labels)
    train_outer_monitor_acc = accuracy(train_outer_predictions, train_outer_labels)
    dev_acc = accuracy(dev_predictions, dev_labels)
    test_acc= accuracy(test_predictions, test_labels)
    if save_prediction:
        save_classification_predictions(train_inner_predictions, f"{os.path.join(setup.args.output_directory, f'train-inner-predictions-{state.step}.tsv')}")
        save_classification_predictions(train_inner_labels, f"{os.path.join(setup.args.output_directory, f'train-inner-labels-{state.step}.tsv')}")
        save_classification_predictions(train_outer_predictions, f"{os.path.join(setup.args.output_directory, f'train-outer-predictions-{state.step}.tsv')}")
        save_classification_predictions(train_outer_labels, f"{os.path.join(setup.args.output_directory, f'train-outer-labels-{state.step}.tsv')}")
        save_classification_predictions(dev_predictions, f"{os.path.join(setup.args.output_directory, f'dev-predictions-{state.step}.tsv')}")
        save_classification_predictions(dev_labels, f"{os.path.join(setup.args.output_directory, f'dev-labels-{state.step}.tsv')}")
        save_classification_predictions(test_predictions, f"{os.path.join(setup.args.output_directory, f'test-predictions-{state.step}.tsv')}")
        save_classification_predictions(test_labels, f"{os.path.join(setup.args.output_directory, f'test-labels-{state.step}.tsv')}")

    # dev entroy typo is kept for backward compatibility
    return {"train_inner_loss": train_inner_loss,
            "train_outer_loss": train_outer_loss,
            "train_inner_accuracy": train_inner_monitor_acc,
            "train_outer_accuracy": train_outer_monitor_acc,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "dev_accuracy": dev_acc,
            "test_accuracy": test_acc,
            "train_inner_entropy": train_inner_entropy,
            "train_outer_entropy": train_outer_entropy,
            "dev_entroy": dev_entropy,
            "test_entropy": test_entropy}