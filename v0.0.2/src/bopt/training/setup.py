import code
import glob
import os.path
import random

import torch.utils.data

from bopt.data import preprocessors
from bopt.modeling import load_model
from bopt.modeling.classifier import Classifier
from bopt.training import ClassificationTrainingSetup
from bopt.training.optimizer import build_optimizers
from bopt.unigram_lm_tokenizers.loading import load_input_tokenizer, load_label_tokenizer
from bopt.utils import load_vocab
from experiments.utils.functions import ramp_function
from experiments.utils.memoizer import OnDiskTensorMemoizer
from experiments.utils.seeding import seed
from experiments.utils.datasets import list_collate

from torch.utils.data import DataLoader, RandomSampler


def setup_classification(args):
    seed(args.seed)

    if os.path.exists(args.output_directory) and not args.overwrite_output_directory:
        raise ValueError("Please set overwrite_output_directory to true when using existing directories.")

    # load vocabularies
    input_vocab = load_vocab(args.input_vocab)
    input_vocab.specials = set(args.special_tokens)
    output_vocab = load_vocab(args.output_vocab)

    # load datasets
    train_dataset = preprocessors[args.domain](args.train_dataset, args)
    train_monitor_dataset = torch.utils.data.Subset(train_dataset, list(range(min(100, len(train_dataset)))))
    dev_dataset = preprocessors[args.domain](args.dev_dataset, args)
    test_dataset = preprocessors[args.domain](args.test_dataset, args)

    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.gpu_batch_size,
                                  num_workers=args.data_num_workers,
                                  collate_fn=list_collate)
    train_monitor_dataloader = DataLoader(train_monitor_dataset,
                                  sampler=RandomSampler(train_monitor_dataset),
                                  batch_size=args.gpu_batch_size,
                                  num_workers=args.data_num_workers,
                                  collate_fn=list_collate)
    dev_dataloader = DataLoader(dev_dataset,
                                  sampler=RandomSampler(dev_dataset),
                                  batch_size=args.gpu_batch_size,
                                  num_workers=args.data_num_workers,
                                  collate_fn=list_collate)
    test_dataloader = DataLoader(test_dataset,
                                  sampler=RandomSampler(test_dataset),
                                  batch_size=args.gpu_batch_size,
                                  num_workers=args.data_num_workers,
                                  collate_fn=list_collate)

    # memoizers / cache
    train_tokenization_memoizer = OnDiskTensorMemoizer(args.train_tokenization_cache, overwrite=args.overwrite_cache)
    dev_tokenization_memoizer = OnDiskTensorMemoizer(args.dev_tokenization_cache, overwrite=args.overwrite_cache)
    test_tokenization_memoizer = OnDiskTensorMemoizer(args.test_tokenization_cache, overwrite=args.overwrite_cache)
    train_label_memoizer = dict()
    dev_label_memoizer = dict()
    test_label_memoizer = dict()

    # model
    model, config = load_model(args.config, pad_token_id=input_vocab.index(args.pad_token), bias_mode=args.bias_mode, saved_model=args.pretrained_model, ignore=args.pretrained_ignore, include=args.pretrained_include)

    # tokenizer
    input_tokenizer = load_input_tokenizer(args.input_tokenizer_model, args.input_tokenizer_mode, input_vocab, log_space_parametrization=args.log_space_parametrization, weight_file=args.input_tokenizer_weights,
                                           num_hidden_layers=args.nulm_num_hidden_layers, hidden_size=args.nulm_hidden_size, tie_embeddings=args.nulm_tie_embeddings, model=model)
    label_tokenizer = load_label_tokenizer(args.input_tokenizer_mode, output_vocab)

    # classifier
    classifier = Classifier(model, input_tokenizer, label_tokenizer).to(args.device)


    # optimizer
    optimizer, scheduler = build_optimizers(args, model, input_tokenizer, args.task_model_learning_rate, args.input_tokenizer_learning_rate, embedding_lr=args.task_model_embedding_learning_rate)

    specials = set(args.special_tokens)

    # scheduler for annealing
    annealing_scheduler = ramp_function(0,
                  args.annealing,
                  args.annealing_start_steps,
                  args.annealing_end_steps)

    return ClassificationTrainingSetup(args=args,
                                       train_dataloader=train_dataloader,
                                       train_monitor_dataloader=train_monitor_dataloader,
                                       dev_dataloader=dev_dataloader,
                                       test_dataloader=test_dataloader,
                                       train_tokenization_memoizer=train_tokenization_memoizer,
                                       dev_tokenization_memoizer=dev_tokenization_memoizer,
                                       test_tokenization_memoizer=test_tokenization_memoizer,
                                       train_label_memoizer=train_label_memoizer,
                                       dev_label_memoizer=dev_label_memoizer,
                                       test_label_memoizer=test_label_memoizer,
                                       classifier=classifier,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       specials=specials,
                                       annealing_scheduler=annealing_scheduler)