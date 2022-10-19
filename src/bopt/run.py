import glob
import os
from collections import OrderedDict
from time import time
import cProfile

from torch.optim import Adam
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from bopt.forward_step import morpheme_prediction_step, language_modeling_step
from bopt.forward_loop import language_modeling_loop

from bopt.arguments import parse_args
from bopt.core.tokenizer import Tokenizer
from bopt.data.morpheme_prediction.lattice import preprocess_morpheme_prediction_with_lattices_dataset, \
    MorphemePredictionLatticeDataset
from bopt.data.language_modeling.lattice import LanguageModelingLatticeDataset, preprocess_language_modeling_with_lattices_dataset
from grid_utils import acquire_all_available_gpu
import logging
import torch
import random
import numpy as np
import code
from bopt.core.modeling_bert import BertForMaskedLM, BertConfig
from bopt.data.utils import load_vocab, load_weights, constant_initializer, save_weights
import json

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

acquire_all_available_gpu()

def initialize(args):
    # seed the experiment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    return device, n_gpu

def load_vocab_and_weights(args):
    logger.info("Loading vocabs and weight...")
    input_vocab = output_vocab = load_vocab(args.input_vocab)
    if args.output_vocab:
        output_vocab = load_vocab(args.output_vocab)
    if args.weights_file:
        weights = load_weights(args.weights_file)
    else:
        weights = constant_initializer(input_vocab, constant=0.0)
    return input_vocab, output_vocab, weights

def load_model(args, device):
    logger.info("Loading model...")
    if args.model_name is not None:
        config = BertConfig.from_json_file(os.path.join(args.model_name, "config.json"))
        model = BertForMaskedLM.from_pretrained(args.model_name)
    else:
        config = BertConfig.from_json_file(args.config)
        model = BertForMaskedLM(config)
    model.to(device)
    model.bias_mode = args.bias_mode
    return model, config


def load_tokenizer(args, input_vocab, device):
    logger.info("Building tokenizer...")
    tokenizer = Tokenizer(vocab=input_vocab,
                          weights=constant_initializer(input_vocab),
                          log_space_parametrization=False,
                          continuing_subword_prefix=args.continuing_subword_prefix,
                          pad_token="[PAD]",
                          max_unit_length=args.max_unit_length,
                          specials=args.specials,
                          )
    tokenizer.to(device)
    return tokenizer

def create_or_clear_cache(args, cache_dir):
    if os.path.exists(cache_dir):
        if not args.overwrite_cache:
            logger.info(f"{cache_dir} exists, using it as is...")
            return False
        else:
            for f in glob.glob(f'{cache_dir}/*'):
                os.remove(f)
            return True
    else:
        os.makedirs(cache_dir)
        return True

def preprocess_datasets(args, tokenizer, input_vocab, output_vocab):
    logger.info("Preprocessing Datasets...")
    if args.task == "morpheme_prediction":
        datasets = {}
        dataloaders = {}
        for name, data in zip(["train", "eval"], [args.train_dataset, args.eval_dataset]):
            if data is None:
                logger.info(f"No {name} dataset specified, continuing...")
                continue
            cache_dir = os.path.join(args.output_dir, f"cache", os.path.basename(data))
            flag = create_or_clear_cache(args, cache_dir)
            if flag:
                preprocess_morpheme_prediction_with_lattices_dataset(data,
                                                                     cache_dir,
                                                                     tokenizer,
                                                                     output_vocab,
                                                                     args.max_blocks,
                                                                     args.max_block_length,
                                                                     args.max_unit_length,
                                                                     debug=False)
            datasets[name] = dataset = MorphemePredictionLatticeDataset(cache_dir)
            sampler = RandomSampler(dataset) if name == "train" else SequentialSampler(dataset)
            dataloaders[name] = dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.gpu_batch_size, num_workers=args.data_num_workers)
    elif args.task == "language_modeling":
        datasets = {}
        dataloaders = {}
        for name, data in zip(["train", "eval"], [args.train_dataset, args.eval_dataset]):
            if data is None:
                logger.info(f"No {name} dataset specified, continuing...")
                continue
            cache_dir = os.path.join(args.output_dir, f"cache", os.path.basename(data))
            flag = create_or_clear_cache(args, cache_dir)
            if flag:
                preprocess_language_modeling_with_lattices_dataset(data,
                                                                     cache_dir,
                                                                     tokenizer,
                                                                     output_vocab,
                                                                     args.max_blocks,
                                                                     args.max_block_length,
                                                                     args.max_unit_length)
            datasets[name] = dataset = LanguageModelingLatticeDataset(cache_dir)
            sampler = RandomSampler(dataset) if name == "train" else SequentialSampler(dataset)
            dataloaders[name] = dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.gpu_batch_size,
                                                        num_workers=args.data_num_workers)
    else:
        raise ValueError(args.task)
    return datasets, dataloaders

def build_optimizers(args, tokenizer, model):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    model_params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, # TODO: if we want weight decay
        {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    if args.vopt:
        optimizer_grouped_parameters.append({'params': tokenizer.parameters(), 'weight_decay': 0.0, 'lr': args.weights_learning_rate})
    optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / args.warmup_epochs , end_factor=1, total_iters=args.warmup_epochs)
    return optimizer, warmup_scheduler


def regulairzation(args, tokenizer, model, lengths, entropic_weight, ent, device="cpu"):
    l1 = torch.zeros((1,), device=device)
    e = torch.zeros((1,), device=device)
    if args.l1 > 0:
        if not tokenizer.lsp:
            l1 = args.l1 * tokenizer.weights.weight.mean()
        else:
            l1 = args.l1 * tokenizer.weights.weight.exp().mean()
    if args.vopt and entropic_weight > 0:
        nchars = lengths.sum(-1)
        avg_ent = ent / nchars
        e = entropic_weight * avg_ent.mean()
    return l1, e

def save_checkpoint(args, epoch, step, model, tokenizer):
    checkpointdir = os.path.join(args.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpointdir, exist_ok=True)
    # Save a trained model
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(checkpointdir, "pytorch_model.bin")
    output_config_file = os.path.join(checkpointdir, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    output_vocab_file = os.path.join(checkpointdir, "learned_vocab.txt")
    weights = tokenizer.weights.weight.detach().tolist() if tokenizer.lsp else tokenizer.weights.weight.log().detach().tolist()
    save_weights(OrderedDict(zip(tokenizer.vocab, weights)), output_vocab_file)

def train(args, model: BertForMaskedLM, tokenizer:Tokenizer, train_dataloader: DataLoader,eval_dataloader: DataLoader, optimizer, lr_scheduler, device="cpu"):
    logger.info("Training...")
    bn = 0
    step = 0
    for epoch in range(args.train_epochs):
        entropic_weight = args.entropic * max(0, min(1, (epoch - args.entropy_start) / (args.entropy_end - args.entropy_start)))
        weight = args.gpu_batch_size / args.train_batch_size # TODO: if gpu_batch_size approaches the size of the dataset, make sure to drop_last
        epoch_loss = epoch_l1 = epoch_e =  epoch_examples = 0
        tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader) * args.train_epochs, initial=epoch * len(train_dataloader))
        for batch in tqdm_bar:
            bn += 1
            # load inputs
            batch_size = batch[0].size(0)

            if args.task == "morpheme_prediction":
                loss, ent, lengths, ntokens = morpheme_prediction_step(args, batch, tokenizer, model, device)
            elif args.task == "language_modeling":
                loss, ent, lengths, ntokens = language_modeling_step(args, batch, tokenizer, model, device)
            else:
                raise ValueError

            # get regularizations
            l1, e = regulairzation(args, tokenizer, model, lengths, entropic_weight, ent, device=device)

            # weight the loss and backpropograte
            Li = weight * (loss + l1 + e)
            Li.backward()

            # step
            if (bn + 1) % ( args.train_batch_size // args.gpu_batch_size) == 0:
                # clip grad
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_((param for param in group['params']), args.max_grad_norm)
                optimizer.step()
                if not tokenizer.lsp:
                    # make sure weights are positive if parametrized as real numbers
                    tokenizer.clamp_weights()
                optimizer.zero_grad(set_to_none=True)
                tokenizer.reset_padding_weight()
                tokenizer.reset_specials_weight()
                tokenizer.reset_singleton_weight()
                step += 1

                # evaluate
                if (step % 100) == 0:
                    if args.task == "morpheme_prediction":
                        pass
                    elif args.task == "language_modeling":
                        eval_loss_avg_c, eval_loss_avg_t, eval_loss, eval_NC, eval_NT = language_modeling_loop(args, eval_dataloader, tokenizer,
                                                                                  model, device)
                        print(f"Eval loss at step {step}: avgc = {eval_loss_avg_c}, avgt = {eval_loss_avg_t}, loss = {eval_loss}, NC = {eval_NC}, NT = {eval_NT}")
                        with open(os.path.join(args.output_dir, "log.json"), "a") as f:
                            print(json.dumps({
                                "step": step,
                                "avg_char": eval_loss_avg_c,
                                "avg_token": eval_loss_avg_t,
                                "loss": eval_loss,
                                "n_char": eval_NC,
                                "n_token": eval_NT
                            }), file=f)
                    else:
                        raise ValueError
                    save_checkpoint(args, epoch, step, model, tokenizer)

            # bookkeep
            epoch_examples += batch_size
            epoch_loss += loss * batch_size
            epoch_l1 += l1.item() * batch_size
            epoch_e += e.item() * batch_size
            tqdm_bar.desc = f"Epoch {epoch:<4} " \
                            f"Step {step:<4} " \
                            f"Task {epoch_loss / epoch_examples:<4.2f} " \
                            f"L1 {epoch_l1 / epoch_examples:<6.4f} " \
                            f"Ent {epoch_e / epoch_examples:<6.4f}"
        if (epoch + 1) % args.save_epochs == 0:
            save_checkpoint(args, epoch, step, model, tokenizer)
        lr_scheduler.step()

def eval(args, model: BertForMaskedLM, tokenizer:Tokenizer, eval_dataloader: DataLoader, device="cpu"):
    if args.task == "morpheme_prediction":
        pass
    elif args.task == "language_modeling":
        eval_loss_avg, eval_loss, eval_N = language_modeling_loop(args, eval_dataloader, tokenizer, model, device)
        print(f"Eval loss: avg = {eval_loss_avg}, loss = {eval_loss}, N = {eval_N}")

def main():
    args = parse_args()

    # initialize experiment
    device, n_gpu = initialize(args)

    # load labels, vocab, and weights from cmdline arguments
    input_vocab, output_vocab, weights = load_vocab_and_weights(args)

    # load model from path / config file
    model, config = load_model(args, device)

    # build tokenizer
    tokenizer = load_tokenizer(args, input_vocab, device)

    # do some logging of model size
    model_size = sum(parameter.numel() for parameter in model.parameters())
    tokenizer_size = sum(parameter.numel() for parameter in tokenizer.parameters())
    logger.info(
        f"Loaded transformer model with {model_size} parameters and vocab weight with {tokenizer_size} parameters, "
        f"percentage of weight among all parameters weights is {tokenizer_size / (tokenizer_size + model_size):e}")

    #input("Please hit enter if you want to overwrite the directory...")
    with open(os.path.join(args.output_dir, "log.json"), "wt") as f:
        pass

    # build datasets
    datasets, dataloaders = preprocess_datasets(args, tokenizer, input_vocab, output_vocab)

    # build optimizers
    optimizer, lr_scheduler = build_optimizers(args, tokenizer, model)

    # train!
    if args.do_train:
        logger.info("Training...")
        train(args, model, tokenizer, dataloaders["train"], dataloaders["eval"], optimizer, lr_scheduler, device=device)
    #    code.interact(local=locals())
    if args.do_eval:
        logger.info("Evaluating...")
        eval(args, model, tokenizer, dataloaders["eval"], device=device)
    #    code.interact(local=locals())



if __name__ == "__main__":
    torch.set_printoptions(profile="full", sci_mode=False, precision=2, linewidth=200)
    main()
    # cProfile.run("main()")
