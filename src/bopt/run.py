import glob
import math
import os
import sys
from collections import OrderedDict, defaultdict
from time import time
import cProfile

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from bopt.core.utils import length_normalized_initialization
from bopt.data.language_modeling.unigram import preprocess_language_modeling_with_unigram_dataset, \
    LanguageModelingUnigramDataset, tokenize_language_modeling_with_unigram_dataset, \
    preprocess_language_modeling_with_unigram_node_dataset
from bopt.forward_step import morpheme_prediction_lattice_step, language_modeling_lattice_step, \
    language_modeling_unigram_step
from bopt.forward_loop import language_modeling_lattice_loop, language_modeling_unigram_loop, \
    language_modeling_lattice_decode_loop, language_modeling_unigram_decode_loop

from bopt.arguments import parse_args
from bopt.core.tokenizer import Tokenizer
from bopt.data.morpheme_prediction.lattice import preprocess_morpheme_prediction_with_lattices_dataset, \
    MorphemePredictionLatticeDataset
from bopt.data.language_modeling.lattice import LanguageModelingLatticeDataset, \
    preprocess_language_modeling_with_lattices_dataset, preprocess_language_modeling_with_viterbi_lattices_dataset, \
    preprocess_language_modeling_with_lattices_output_viterbi_dataset, LanguageModelingLatticeOutputViterbiDataset
from grid_utils import acquire_all_available_gpu
import logging
import torch
import random
import numpy as np
import code
from bopt.core.modeling_bert import BertForMaskedLM, BertConfig
from bopt.data.utils import load_vocab, load_weights, constant_initializer, save_weights
import json

DEBUG = False
INF = 1e9

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    stream=sys.stderr,
                    level = logging.INFO)
logger = logging.getLogger(__name__)

acquire_all_available_gpu()

# torch.set_anomaly_enabled(True)

DEBUG = False

def default_collate_(batch):
    for i in range(len(batch[0])):
        for j in [27]:
            print(i, len(batch[j][i]), batch[j][i])
    return default_collate(batch)

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
    model.bias_mode = args.bias_mode if args.vopt else "mult_then_renorm" # mult_then_renorm is for attention causal masking
    if args.no_pos:
        model.bert.embeddings.no_pos = True
    return model, config


def load_tokenizer(args, input_vocab, weights, device):
    logger.info("Building tokenizer...")
    tokenizer = Tokenizer(vocab=input_vocab,
                          weights=weights,
                          log_space_parametrization=False,
                          continuing_subword_prefix=args.continuing_subword_prefix,
                          pad_token="[PAD]",
                          max_unit_length=args.max_unit_length,
                          specials=args.specials,
                          )
    if args.vopt:
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
                if args.vopt:
                    if args.debug_viterbi_lattice:
                        preprocess_language_modeling_with_viterbi_lattices_dataset(data,
                                                                           cache_dir,
                                                                           tokenizer,
                                                                           output_vocab,
                                                                           args.max_blocks if name == "train" or args.eval_max_blocks is None else args.eval_max_blocks,
                                                                           args.max_block_length if name == "train" or args.eval_max_block_length is None else args.eval_max_block_length,
                                                                           args.max_unit_length if name == "train" or args.eval_max_unit_length is None else args.eval_max_unit_length)
                    elif args.output_viterbi:
                        preprocess_language_modeling_with_lattices_output_viterbi_dataset(args, data,
                                                                           cache_dir,
                                                                           tokenizer,
                                                                           output_vocab,
                                                                           args.max_blocks if name == "train" or args.eval_max_blocks is None else args.eval_max_blocks,
                                                                           args.max_block_length if name == "train" or args.eval_max_block_length is None else args.eval_max_block_length,
                                                                           args.max_unit_length if name == "train" or args.eval_max_unit_length is None else args.eval_max_unit_length)
                    else:
                        preprocess_language_modeling_with_lattices_dataset(data,
                                                                           cache_dir,
                                                                           tokenizer,
                                                                           output_vocab,
                                                                           args.max_blocks if name == "train" or args.eval_max_blocks is None else args.eval_max_blocks,
                                                                           args.max_block_length if name == "train" or args.eval_max_block_length is None else args.eval_max_block_length,
                                                                           args.max_unit_length if name == "train" or args.eval_max_unit_length is None else args.eval_max_unit_length)
                else:
                    if args.debug_node_unigram:
                        preprocess_language_modeling_with_unigram_node_dataset(data,
                                                                      cache_dir,
                                                                      tokenizer,
                                                                      output_vocab,
                                                                      args.max_blocks if name == "train" or args.eval_max_blocks is None else args.eval_max_blocks,
                                                                      args.max_block_length if name == "train" or args.eval_max_block_length is None else args.eval_max_block_length,
                                                                      args.max_unit_length if name == "train" or args.eval_max_unit_length is None else args.eval_max_unit_length,
                                                                      args.max_length if name == "train" or args.eval_max_length is None else args.eval_max_length,
                                                                      pos_length=args.pos_length)
                    else:
                        preprocess_language_modeling_with_unigram_dataset(data,
                                                                      cache_dir,
                                                                      tokenizer,
                                                                      output_vocab,
                                                                      args.max_blocks if name == "train" or args.eval_max_blocks is None else args.eval_max_blocks,
                                                                      args.max_block_length if name == "train" or args.eval_max_block_length is None else args.eval_max_block_length,
                                                                      args.max_unit_length if name == "train" or args.eval_max_unit_length is None else args.eval_max_unit_length,
                                                                      args.max_length if name == "train" or args.eval_max_length is None else args.eval_max_length)
            if args.vopt:
                if args.output_viterbi:
                    datasets[name] = dataset = LanguageModelingLatticeOutputViterbiDataset(cache_dir)
                else:
                    datasets[name] = dataset = LanguageModelingLatticeDataset(cache_dir)
            else:
                datasets[name] = dataset = LanguageModelingUnigramDataset(cache_dir)
            sampler = RandomSampler(dataset) if name == "train" else SequentialSampler(dataset)
            dataloaders[name] = dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.gpu_batch_size if name == "train" or args.eval_gpu_batch_size is None else args.eval_gpu_batch_size,
                                                        num_workers=args.data_num_workers, collate_fn=default_collate)
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
    plat_scheduler = ReduceLROnPlateau(optimizer,'min', factor=1/4 )
    return optimizer, plat_scheduler


class Regularizers:

    @classmethod
    def regulairzation(cls, args, tokenizer, model, lengths, entropic_weight, ent, out_marginals, out_units, log_marginal_counts, counts, prev_lmc, prev_c, lmc, device="cpu"):
        l1 = torch.zeros((1,), device=device)
        e = torch.zeros((1,), device=device)
        gl = torch.zeros((1,), device=device)
        if args.l1 > 0:
            if not tokenizer.lsp:
                l1 = args.l1 * tokenizer.weights.weight.mean()
            else:
                l1 = args.l1 * tokenizer.weights.weight.exp().mean()
        if args.vopt and entropic_weight != 0:
            nchars = lengths.sum(-1)
            avg_ent = ent / nchars
            e = entropic_weight * avg_ent.mean()
        if args.group_lasso > 0:
            # if cls.log_marginal_counts is None:
            #     cls.log_marginal_counts = torch.ones((len(tokenizer.vocab),), dtype=torch.float)
            for om, ou in zip(out_marginals.detach().cpu(), out_units.detach().cpu()):
                log_marginal_counts[ou] = torch.logaddexp(log_marginal_counts[ou], 2 * om) # the 2 is for the squaring
                lmc[ou] = torch.logaddexp(lmc[ou], om)
                counts[ou] = counts[ou] + 1.0
            if prev_lmc is not None:
                try:
                    # d/d out_marginals (group lasso) = lambda * sqrt(group_size) / sqrt(sum of squared marginals) * 2 exp om * exp om
                    log_global_multiplier = (prev_c.sqrt().log() - log_marginal_counts/2).to(device)
                    log_individual_multiplier = log_global_multiplier[out_units] + 2 * out_marginals.detach()
                    gl = args.group_lasso * (log_individual_multiplier.exp().to(device) * out_marginals).sum()
                except Exception as ex:
                    code.interact(local=locals())
        return l1, e, gl

def save_checkpoint(args, epoch, step, model, tokenizer, optimizer):
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
    output_optim_file = os.path.join(checkpointdir, "optim.bin")
    torch.save({"optimizer_state_dict": optimizer.state_dict()}, output_optim_file)

def train(args, model: BertForMaskedLM, tokenizer:Tokenizer, train_dataloader: DataLoader,eval_dataloader: DataLoader, optimizer, lr_scheduler, device="cpu"):
    logger.info("Training...")
    bn = 0
    step = 0
    entropic_weight = 0
    prev_counts = prev_log_marginal_counts = prev_lmc = prev_type_ent = None
    for epoch in range(args.train_epochs):
        if args.entropic != 0:
            if epoch < args.entropy_start_dec:
                entropic_weight = args.entropic * max(0, min(1, (epoch - args.entropy_start) / (args.entropy_end - args.entropy_start)))
            else:
                entropic_weight = args.entropic * min(1, max(0,  1 - (epoch - args.entropy_start_dec) / (args.entropy_end_dec - args.entropy_start_dec)))
        weight = args.gpu_batch_size / args.train_batch_size # TODO: if gpu_batch_size approaches the size of the dataset, make sure to drop_last
        epoch_loss = epoch_l1 = epoch_e =  epoch_examples = epoch_gl = 0
        tqdm_bar = tqdm(train_dataloader, total=len(train_dataloader) * args.train_epochs, initial=epoch * len(train_dataloader))
        log_marginal_counts = torch.ones((len(tokenizer.vocab),), dtype=torch.float) * -INF
        lmc = torch.ones((len(tokenizer.vocab),), dtype=torch.float) * -INF
        counts = torch.zeros((len(tokenizer.vocab),), dtype=torch.float)
        for batch in tqdm_bar:
            # if not batch[-2][0].startswith("some changes to the plan"):
            #     continue

            bn += 1
            # load inputs
            batch_size = batch[0].size(0)

            if args.task == "morpheme_prediction":
                loss, ent, lengths, ntokens, out_marginals, out_units = morpheme_prediction_lattice_step(args, batch, tokenizer, model, device)
            elif args.task == "language_modeling":
                if args.vopt:
                    loss, ent, lengths, ntokens, out_marginals, out_units = language_modeling_lattice_step(args, batch, tokenizer, model, device)
                else:
                    loss, ent, lengths, ntokens, out_marginals, out_units = language_modeling_unigram_step(args, batch, tokenizer, model, device)
            else:
                raise ValueError

            # get regularizations
            l1, e, gl = Regularizers.regulairzation(args, tokenizer, model, lengths, entropic_weight, ent, out_marginals, out_units, log_marginal_counts, counts, prev_log_marginal_counts, prev_counts, lmc, device=device)

            # weight the loss and backpropograte
            Li = weight * (loss + l1 + e + gl)
            Li.backward()
            # code.interact(local=locals())
            # for group in optimizer.param_groups:
            #     for param in group["params"]:
            #         if param.grad.isnan().any():
            #             print("Nan gradient")
            #             code.interact(local=locals())

            # bookkeep
            epoch_examples += batch_size
            epoch_loss += loss.item() * batch_size

            epoch_l1 += l1.item() * batch_size
            epoch_gl += gl.item() * batch_size
            epoch_e += e.item() * batch_size
            tqdm_bar.desc = f"Epoch {epoch:<4} " \
                            f"Step {step:<4} " \
                            f"Task {epoch_loss / epoch_examples:<4.2f} " \
                            f"L1 {epoch_l1 / epoch_examples:<6.4f} " \
                            f"GL {epoch_gl / epoch_examples:<6.4f} " \
                            f"Ent {epoch_e / epoch_examples:<6.4f} " \
                            f"TEnt {-42.0 if prev_type_ent is None else prev_type_ent.item():<6.4f} " \
                            f"GnormM {(sum([(param.grad ** 2).sum() for param in list(model.parameters()) if param.grad is not None], torch.tensor(0, device=device))**0.5).item():<6.4f} " \
                            f"GnormV {(sum([(param.grad ** 2).sum() for param in list(tokenizer.parameters()) if param.grad is not None], torch.tensor(0, device=device)) ** 0.5).item():<6.4f} " \
                            f"LR " + " ".join([f"{param_group['lr']:<6.4f}" for param_group in optimizer.param_groups])
            # step
            if (bn + 1) % ( args.train_batch_size // args.gpu_batch_size) == 0:
                # clip grad
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_((param for param in group['params']), args.max_grad_norm)
                optimizer.step()
                if not tokenizer.lsp:
                    # make sure weights are positive if parametrized as real numbers
                    tokenizer.clamp_weights()
                sweight = tokenizer.get_singleton_weight()
                optimizer.zero_grad(set_to_none=False)
                tokenizer.reset_padding_weight()
                tokenizer.reset_specials_weight()
                tokenizer.set_singleton_weight(sweight)
                step += 1
                if DEBUG:
                    code.interact(local=locals())
                # evaluate
                if (step % args.eval_steps) == 0:
                    if args.task == "morpheme_prediction":
                        pass
                    elif args.task == "language_modeling":
                        if args.vopt:
                            eval_loss_avg_c, eval_loss_avg_t, eval_loss, eval_NC, eval_NT = language_modeling_lattice_loop(args, eval_dataloader, tokenizer,
                                                                                                                       model, device)
                        else:
                            eval_loss_avg_c, eval_loss_avg_t, eval_loss, eval_NC, eval_NT = language_modeling_unigram_loop(args, eval_dataloader,
                                                                                              tokenizer, model, device)
                        logger.info(f"Eval loss at step {step}: avgc = {eval_loss_avg_c}, avgt = {eval_loss_avg_t}, loss = {eval_loss}, NC = {eval_NC}, NT = {eval_NT}")
                        with open(os.path.join(args.output_dir, "log.json"), "a") as f:
                            print(json.dumps({
                                "step": step,
                                "avg_char": eval_loss_avg_c,
                                "avg_token": eval_loss_avg_t,
                                "eval_loss": eval_loss,
                                "n_char": eval_NC,
                                "n_token": eval_NT,
                                "train_loss": epoch_loss / epoch_examples,
                                "train_ent": epoch_e / epoch_examples,
                                "train_l1": epoch_l1 / epoch_examples,
                                "group_lasso": epoch_gl / epoch_examples,
                                "type_entropy": -42.0 if prev_type_ent is None else prev_type_ent.item()
                            }), file=f)
                    else:
                        raise ValueError
                if (step % args.save_steps) == 0:
                    save_checkpoint(args, epoch, step, model, tokenizer, optimizer)
        prev_log_marginal_counts = log_marginal_counts
        prev_counts = counts
        prev_lmc = lmc
        prev_type_ent = (-(prev_lmc - prev_lmc.logsumexp(-1)).to(torch.double) * (prev_lmc - prev_lmc.logsumexp(-1)).to(torch.double).exp()).sum()

        if (epoch + 1) % args.save_epochs == 0:
            save_checkpoint(args, epoch, step, model, tokenizer, optimizer)
        lr_scheduler.step(epoch_loss / epoch_examples)

def eval(args, model: BertForMaskedLM, tokenizer:Tokenizer, eval_dataloader: DataLoader, device="cpu"):
    if args.task == "morpheme_prediction":
        pass
    elif args.task == "language_modeling":
        if args.vopt:
            eval_loss_avg_c, eval_loss_avg_t, eval_loss, eval_NC, eval_NT = language_modeling_lattice_loop(args, eval_dataloader, tokenizer, model, device)
            logger.info(f"Eval loss: avgc = {eval_loss_avg_c}, avgt = {eval_loss_avg_t}, loss = {eval_loss}, NC = {eval_NC}, NT = {eval_NT}")
        else:
            eval_loss_avg_c, eval_loss_avg_t, eval_loss, eval_NC, eval_NT = language_modeling_unigram_loop(args, eval_dataloader, tokenizer, model, device)
            logger.info(f"Eval loss: avgc = {eval_loss_avg_c}, avgt = {eval_loss_avg_t}, loss = {eval_loss}, NC = {eval_NC}, NT = {eval_NT}")


def decode(args, model, tokenizer, eval_dataloader, device="cpu"):
    if args.task == "morpheme_prediction":
        pass
    elif args.task == "language_modeling":
        if args.vopt:
            decodings = language_modeling_lattice_decode_loop(args, eval_dataloader, tokenizer, model, device, remove_csp=args.decode_remove_csp, remove_padding=args.decode_remove_padding)
        else:
            decodings = language_modeling_unigram_decode_loop(args, eval_dataloader, tokenizer, model, device, remove_csp=args.decode_remove_csp, remove_padding=args.decode_remove_padding)
        with open(os.path.join(args.output_dir, f"{os.path.basename(args.eval_dataset)}.viterbi.txt"), "wt") as f:
            for decoding in decodings:
                print(" ".join(decoding), file=f)
def tokenize(args, tokenizer):
    if args.task == "language_modeling":
        tokenize_language_modeling_with_unigram_dataset(args.eval_dataset, tokenizer)
    else:
        raise ValueError

def main():
    args = parse_args()

    # initialize experiment
    device, n_gpu = initialize(args)
    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    # load labels, vocab, and weights from cmdline arguments
    input_vocab, output_vocab, weights = load_vocab_and_weights(args)

    # build tokenizer
    tokenizer = load_tokenizer(args, input_vocab, weights, device)

    if args.do_tokenize:
        logger.info("Tokenizing...")
        # toknenizing is a separate mode from training that only uses the tokenizer
        tokenize(args, tokenizer)
        return None

    # load model from path / config file
    model, config = load_model(args, device)
    if args.model_name is None and args.length_normalized_initialization:
        length_normalized_initialization(model, tokenizer)
    # do some logging of model size
    model_size = sum(parameter.numel() for parameter in model.parameters())
    tokenizer_size = sum(parameter.numel() for parameter in tokenizer.parameters())
    logger.info(
        f"Loaded transformer model with {model_size} parameters and vocab weight with {tokenizer_size} parameters, "
        f"percentage of weight among all parameters weights is {tokenizer_size / (tokenizer_size + model_size):e}")

    if args.do_train:
        if not args.quiet:
            input("Please hit enter if you want to overwrite the directory (esp. log.json)...")
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
        model.eval()
        eval(args, model, tokenizer, dataloaders["eval"], device=device)
    #    code.interact(local=locals())
    if args.do_decode:
        logger.info("Decoding...")
        decode(args, model, tokenizer, dataloaders["eval"], device=device)
    if args.do_inspection:
        logger.info("Decoding...")
        code.interact(local=locals())


if __name__ == "__main__":
    torch.set_printoptions(profile="full", sci_mode=False, precision=2, linewidth=200)
    main()
    # cProfile.run("main()")
