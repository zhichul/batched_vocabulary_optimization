import csv
import json
import os
import pickle
import sys
from pathlib import Path

import code
from typing import List

import torch
import glob
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import RandomSampler, DataLoader

from tqdm import tqdm

from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.core.tokenizer.tokenization import TokenizationMixin
from bopt.core.utils import increasing_roll_left
from bopt.data.datasets import LazyDataset
from bopt.data.utils import load_vocab, load_weights, constant_initializer

MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2

def preprocess_language_modeling_with_unigram_dataset(data_file: str,
                   cache_dir: str,
                   input_tokenizer: Tokenizer,
                   output_vocab: Integerizer,
                   max_length: int):
    for f in glob.glob(f'{cache_dir}/*'):
        os.remove(f)
    with open(data_file, encoding='utf_8') as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = ["[BOS]"] + text_str.split(" ") + ["[EOS]"]

            # viterbi segmentation
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_batch(input_tokens, input_tokenizer.max_unit_length)
            fwd_ts = input_tokenizer.get_weights(fwd_ids)
            _, _, backpointers = input_tokenizer.viterbi_algorithm(fwd_ts, fwd_ms, lengths)
            word_ids = input_tokenizer.decode_backpointers(fwd_ids, lengths, backpointers)

            # truncate if necessary
            input_ids = []
            for j, word_id in enumerate(word_ids):
                if len(input_ids) + len(word_id) > max_length:
                    print(f"[WARNING] Truncating {input_tokens} to {' '.join(sum(input_tokens[:i], []))}")
                    break
                input_ids.extend(word_id)
            input_subwords = [input_tokenizer.vocab[id] for id in input_ids]
            length = sum([input_tokenizer.len_type(token) for token in input_tokens[:j]])
            ntokens = j

            # pad if necessary
            if len(input_ids) < max_length:
                input_ids += [input_tokenizer.pad_index] * (max_length - len(input_ids))
                input_subwords += [input_tokenizer.pad_token] * (max_length - len(input_subwords))

            # pos ids, labels, and mask
            pos_ids = list(range(max_length))
            labels = [id if id != input_tokenizer.pad_index else -100 for id in input_ids[1:]] + [-100]
            mask = [int(id != input_tokenizer.pad_index) for id in input_ids]

            # log to file
            item_name = os.path.join(cache_dir, f"{i}.pkl")
            dd =  {"input_ids": input_ids,
                     "pos_ids": pos_ids,
                     "input_mask": mask,
                     "labels": labels,
                     "text": text_str,
                     "length": [length], # in terms of characters
                     "ntokens": [ntokens]
                     }
            # print(json.dumps(dd, indent=4))
            # code.interact(local=locals())
            with open(item_name, "wb") as f:
                pickle.dump(
                   dd, file=f)

def prefix_sum(l):
    cumsum = 0
    out = []
    for item in l:
        cumsum += item
        out.append(cumsum)
    return out

def preprocess_language_modeling_with_unigram_node_dataset(data_file: str,
                   cache_dir: str,
                   input_tokenizer: Tokenizer,
                   output_vocab: Integerizer,
                   max_length: int,
                   pos_length: bool = False):
    max_length = max_length // 2
    for f in glob.glob(f'{cache_dir}/*'):
        os.remove(f)
    with open(data_file, encoding='utf_8') as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = ["[BOS]"] + text_str.split(" ") + ["[EOS]"]

            # viterbi segmentation
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_batch(input_tokens, input_tokenizer.max_unit_length)
            fwd_ts = input_tokenizer.get_weights(fwd_ids)
            _, _, backpointers = input_tokenizer.viterbi_algorithm(fwd_ts, fwd_ms, lengths)
            word_ids = input_tokenizer.decode_backpointers(fwd_ids, lengths, backpointers)

            # truncate if necessary
            input_ids = []
            for j, word_id in enumerate(word_ids):
                if len(input_ids) + len(word_id) > max_length:
                    print(f"[WARNING] Truncating {input_tokens} to {' '.join(input_tokens[:j])}")
                    break
                input_ids.extend(word_id)
            input_subwords = [input_tokenizer.vocab[id] for id in input_ids]
            length = sum([input_tokenizer.len_type(token) for token in input_tokens[:j]])
            pos_increments = [0] + [input_tokenizer.len_type(token) for token in input_subwords][:-1]
            pos_ids = prefix_sum(pos_increments)
            if pos_length:
                pos_ids = pos_increments
            ntokens = j
            # if text_str.startswith("downgraded by moody 's were houston"):
            #     print(json.dumps(dd, indent=4))
            #     code.interact(local=locals())
            # pad if necessary
            if len(input_ids) < max_length:
                input_ids += [input_tokenizer.pad_index] * (max_length - len(input_ids))
                input_subwords += [input_tokenizer.pad_token] * (max_length - len(input_subwords))
                pos_ids += [0] * (max_length - len(pos_ids))

            # pos ids, labels, and mask
            labels = [id if id != input_tokenizer.pad_index else -100 for id in input_ids[1:]] + [-100]
            mask = [int(id != input_tokenizer.pad_index) for id in input_ids]

            # log to file
            item_name = os.path.join(cache_dir, f"{i}.pkl")
            dd = {"input_ids": input_ids + [input_tokenizer.node_index for _ in input_ids],
                     "pos_ids": pos_ids + [pos_id + 1 for pos_id in pos_ids],
                     "input_mask": mask + mask,
                     "labels": [-100] * len(labels) + labels,
                     "text": text_str,
                     "length": [length], # in terms of characters
                     "ntokens": [ntokens]
                     }
            # code.interact(local=locals())
            with open(item_name, "wb") as f:
                pickle.dump(
                    dd, file=f)



class LanguageModelingUnigramDataset(LazyDataset):

    def encode(self, ex, index):
        """
        All ids and masks are padded so every example should have same dimension.
        Valid Keys:
            "input_ids"
            "pos_ids"
            "input_mask"
            "labels"
            "text"
            "length"
            "ntokens"
        """
        ret =  (torch.LongTensor(ex["input_ids"]),
                torch.LongTensor(ex["pos_ids"]),
                torch.LongTensor(ex["input_mask"]),
                torch.LongTensor(ex["labels"]),
                torch.LongTensor(ex["length"]),
                torch.LongTensor(ex["ntokens"]),
                ex["text"])
        return ret

def viterbi_tokenize(tokenizer: Tokenizer, tokens: List[str]):
    fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = tokenizer.encode_batch(tokens, tokenizer.max_unit_length)
    fwd_ts = tokenizer.get_weights(fwd_ids)
    max_log_alpha, _, backpointers = tokenizer.viterbi_algorithm(fwd_ts, fwd_ms, lengths)
    log_alpha, _ = tokenizer.forward_algorithm(fwd_ts, fwd_ms, lengths)
    word_ids = tokenizer.decode_backpointers(fwd_ids, lengths, backpointers)
    return [[tokenizer.id2str(id, remove_csp=False) for id in word_id] for word_id in word_ids]

def tokenize_language_modeling_with_unigram_dataset(data_file: str, input_tokenizer: Tokenizer):
    with open(data_file, encoding='utf_8') as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = ["[BOS]"] + text_str.split(" ") + ["[EOS]"]

            # viterbi segmentation
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_batch(input_tokens, input_tokenizer.max_unit_length)
            fwd_ts = input_tokenizer.get_weights(fwd_ids)
            max_log_alpha, _, backpointers = input_tokenizer.viterbi_algorithm(fwd_ts, fwd_ms, lengths)
            log_alpha, _ = input_tokenizer.forward_algorithm(fwd_ts, fwd_ms, lengths)
            word_ids = input_tokenizer.decode_backpointers(fwd_ids, lengths, backpointers)
            input_ids = sum(word_ids, [])
            input_subwords = [input_tokenizer.id2str(id, remove_csp=True) for id in input_ids]
            print(f"{round((max_log_alpha - log_alpha).sum().item(), 2):.2f}\t{' '.join(input_subwords)}")
