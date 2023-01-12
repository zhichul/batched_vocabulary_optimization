import code
import csv
import os
import pickle
from pathlib import Path

import torch
import glob
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import RandomSampler, DataLoader

from tqdm import tqdm

from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.core.tokenizer.tokenization import TokenizationMixin
from bopt.data.datasets import LazyDataset
from bopt.data.language_modeling.utils import clear_cache, truncated_and_pad_packed_chunks, viterbi_tokenize, \
    pack_viterbi_chunks, use_gold_segmentations, load_segmentation_dictionary
from bopt.data.utils import load_vocab, load_weights, constant_initializer

MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2

TMASK_CACHE = dict()

def preprocess_morpheme_prediction_with_unigram_dataset(args, data_file: str,
                   cache_dir: str,
                   input_tokenizer: TokenizationMixin,
                   output_vocab: Integerizer,
                   max_blocks: int = None,
                   max_block_length: int = None,
                   max_unit_length: int = None,
                   max_length: int = None,
                   debug=False):
    clear_cache(cache_dir)
    total_tokens = 0
    replaced_tokens = 0
    is_gold_tokens = 0
    is_ambiguous_tokens = 0
    is_matching_tokens = 0
    if args.segmentation_dictionary is not None:
        segmentation_dictionary = load_segmentation_dictionary(*args.segmentation_dictionary)
    else:
        segmentation_dictionary = None

    ws = Whitespace()
    with open(data_file, encoding='utf_8') as csvfile:
        reader = csv.DictReader(csvfile,fieldnames=["id", "label", "text", "features", "segmentation"])
        for i, row in enumerate(tqdm(reader)):
            # pretokenize
            text_str = row["text"]
            input_tokens = [pair[0] for pair in ws.pre_tokenize_str(text_str)]
            output_labels = [output_vocab.index(tok, unk=True) for tok in row["features"].split("-")]
            input_tokens = ["[SP1]", "[SP2]", "[SP3]"] + input_tokens

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = sum([len(chunk) for chunk in kept_chunks])

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, input_tokens)
            input_tokenizations, is_gold, is_ambiguous, is_matching, replaced = use_gold_segmentations(input_tokenizer, input_tokens, input_tokenizations, segmentation_dictionary)
            viterbi_chunks, viterbi_tokens = pack_viterbi_chunks(kept_chunks, input_tokenizations)

            # bookkeeping
            assert viterbi_tokens == ntokens
            total_tokens += viterbi_tokens
            replaced_tokens += sum(replaced[:viterbi_tokens])
            is_gold_tokens += sum(is_gold[:viterbi_tokens])
            is_ambiguous_tokens += sum(is_ambiguous[:viterbi_tokens])
            is_matching_tokens += sum(is_matching[:viterbi_tokens])

            # encode the chunks into lattice / serial versions, and build label ids

            # integerize and pad if necessary
            input_ids = [input_tokenizer.vocab.index(subword_type) for chunk in viterbi_chunks for subword_type in chunk]
            if len(input_ids) <= max_length:
                input_ids += [input_tokenizer.pad_index] * (max_length - len(input_ids))
            else:
                raise ValueError(f"{text_str}\n{viterbi_chunks}\n{max_length}\n{input_ids}")

            # pos ids and mask
            pos_ids = list(range(max_length))
            mask = [int(id != input_tokenizer.pad_index) for id in input_ids]
            label_ids = [-100] * len(input_ids) # default value for ignore label
            for j, out_id in enumerate(output_labels):
                label_ids[j] = out_id

            item_name = os.path.join(cache_dir, f"{i}.pkl")
            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": input_ids,
                     "pos_ids": pos_ids,
                     "input_mask": mask,
                     "labels_ids": label_ids,
                     "text":text_str,
                }, file=f
                )

def tmask(max_blocks, max_unit_length, E):
    if (max_blocks, max_unit_length, E) not in TMASK_CACHE:
        task_mask = torch.ones((max_blocks * E, max_blocks * E))
        for k in range(3):
            task_mask[:, k * max_unit_length] = 0
            task_mask[k * max_unit_length, k * max_unit_length] = 1
        TMASK_CACHE[(max_blocks, max_unit_length, E)] = task_mask
    return TMASK_CACHE[(max_blocks, max_unit_length, E)]

class MorphemePredictionUnigramDataset(LazyDataset):


    def encode(self, ex, index):
        """
        All ids and masks are padded so every example should have same dimension.
        Valid Keys:
            "input_ids"
            "pos_ids"
            "input_mask"
            "labels_ids"
            "text"
        """
        return (torch.LongTensor(ex["input_ids"]),
                torch.LongTensor(ex["pos_ids"]),
                torch.LongTensor(ex["input_mask"]),
                torch.LongTensor(ex["labels_ids"]))
