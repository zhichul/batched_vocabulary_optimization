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
from bopt.data.language_modeling.utils import clear_cache, truncated_and_pad_packed_chunks
from bopt.data.utils import load_vocab, load_weights, constant_initializer

MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2

TMASK_CACHE = dict()

def preprocess_sentiment_analysis_with_lattices_dataset(args,
                   data_file: str,
                   cache_dir: str,
                   input_tokenizer: TokenizationMixin,
                   output_vocab: Integerizer,
                   max_blocks: int = None,
                   max_block_length: int = None,
                   max_unit_length: int = None,
                   debug=False):
    clear_cache(cache_dir)

    E = (max_block_length * (max_block_length + 1)) // 2 - ((max_block_length - max_unit_length) * (max_block_length - max_unit_length + 1)) // 2
    ws = Whitespace()
    with open(data_file, encoding='utf_8' if not args.encoding else args.encoding) as csvfile:
        reader = csv.DictReader(csvfile,fieldnames=["label", "text"])
        for i, row in enumerate(tqdm(reader)):
            # pretokenize
            text_str = row["text"]
            input_tokens = [pair[0] for pair in ws.pre_tokenize_str(text_str)]
            output_labels = [output_vocab.index(row["label"], unk=True)]
            input_tokens = ["[SP1]"] + input_tokens

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)

            # encode the chunks into lattice / serial versions, and build label ids
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(kept_chunks, max_unit_length, max_block_length, compact=True)
            ids, mask, pos_ids, _, _, _ = input_tokenizer.integerize_packed_chunks(kept_chunks, max_unit_length, max_block_length)
            label_ids: torch.LongTensor = torch.ones_like(ids, dtype=torch.long) * -100 # default value for ignore label
            for j, out_id in enumerate(output_labels):
                label_ids[j * max_unit_length] = out_id

            ## FOR DEBUGGING ONLY ###
            if debug:
                print()
                for i in range(max_blocks):
                    offset = 0
                    for j in range(max_block_length):
                        for l in range(min(max_unit_length, max_block_length - j)):
                            print(f"{input_tokenizer.vocab[ids[i*E + offset + l]] if mask[i*E + offset + l] else '':>10s} "
                                  f"{pos_ids[i*E + offset + l] if mask[i*E + offset + l] else '':<2} "
                                  f"{(': '+ output_vocab[label_ids[i*E + offset + l]]) if label_ids[i*E + offset + l] >= 0 else '':<6s}", end=" ")
                        offset += min(max_unit_length, max_block_length - j)
                        print()
                    print()
                print("########")
            ## FOR DEBUGGING ONLY ###

            item_name = os.path.join(cache_dir, f"{i}.pkl")
            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": ids.tolist(),
                     "pos_ids": pos_ids.tolist(),
                     "input_mask": mask.tolist(),
                     "labels_ids": label_ids.tolist(),
                     "text":text_str,
                     "fwd_ids": fwd_ids.tolist(),
                     "fwd_ms": fwd_ms.tolist(),
                     "lengths": lengths.tolist(),
                     "bwd_ids": bwd_ids.tolist(),
                     "bwd_ms": bwd_ms.tolist(),
                     "bwd_lengths": bwd_lengths.tolist(),
                     "max_blocks": max_blocks,
                     "max_block_length": max_block_length,
                     "max_unit_length": max_unit_length,
                     "E": E
                }, file=f
                )

def tmask(max_blocks, max_unit_length, E):
    if (max_blocks, max_unit_length, E) not in TMASK_CACHE:
        task_mask = torch.ones((max_blocks * E, max_blocks * E))
        for k in range(1):
            task_mask[:, k * max_unit_length] = 0
            task_mask[k * max_unit_length, k * max_unit_length] = 1
        TMASK_CACHE[(max_blocks, max_unit_length, E)] = task_mask
    return TMASK_CACHE[(max_blocks, max_unit_length, E)]

class SentimentAnalysisLatticeDataset(LazyDataset):


    def encode(self, ex, index):
        """
        All ids and masks are padded so every example should have same dimension.
        Valid Keys:
            "input_ids"
            "pos_ids"
            "input_mask"
            "labels_ids"
            "text"
            "fwd_ids"
            "fwd_ms"
            "lengths"
            "bwd_ids"
            "bwd_ms"
            "bwd_lengths"
            # "mmask" these are dynamically re-computed to save cache size
            # "emask"
            # "tmask"
        """
        return (torch.LongTensor(ex["input_ids"]),
                torch.LongTensor(ex["pos_ids"]),
                torch.LongTensor(ex["input_mask"]),
                torch.LongTensor(ex["labels_ids"]),
                torch.LongTensor(ex["fwd_ids"]),
                torch.FloatTensor(ex["fwd_ms"]),
                torch.LongTensor(ex["lengths"]),
                torch.LongTensor(ex["bwd_ids"]),
                torch.FloatTensor(ex["bwd_ms"]),
                torch.LongTensor(ex["bwd_lengths"]),
                tmask(ex["max_blocks"], ex["max_unit_length"], ex["E"]),
                ex["text"]
                )