import csv
import os
import pickle
from pathlib import Path

import code
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

def preprocess_language_modeling_with_lattices_dataset(data_file: str,
                   cache_dir: str,
                   input_tokenizer: Tokenizer,
                   output_vocab: Integerizer,
                   max_blocks: int = None,
                   max_block_length: int = None,
                   max_unit_length: int = None):
    for f in glob.glob(f'{cache_dir}/*'):
        os.remove(f)
    with open(data_file, encoding='utf_8') as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = ["[BOS]"] + text_str.split(" ") + ["[EOS]"]
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            if len(packed_chunks) > max_blocks:
                print(f"[WARNING] Truncating {packed_chunks} to {' '.join(sum(packed_chunks[:max_blocks], []))}")
                packed_chunks = packed_chunks[:max_blocks]
            elif len(packed_chunks) < max_blocks:
                packed_chunks += [[]] * (max_blocks - len(packed_chunks))
            ntokens = []
            for packed_chunk in packed_chunks:
                ntokens.append(len(packed_chunk))
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(packed_chunks, max_unit_length, max_block_length, compact=True)

            ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids = input_tokenizer.integerize_packed_chunks(packed_chunks, max_unit_length, max_block_length)
            item_name = os.path.join(cache_dir, f"{i}.pkl")
            binary_mask = torch.cat([mask, lm_mask], 0)
            # causal_mask = input_tokenizer.causal_mask(max_blocks, max_block_length, max_unit_length)
            # code.interact(local=locals())
            # for t in (ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask, causal_mask, binary_mask):
            #     print(causal_mask.size())
            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": ids.tolist() + lm_ids.tolist(),
                     "pos_ids": pos_ids.tolist() + lm_pos_ids.tolist(),
                     "input_mask": mask.tolist() + lm_mask.tolist(),
                     "text":text_str,
                     "fwd_ids": fwd_ids.tolist(),
                     "fwd_ms": fwd_ms.tolist(),
                     "lengths": lengths.tolist(),
                     "ntokens": ntokens,
                     "bwd_ids": bwd_ids.tolist(),
                     "bwd_ms": bwd_ms.tolist(),
                     "bwd_lengths": bwd_lengths.tolist(),
                     "mmask": mmask.tolist(),
                     "emask": emask.tolist(),
                     "binary_mask": binary_mask.tolist(),
                     "text_str": text_str,
                     }, file=f)
            # code.interact(local=locals())



class LanguageModelingLatticeDataset(LazyDataset):

    def encode(self, ex, index):
        """
        All ids and masks are padded so every example should have same dimension.
        Valid Keys:
            "input_ids"
            "pos_ids"
            "input_mask"
            "text"
            "fwd_ids"
            "fwd_ms"
            "lengths"
            "bwd_ids"
            "bwd_ms"
            "bwd_lengths"
            "mmask"
            "emask"
            "binary_mask"
        """
        return (torch.LongTensor(ex["input_ids"]),
                torch.LongTensor(ex["pos_ids"]),
                torch.LongTensor(ex["input_mask"]),
                torch.LongTensor(ex["fwd_ids"]),
                torch.FloatTensor(ex["fwd_ms"]),
                torch.LongTensor(ex["lengths"]),
                torch.LongTensor(ex["bwd_ids"]),
                torch.FloatTensor(ex["bwd_ms"]),
                torch.LongTensor(ex["bwd_lengths"]),
                torch.LongTensor(ex["mmask"]),
                torch.LongTensor(ex["emask"]),
                torch.FloatTensor(ex["binary_mask"]),
                ex["text_str"],
                torch.LongTensor(ex["ntokens"]))
