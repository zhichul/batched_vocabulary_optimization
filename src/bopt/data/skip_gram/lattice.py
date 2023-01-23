import code
import json
from collections import defaultdict

from tqdm import tqdm
from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.data.datasets import LazySkipGramDataset
from bopt.data.language_modeling.utils import clear_cache

import os
import pickle
import torch

from bopt.data.skip_gram.utils import sft

"""
MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2
"""

def preprocess_skip_gram_with_lattices_dataset(args,
                    data_file: str,
                    cache_dir: str,
                    input_tokenizer: Tokenizer,
                    output_vocab: Integerizer,
                    max_blocks: int = None,
                    max_block_length: int = None,
                    max_unit_length: int = None,
                    encoding: str = "utf-8"):
    """
    Computes the lattice representation and metadata of each sentence, and store it in a pickled file in a cache dir.
    This method always clears the cache dir FIRST before writing anything into it.

    Args:
        data_file: text file where each line is one sentence
        cache_dir: full path to cache directory, e.g. /home/blu/.../cache/ptb.train.txt/
        input_tokenizer: the tokenizer object (for converting a sentence into a lattice and integerizing).
        output_vocab: for integerizing the output
        max_blocks: the input words are packed into blocks of equal number of characters (padded if necessary)
        max_block_length: each lattice accounts for maximum of L characters
        max_unit_length: the maximum edge length size within each block
        encoding: the format of data_fiile
    Returns:

    """
    clear_cache(cache_dir)
    skip_gram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    words = set()
    with open(data_file, encoding=encoding) as text_file:
        # get skip-gram statistics as well as all the words
        for i, line in enumerate(tqdm(text_file)):
            text_str = line.strip()
            input_tokens = text_str.split(" ")
            for i in range(len(input_tokens)):
                for j in args.skip_gram_distances:
                    if i + j < len(input_tokens):
                        skip_gram_counts[j][input_tokens[i]][input_tokens[i + j]] += 1
                words.add(input_tokens[i])
        words = sorted(list(words))
        for i, word in enumerate(tqdm(words)):
            # encode the words into lattice / serial versions
            packed_chunks = input_tokenizer.pack_chunks([word], max_block_length)
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(packed_chunks, max_unit_length, max_block_length, compact=True)
            ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids = input_tokenizer.integerize_packed_chunks(packed_chunks, max_unit_length, max_block_length)
            # binary_mask = torch.cat([mask, lm_mask], 0)

            # save to cache dir
            item_name = os.path.join(cache_dir, f"{i}.pkl")

            with open(item_name, "wb") as f:
                pickle.dump(
                    {"word_ids": ids.tolist(),
                     "lm_ids": lm_ids.tolist(),
                     "word_pos_ids": pos_ids.tolist(),
                     "lm_pos_ids": lm_pos_ids.tolist(),
                     "word_mask": mask.tolist(),
                     "lm_mask": lm_mask.tolist(),
                     "fwd_ids": fwd_ids.tolist(),
                     "fwd_ms": fwd_ms.tolist(),
                     "lengths": lengths.tolist(), # number of characters for each chunk
                     "bwd_ids": bwd_ids.tolist(),
                     "bwd_ms": bwd_ms.tolist(),
                     "bwd_lengths": bwd_lengths.tolist(),
                     "word": word,
                     }, file=f)
        with open(f"{cache_dir}.index.json", "wt") as index_file:
            json.dump({"counts": skip_gram_counts,
                       "words": words,
                       "total": sum(count for dist in skip_gram_counts for src in skip_gram_counts[dist] for tgt, count in skip_gram_counts[dist][src].items())},
                      index_file)


class SkipGramLatticeDataset(LazySkipGramDataset):

    def encode(self, ex, index):
        dist, src, tgt, i, max_block_length = index
        return (torch.LongTensor(ex["src"]["word_ids"] + ex["tgt"]["word_ids"] + ex["src"]["lm_ids"] + ex["tgt"]["lm_ids"]),
                torch.LongTensor(ex["src"]["word_pos_ids"] + sft(ex["tgt"]["word_pos_ids"], max_block_length) + ex["src"]["lm_pos_ids"] + sft(ex["tgt"]["lm_pos_ids"], max_block_length)),
                torch.LongTensor(ex["src"]["word_mask"] + ex["tgt"]["word_mask"] + ex["src"]["lm_mask"] + ex["tgt"]["lm_mask"]),
                torch.LongTensor(ex["src"]["fwd_ids"] + ex["tgt"]["fwd_ids"]),
                torch.LongTensor(ex["src"]["fwd_ms"] + ex["tgt"]["fwd_ms"]),
                torch.LongTensor(ex["src"]["lengths"] + [ex["tgt"]["lengths"][0]]),
                torch.LongTensor(ex["src"]["bwd_ids"] + ex["tgt"]["bwd_ids"]),
                torch.LongTensor(ex["src"]["bwd_ms"] + ex["tgt"]["bwd_ms"]),
                torch.LongTensor(ex["src"]["bwd_lengths"] + ex["tgt"]["bwd_lengths"]),
                f'{ex["src"]["word"]} {ex["tgt"]["word"]}')