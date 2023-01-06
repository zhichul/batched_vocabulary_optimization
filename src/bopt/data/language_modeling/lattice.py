import code

from tqdm import tqdm
from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.data.datasets import LazyDataset
from bopt.data.language_modeling.utils import viterbi_tokenize, pack_viterbi_chunks
from bopt.data.language_modeling.utils import clear_cache, pretokenize, truncated_and_pad_packed_chunks

import os
import pickle
import torch

"""
MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2
"""

def preprocess_language_modeling_with_lattices_dataset(
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

    with open(data_file, encoding=encoding) as text_file:
        for i, line in enumerate(tqdm(text_file)):
            text_str = line.strip()
            input_tokens = pretokenize(text_str)

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = [len(chunk) for chunk in kept_chunks]

            # encode the chunks into lattice / serial versions
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(kept_chunks, max_unit_length, max_block_length, compact=True)
            ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids = input_tokenizer.integerize_packed_chunks(kept_chunks, max_unit_length, max_block_length)
            binary_mask = torch.cat([mask, lm_mask], 0)

            # save to cache dir
            item_name = os.path.join(cache_dir, f"{i}.pkl")

            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": ids.tolist() + lm_ids.tolist(),
                     "pos_ids": pos_ids.tolist() + lm_pos_ids.tolist(),
                     "input_mask": mask.tolist() + lm_mask.tolist(),
                     "fwd_ids": fwd_ids.tolist(),
                     "fwd_ms": fwd_ms.tolist(),
                     "lengths": lengths.tolist(), # number of characters for each chunk
                     "ntokens": ntokens, # number of tokens for each chunk
                     "bwd_ids": bwd_ids.tolist(),
                     "bwd_ms": bwd_ms.tolist(),
                     "bwd_lengths": bwd_lengths.tolist(),
                     "mmask": mmask.tolist(),
                     "emask": emask.tolist(),
                     "binary_mask": binary_mask.tolist(),
                     "text_str": text_str,
                     "text": text_str,
                     }, file=f)

def preprocess_language_modeling_with_viterbi_lattices_dataset(
                    data_file: str,
                    cache_dir: str,
                    input_tokenizer: Tokenizer,
                    output_vocab: Integerizer,
                    max_blocks: int = None,
                    max_block_length: int = None,
                    max_unit_length: int = None,
                    encoding: str = "utf-8"):
    # transfer to CPU
    device = input_tokenizer.weights.weight.device
    input_tokenizer.to("cpu")

    clear_cache(cache_dir)
    with open(data_file, encoding=encoding) as text_file:
        for i, line in enumerate(tqdm(text_file)):
            text_str = line.strip()
            input_tokens = pretokenize(text_str)

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = [len(chunk) for chunk in kept_chunks]

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, input_tokens)
            viterbi_chunks, _ = pack_viterbi_chunks(kept_chunks, input_tokenizations)

            # encode the chunks into lattice / serial versions
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(viterbi_chunks, max_unit_length, max_block_length, compact=True, verbatim=True)
            ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids = input_tokenizer.integerize_packed_chunks(kept_chunks, max_unit_length, max_block_length)
            binary_mask = torch.cat([mask, lm_mask], 0)

            # save to cache dir
            item_name = os.path.join(cache_dir, f"{i}.pkl")

            with open(item_name, "wb") as f:
                pickle.dump({"input_ids": ids.tolist() + lm_ids.tolist(),
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
    # transfer back to device
    input_tokenizer.to(device)


def preprocess_language_modeling_with_lattices_output_viterbi_dataset(args, data_file: str,
                                                       cache_dir: str,
                                                       input_tokenizer: Tokenizer,
                                                       output_vocab: Integerizer,
                                                       max_blocks: int = None,
                                                       max_block_length: int = None,
                                                       max_unit_length: int = None):
    device = input_tokenizer.weights.weight.device
    input_tokenizer.to("cpu")
    clear_cache(cache_dir)
    with open(data_file, encoding='utf_8') as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = pretokenize(text_str)

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = [len(chunk) for chunk in kept_chunks]

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, input_tokens)
            viterbi_chunks, _ = pack_viterbi_chunks(kept_chunks, input_tokenizations)

            # encode the chunks into lattice / serial versions
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(kept_chunks, max_unit_length, max_block_length, compact=True)
            ids, mask, pos_ids, lm_ids, lm_mask, lm_pos_ids = input_tokenizer.integerize_packed_chunks(kept_chunks, max_unit_length, max_block_length)
            binary_mask = torch.cat([mask, lm_mask], 0)

            viterbi_fwd_ids, viterbi_fwd_ms, _, viterbi_bwd_ids, viterbi_bwd_ms, _, _, _ = input_tokenizer.encode_packed_batch(viterbi_chunks, max_unit_length, max_block_length, compact=True, verbatim=True)

            item_name = os.path.join(cache_dir, f"{i}.pkl")

            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": ids.tolist() + lm_ids.tolist(),
                     "pos_ids": pos_ids.tolist() + lm_pos_ids.tolist(),
                     "input_mask": mask.tolist() + lm_mask.tolist(),
                     "text": text_str,
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
                     "viterbi_fwd_ids": viterbi_fwd_ids.tolist(),
                     "viterbi_fwd_ms": viterbi_fwd_ms.tolist(),
                     "viterbi_bwd_ids": viterbi_bwd_ids.tolist(),
                     "viterbi_bwd_ms": viterbi_bwd_ms.tolist()
                     }, file=f)
    input_tokenizer.to(device)


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


class LanguageModelingLatticeOutputViterbiDataset(LazyDataset):

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
                torch.LongTensor(ex["viterbi_fwd_ids"]),
                torch.FloatTensor(ex["viterbi_fwd_ms"]),
                torch.LongTensor(ex["viterbi_bwd_ids"]),
                torch.FloatTensor(ex["viterbi_bwd_ms"]),
                torch.LongTensor(ex["mmask"]),
                torch.LongTensor(ex["emask"]),
                torch.FloatTensor(ex["binary_mask"]),
                ex["text_str"],
                torch.LongTensor(ex["ntokens"]))
