import json

from tqdm import tqdm
from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.data.datasets import LazyDataset
from bopt.data.language_modeling.utils import clear_cache, viterbi_tokenize, pretokenize, pack_viterbi_chunks, \
    truncated_and_pad_packed_chunks, prefix_sum, load_segmentation_dictionary, use_gold_segmentations

import os
import pickle
import torch
import glob


"""
MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2
"""



def preprocess_language_modeling_with_unigram_dataset(args,
                                                      data_file: str,
                                                      cache_dir: str,
                                                      input_tokenizer: Tokenizer,
                                                      output_vocab: Integerizer,
                                                      max_blocks: int,
                                                      max_block_length: int,
                                                      max_unit_length: int,
                                                      max_length : int,
                                                      encoding: str = "utf-8"):
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
    with open(data_file, encoding=encoding) as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = pretokenize(text_str)

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = [len(chunk) for chunk in kept_chunks]

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, input_tokens)
            input_tokenizations, is_gold, is_ambiguous, is_matching, replaced = use_gold_segmentations(input_tokenizer, input_tokens, input_tokenizations, segmentation_dictionary)
            viterbi_chunks, viterbi_tokens = pack_viterbi_chunks(kept_chunks, input_tokenizations)
            length = sum([input_tokenizer.len_type(subword_type) for chunk in viterbi_chunks for subword_type in chunk])

            # bookkeeping
            assert viterbi_tokens == ntokens
            total_tokens += viterbi_tokens
            replaced_tokens += sum(replaced[:viterbi_tokens])
            is_gold_tokens += sum(is_gold[:viterbi_tokens])
            is_ambiguous_tokens += sum(is_ambiguous[:viterbi_tokens])
            is_matching_tokens += sum(is_matching[:viterbi_tokens])

            # integerize and pad if necessary
            input_ids = [input_tokenizer.vocab.index(subword_type) for chunk in viterbi_chunks for subword_type in chunk]
            if len(input_ids) <= max_length:
                input_ids += [input_tokenizer.pad_index] * (max_length - len(input_ids))
            else:
                raise ValueError(f"{text_str}\n{viterbi_chunks}\n{max_length}\n{input_ids}")

            # pos ids, labels, and mask
            pos_ids = list(range(max_length))
            labels = [id if id != input_tokenizer.pad_index else -100 for id in input_ids[1:]] + [-100]
            mask = [int(id != input_tokenizer.pad_index) for id in input_ids]

            # log to file
            item_name = os.path.join(cache_dir, f"{i}.pkl")

            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": input_ids,
                     "pos_ids": pos_ids,
                     "input_mask": mask,
                     "labels": labels,
                     "text": text_str,
                     "length": [length],  # in terms of characters
                     "ntokens": [ntokens]
                     }, file=f)
    msg = (f"Segmentation dictionary is {args.segmentation_dictionary}, {total_tokens} tokens, "
          f"{replaced_tokens} ({replaced_tokens / total_tokens}) replaced, "
          f"{is_gold_tokens} ({is_gold_tokens / total_tokens}) gold, "
          f"{is_ambiguous_tokens} ({is_ambiguous_tokens / total_tokens}) ambiguous."
          f"{is_matching_tokens} ({is_matching_tokens / replaced_tokens}) matching out of replaced.")
    print(msg)
    with open(f"{cache_dir}.meta.json", "wt") as meta_file:
        print(json.dumps({
            "segmentation_dictionary": args.segmentation_dictionary,
            "total_tokens": total_tokens,
            "replaced_tokens": replaced_tokens,
            "is_gold_tokens": is_gold_tokens,
            "is_ambiguous_tokens": is_ambiguous_tokens,
            "is_matching_tokens": is_matching_tokens
        }, indent=4), file=meta_file)

def preprocess_language_modeling_with_unigram_node_dataset(data_file: str,
                   cache_dir: str,
                   input_tokenizer: Tokenizer,
                   output_vocab: Integerizer,
                   max_blocks: int,
                   max_block_length: int,
                   max_unit_length: int,
                   max_length: int,
                   encoding: str = "utf-8",
                   pos_length: bool = False):
    max_length = max_length // 2
    
    clear_cache(cache_dir)
    
    with open(data_file, encoding=encoding) as textfile:
        for i, line in enumerate(tqdm(textfile)):
            text_str = line.strip()
            input_tokens = pretokenize(text_str)

            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            kept_chunks = truncated_and_pad_packed_chunks(input_tokenizer, packed_chunks, max_blocks)
            ntokens = [len(chunk) for chunk in kept_chunks]

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, input_tokens)
            viterbi_chunks = pack_viterbi_chunks(kept_chunks, input_tokenizations)
            length = sum([input_tokenizer.len_type(subword_type) for chunk in viterbi_chunks for subword_type in chunk])

            # integerize and pos_ids
            input_ids = [input_tokenizer.vocab.index(subword_type) for chunk in viterbi_chunks for subword_type in chunk]
            pos_increments = [0] + [input_tokenizer.len_type(subword_type)  for chunk in viterbi_chunks for subword_type in chunk][:-1]
            pos_ids = prefix_sum(pos_increments) if not pos_length else pos_increments

            
            assert len(pos_ids) == len(input_ids)
            # pad if necessary
            if len(input_ids) <= max_length:
                input_ids += [input_tokenizer.pad_index] * (max_length - len(input_ids))
                pos_ids += [0] * (max_length - len(pos_ids))
            else:
                raise ValueError(f"{viterbi_chunks}\n{max_length}\n{input_ids}")

            # labels, and mask
            labels = [id if id != input_tokenizer.pad_index else -100 for id in input_ids[1:]] + [-100]
            mask = [int(id != input_tokenizer.pad_index) for id in input_ids]

            # log to file
            item_name = os.path.join(cache_dir, f"{i}.pkl")
            with open(item_name, "wb") as f:
                pickle.dump(
                    {"input_ids": input_ids + [input_tokenizer.node_index for _ in input_ids],
                     "pos_ids": pos_ids + [pos_id + 1 for pos_id in pos_ids],
                     "input_mask": mask + mask,
                     "labels": [-100] * len(labels) + labels,
                     "text": text_str,
                     "length": [length], # in terms of characters
                     "ntokens": [ntokens]
                     }, file=f)



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
