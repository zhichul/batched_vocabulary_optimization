import code
import json
from collections import defaultdict

from tqdm import tqdm
from bopt.core.integerize import Integerizer
from bopt.core.tokenizer import Tokenizer
from bopt.data.datasets import LazyDataset, LazySkipGramDataset
from bopt.data.language_modeling.utils import clear_cache, viterbi_tokenize, pretokenize, pack_viterbi_chunks, \
    truncated_and_pad_packed_chunks, prefix_sum, load_segmentation_dictionary, use_gold_segmentations

import os
import pickle
import torch
import glob

from bopt.data.skip_gram.utils import sft

"""
MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2
"""



def preprocess_skip_gram_with_unigram_dataset(args,
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
        for i, word in enumerate(words):
            # pack input into chunks
            packed_chunks = input_tokenizer.pack_chunks([word], max_block_length)

            # viterbi tokenize
            input_tokenizations = viterbi_tokenize(input_tokenizer, [word])
            input_tokenizations, is_gold, is_ambiguous, is_matching, replaced = use_gold_segmentations(input_tokenizer,
                                                                                                       [word],
                                                                                                       input_tokenizations,
                                                                                                       segmentation_dictionary)
            viterbi_chunks, viterbi_tokens = pack_viterbi_chunks(packed_chunks, input_tokenizations)
            length = sum([input_tokenizer.len_type(subword_type) for chunk in viterbi_chunks for subword_type in chunk])

            # bookkeeping
            assert viterbi_tokens == 1
            total_tokens += viterbi_tokens
            replaced_tokens += sum(replaced[:viterbi_tokens])
            is_gold_tokens += sum(is_gold[:viterbi_tokens])
            is_ambiguous_tokens += sum(is_ambiguous[:viterbi_tokens])
            is_matching_tokens += sum(is_matching[:viterbi_tokens])

            # integerize and pad if necessary
            input_ids = [input_tokenizer.vocab.index(subword_type) for chunk in viterbi_chunks for subword_type in
                         chunk]
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
                     }, file=f)
        with open(f"{cache_dir}.index.json", "wt") as index_file:
            json.dump({"counts": skip_gram_counts,
                       "words": words,
                       "total": sum(count for dist in skip_gram_counts for w in skip_gram_counts[dist] for count in
                                    skip_gram_counts[dist][w])},
                      index_file)
    msg = (f"Segmentation dictionary is {args.segmentation_dictionary}, {total_tokens} tokens, "
          f"{replaced_tokens} ({replaced_tokens / total_tokens}) replaced, "
          f"{is_gold_tokens} ({is_gold_tokens / total_tokens}) gold, "
          f"{is_ambiguous_tokens} ({is_ambiguous_tokens / total_tokens}) ambiguous."
          f"{is_matching_tokens} ({(is_matching_tokens / replaced_tokens) if replaced_tokens > 0 else 0.0}) matching out of replaced.")
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

class SkipGramUnigramDataset(LazySkipGramDataset):

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
        dist, src, tgt, i, max_block_length = index
        code.interact(local=locals())
        ret =  (torch.LongTensor(ex["src"]["input_ids"] + ex["tgt"]["input_ids"]),
                torch.LongTensor(ex["src"]["pos_ids"] + sft(ex["tgt"]["pos_ids"], max_block_length)),
                torch.LongTensor(ex["src"]["input_mask"] + ex["tgt"]["input_mask"]),
                torch.LongTensor([-100 for _ in ex["src"]["labels"]] + [ex["tgt"]["labels"]]),
                torch.LongTensor(ex["tgt"]["length"]),
                torch.LongTensor([1]),
                ex["text"])
        return ret
