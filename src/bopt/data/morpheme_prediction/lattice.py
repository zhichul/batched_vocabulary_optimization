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
from bopt.data.utils import load_vocab, load_weights, constant_initializer

MAX_BLOCKS = 10 # N: Number of words roughly in a sentence
MAX_BLOCK_LENGTH = 20 # L: number of characters in a block
MAX_UNIT_LENGTH = 20 # M: number of characters in a candidate unit
# max number of edges in a lattice for a block
MAX_BLOCK_TOKENS = (MAX_BLOCK_LENGTH * (MAX_BLOCK_LENGTH + 1)) // 2 - ((MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH) * (MAX_BLOCK_LENGTH - MAX_UNIT_LENGTH + 1)) // 2

def preprocess_morpheme_prediction_with_lattices_dataset(data_file: str,
                   cache_dir: str,
                   input_tokenizer: TokenizationMixin,
                   output_vocab: Integerizer,
                   max_blocks: int = None,
                   max_block_length: int = None,
                   max_unit_length: int = None,
                   debug=False):
    for f in glob.glob(f'{cache_dir}/*'):
        os.remove(f)
    E = (max_block_length * (max_block_length + 1)) // 2 - ((max_block_length - max_unit_length) * (max_block_length - max_unit_length + 1)) // 2
    ws = Whitespace()
    with open(data_file, encoding='utf_8') as csvfile:
        reader = csv.DictReader(csvfile,fieldnames=["id", "label", "text", "features", "segmentation"])
        task_mask = torch.ones((max_blocks * E, max_blocks * E))
        for k in range(3):
            task_mask[:, k * max_unit_length] = 0
            task_mask[k * max_unit_length, k * max_unit_length] = 1
        for i, row in enumerate(tqdm(reader)):
            text_str = row["text"]
            input_tokens = [pair[0] for pair in ws.pre_tokenize_str(text_str)]
            output_labels = [output_vocab.index(tok, unk=True) for tok in row["features"].split("-")]
            input_tokens = ["[SP1]", "[SP2]", "[SP3]"] + input_tokens
            packed_chunks = input_tokenizer.pack_chunks(input_tokens, max_block_length)
            if len(packed_chunks) > max_blocks:
                print(f"[WARNING] Truncating {packed_chunks} to {' '.join(sum(packed_chunks[:max_blocks], []))}")
                packed_chunks = packed_chunks[:max_blocks]
            elif len(packed_chunks) < max_blocks:
                packed_chunks += [[]] * (max_blocks - len(packed_chunks))
            fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = input_tokenizer.encode_packed_batch(packed_chunks, max_unit_length, max_block_length)
            ids, mask, pos_ids, _, _, _ = input_tokenizer.integerize_packed_chunks(packed_chunks, max_unit_length, max_block_length)
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
                     "mmask": mmask.tolist(),
                     "emask": emask.tolist(),
                     "tmask": task_mask.tolist()
                     }, file=f
                )



class MorphemePredictionLatticeDataset(LazyDataset):

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
            "mmask"
            "emask"
            "tmask"
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
                torch.LongTensor(ex["mmask"]),
                torch.LongTensor(ex["emask"]),
                torch.LongTensor(ex["tmask"]))

if __name__ == "__main__":
    torch.manual_seed(42)
    temp_root = "/tmp/bopt_morpheme_prediction/"
    temp_csv = os.path.join(temp_root + "debug.csv") # change this if your system doesn't have a tmp dir
    temp_cache_dir = os.path.join(temp_root + "cache")
    os.makedirs(temp_root, exist_ok=True)
    os.makedirs(temp_cache_dir, exist_ok=True)

    max_blocks = 2
    max_block_length = 3
    max_unit_length = 2
    continuing_subword_prefix = "@@"
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[WBD]", "[SP1]", "[SP2]", "[SP3]", "[SP4]", "[SP5]"]
    input_vocab = Integerizer(specials + ["o", "ox", "@@x", "@@xx"])
    output_vocab = Integerizer(specials + ["p2", "r8", "s8", "s10"])
    raw_input_vocab = [w if not w.startswith(continuing_subword_prefix) else w[len(continuing_subword_prefix):] for w in input_vocab if w not in specials ]

    tokenizer = Tokenizer(vocab = input_vocab,
                          weights = constant_initializer(input_vocab),
                          log_space_parametrization = False,
                          continuing_subword_prefix = continuing_subword_prefix,
                          pad_token = "[PAD]",
                          max_unit_length = max_unit_length,
                          specials = specials,
    )

    with open(temp_csv, "wt") as f:
        print(f"{0},o:2/x:8/x:8,oxx,p2-r8-s8,o-x-x", file=f)
        print(f"{1},o:2/x:8,ox,p2-r8-s10,o-x-[SP3]", file=f)

    preprocess_morpheme_prediction_with_lattices_dataset(temp_csv,
                   temp_cache_dir,
                   tokenizer,
                   output_vocab,
                   max_blocks,
                   max_block_length,
                   max_unit_length,
                   debug=False)
    dataset = MorphemePredictionLatticeDataset(temp_cache_dir)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=2, num_workers=1)
    for batch in data_loader:
        input_ids, pos_ids, input_mask, label_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask, tmask = batch
        ent, a, m, c = tokenizer.forward(fwd_ids,
                          fwd_ms,
                          lengths,
                          bwd_ids,
                          bwd_ms,
                          bwd_lengths,
                          mmask,
                          emask,
                          tmask)
        E = (max_block_length * (max_block_length + 1)) // 2 - (
                    (max_block_length - max_unit_length) * (max_block_length - max_unit_length + 1)) // 2
        for b in range(input_ids.size(0)):
            for i in range(max_blocks):
                offset = 0
                for j in range(max_block_length):
                    for l in range(min(max_unit_length, max_block_length - j)):
                        print(f"{tokenizer.vocab[input_ids[b, i * E + offset + l]] if input_mask[b, i * E + offset + l] else 'BLANK':>10s} "
                              f"{pos_ids[b, i * E + offset + l] if input_mask[b, i * E + offset + l] else '':<2} "
                              f"{(': ' + output_vocab[label_ids[b, i * E + offset + l]]) if label_ids[b, i * E + offset + l] >= 0 else '':<6s}",
                              end=" | ")
                    offset += min(max_unit_length, max_block_length - j)
                    print()
                print()
        print("########")
        print("entropy exponeitated (should equal to number of paths in the lattices):", ent.exp().tolist())
        all_input_ids = input_ids.reshape(-1)
        all_mask = input_mask.reshape(-1)
        cexp = c.exp()
        aexp = a.exp()
        for b in range(input_ids.size(0)):
            for block in range(max_blocks):
                for src in range(c.size(2)):
                    for tgt in range(c.size(2)):
                        src_index = src
                        tgt_index = tgt
                        src_s = input_vocab[input_ids[b, block * E + src_index]]
                        tgt_s = input_vocab[input_ids[b, block * E + tgt_index]]
                        print(f"[{src_s:>5s} -> {tgt_s:<5s}] {round(cexp[b, block, src, tgt].item(),2) if cexp[b, block, src, tgt] > 0 else '':<4}", end=" ")
                    print()
                print()
        for b in range(input_ids.size(0)):
            print([input_vocab[id] for id in input_ids[b]])
            for block_s in range(max_blocks):
                for src in range(c.size(2)):
                    for block_t in range(max_blocks):
                        for tgt in range(c.size(2)):
                            src_index = block_s * E + src
                            tgt_index = block_t * E + tgt
                            src_s = input_vocab[input_ids[b, src_index]]
                            tgt_s = input_vocab[input_ids[b, tgt_index]]
                            print(
                                f"[{src_s:>5s} -> {tgt_s:<5s}] {round(aexp[b, src_index, tgt_index].item(), 2) if aexp[b, src_index, tgt_index] > 0 else '':<4}",
                                end=" ")
                    print()
            print()
        input("Hit enter to continue...")

