import code
import csv
import os
import sys
from argparse import ArgumentParser
from typing import Optional

import torch
from tqdm import tqdm
from dataclasses import dataclass
from bopt.arguments import add_logging_parameters, add_tokenizer_arguments
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID
from bopt.utils import load_vocab as load_vocab_stateless
from experiments.utils.memoizer import memoize_by_args


@dataclass
class Example:

    text:Optional[str]
    segmentation:Optional[str]

@memoize_by_args
def load_morpheme_prediction(file):
    examples = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["id", "label", "text", "features", "segmentation"])
        for i, row in enumerate(tqdm(reader)):
            text = row["text"]
            segmentation = row["segmentation"]
            examples.append(Example(text=text, segmentation=segmentation))
    return examples

load_vocab = memoize_by_args(load_vocab_stateless)

def save_weight_size_values(name, output_prefix, vocab, *values):
    with open(os.path.join(output_prefix, f"{name}.txt"), "wt") as f:
        for i, vs in tqdm(enumerate(zip(*[value.tolist() for value in values]))):
                line = [vocab[i]] + [f"{value:.2}" for value in vs]
                print("\t".join(line), file=f)

def save_lattice_values(name, output_prefix, examples, vocab, ids, forward_encodings, *singleton_values):
    attribution = torch.tensor(ids)[:,None,None,None].expand_as(forward_encodings)
    mask = (forward_encodings != NONEDGE_ID) & (forward_encodings != PADEDGE_ID)
    singleton_values = [singleton_v[mask].tolist() for singleton_v in singleton_values]
    attributions = attribution[mask].tolist()
    wis = forward_encodings[mask].tolist()
    with open(os.path.join(output_prefix, f"{name}.tsv"), "at") as f:
        with open(os.path.join(output_prefix, f"{name}.ex.tsv"), "at") as g:
            for tup in tqdm(zip(*([wis, attributions] + singleton_values))):
                wi = tup[0]
                attr = tup[1]
                svs = tup[2:]
                text = examples[attr].text
                segmentation = examples[attr].segmentation
                svs_str = [f"{sv:.2}" for sv in svs]
                line = [vocab[wi], text, segmentation] + svs_str
                # with open(os.path.join(output_prefix, f"{name}.{vocab[wi]}.tsv"), "at") as f:
                # with open(os.path.join(output_prefix, f"{name}.tsv"), "at") as f:
                print("\t".join(line), file=f)
                line = [text, segmentation, vocab[wi]] + svs_str
                # with open(os.path.join(output_prefix, f"{name}.ex.{text}.tsv"), "at") as f:
                # with open(os.path.join(output_prefix, f"{name}.ex.tsv"), "at") as f:
                print("\t".join(line), file=g)

def save_pairwise_values(name, output_prefix, examples, vocab, ids, input_ids, attention_mask, *pairwise_values):
    attribution = torch.tensor(ids)[:,None,None].expand_as(pairwise_values[0])
    src = input_ids[:,:,None].expand_as(pairwise_values[0])
    tgt = input_ids[:,None,:].expand_as(pairwise_values[0])
    mask = (attention_mask[:,:,None] * attention_mask[:,None,:]).to(torch.bool)
    src_wids = src[mask].tolist()
    tgt_wids = tgt[mask].tolist()
    attributions = attribution[mask].tolist()
    pvs = [pairwise_v[mask].tolist() for pairwise_v in pairwise_values]
    with open(os.path.join(output_prefix, f"{name}.tsv"), "at") as f:
        with open(os.path.join(output_prefix, f"{name}.ex.tsv"), "at") as g:
            for tup in tqdm(zip(*([attributions, src_wids, tgt_wids] + pvs))):
                ws = tup[1]
                wt = tup[2]
                attr = tup[0]
                pvs = tup[3:]
                text = examples[attr].text
                segmentation = examples[attr].segmentation
                pvs_str = [f"{pv:.2}" for pv in pvs]
                line = [vocab[ws], vocab[wt], text, segmentation] + pvs_str
                # with open(os.path.join(output_prefix, f"{name}.{vocab[ws]}.{vocab[wt]}.tsv"), "at") as f:
                # with open(os.path.join(output_prefix, f"{name}.tsv"), "at") as f:
                print("\t".join(line), file=f)
                line = [text, segmentation, vocab[ws], vocab[wt]] + pvs_str
                # with open(os.path.join(output_prefix, f"{name}.ex.{text}.tsv"), "at") as f:
                # with open(os.path.join(output_prefix, f"{name}.ex.tsv"), "at") as f:
                print("\t".join(line), file=g)


def post_process_file(file, args):
    contents = torch.load(file)
    input_ids = contents["input_ids"]
    type_ids = contents["type_ids"]
    position_ids = contents["position_ids"]
    attention_mask = contents["attention_mask"]
    forward_encodings = contents["forward_encodings"]
    attentions = contents["attentions"]
    attentions_std = contents["attentions_std"]
    attentions_grad = contents["attentions_grad"]
    attentions_grad_std = contents["attentions_grad_std"]
    conditional_marginals = contents["conditional_marginals"]
    conditional_marginals_grad = contents["conditional_marginals_grad"]
    weights = contents["weights"]
    weights_grad_task = contents["weights_grad_task"]
    weights_grad_entropy = contents["weights_grad_entropy"]
    weights_grad_l1 = contents["weights_grad_l1"]
    ids = contents["ids"]
    sentences = contents["sentences"]
    labels = contents["labels"]
    step = contents["step"]
    raw_step = contents["raw_step"]
    exp_args = contents["args"]

    examples = load_morpheme_prediction(exp_args.train_dataset)
    vocab = load_vocab(exp_args.input_vocab)
    # save
    output_prefix = os.path.join(args.output_directory, f"expanded-dynamics-{step}")
    os.makedirs(output_prefix, exist_ok=True)
    save_weight_size_values("weights", output_prefix, vocab, weights, weights_grad_l1)
    save_lattice_values("lattice", output_prefix, examples, vocab, ids, forward_encodings, -weights_grad_entropy, -weights_grad_task)
    save_pairwise_values("pairwise", output_prefix, examples, vocab, ids, input_ids, attention_mask, conditional_marginals, -conditional_marginals_grad, attentions, -attentions_grad)

def main():
    parser = ArgumentParser()

    parser.add_argument("--inputs", required=True, nargs="+")
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--overwrite_output_directory", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.output_directory) and not args.overwrite_output_directory:
        print("Warning: output directory exists, if you did not clear the results from a previous run the new run will be APPENDED to it rather than overwrite it.")
        # raise ValueError("Please set overwrite_output_directory to true when using existing directories.")
    if os.path.exists(args.output_directory) and args.overwrite_output_directory:
        ans = input(f"{args.output_directory} exists, about to wipe content, continue?")
        if ans.lower() in ["y", "yes"]:
            os.system(f"rm -r {args.output_directory}/expanded-dynamics-*")
        else:
            exit(0)
    args = parser.parse_args()

    for file in tqdm(args.inputs):
        post_process_file(file, args)

if __name__ == "__main__":
    main()