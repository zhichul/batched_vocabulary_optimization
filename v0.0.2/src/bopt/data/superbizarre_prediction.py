from tqdm import tqdm

from bopt.superbizarre.derivator import Derivator
from bopt.utils import load_vocab
from experiments.utils.datasets import IDExampleDataset
from typing import List
import csv

def preprocess_superbizarre_prediction_dataset(file, args):
    examples = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["text", "score", "label"])
        for i, row in enumerate(tqdm(reader)):
            if i == 0: continue # skip the header line
            if args.input_tokenizer_model in ["unigram", "nulm"]:
                if (len(row["text"]) > args.max_block_length - 2): continue # skip lines that are longer than threshold, -2 is to make space for the [CLS] and for the added ▁
                text = "▁".join(["[CLS]", row["text"]])
            else:
                text = row["text"]
            labels = [row["label"]]
            examples.append([text, labels])
    return IDExampleDataset(examples)


def preprocess_superbizarre_prediction_gold_dataset(file, args):
    vocab = load_vocab(args.input_vocab)
    examples = []
    dr = Derivator()
    ids = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["text", "score", "label"])
        for i, row in enumerate(tqdm(reader)):
            if i == 0: continue # skip the header line
            pfxes, root, sfxes = dr.derive(row["text"], mode="morphemes")
            core_segmentation = []
            for j, pfx in enumerate(pfxes):
                if j == 0:
                    core_segmentation.append(args.space_character + pfx)
                else:
                    core_segmentation.append(pfx)
            core_segmentation.append(root)
            core_segmentation.extend(sfxes)
            if args.input_tokenizer_model in ["unigram", "nulm"]:
                if (len(row["text"]) > args.max_block_length - 2): continue # skip lines that are longer than threshold, -2 is to make space for the [CLS] and for the added ▁
                text = "▁".join(["[CLS]", row["text"]])
                segmentation = ["[CLS]"] + core_segmentation
            else:
                text = row["text"]
                segmentation = ["[CLS]"] + core_segmentation + ["[SEP]"]
            ids.append([vocab.index(s) if s in vocab else vocab.index(args.space_character + s, unk=True) for s in segmentation])
            labels = [row["label"]]
            examples.append([text, labels])
    gold_n = int(args.gold_percentage * len(examples))
    new_examples = []
    for i, ((text, labels), id) in enumerate(zip(examples, ids)):
        if i < gold_n:
            new_examples.append([[text, id], labels])
        else:
            new_examples.append([[text, None], labels])
    return IDExampleDataset(new_examples)

