import code
import csv
import os
import pickle
from collections import defaultdict

from tqdm import tqdm

from bopt.data.utils import load_vocab


def load_gold_segmentations(*files):
    d = dict()
    tmp = dict()
    special = {"[SP1]", "[SP2]", "[SP3]"}
    for file in files:
        with open(file, encoding='utf_8') as csvfile:
            reader = csv.DictReader(csvfile,fieldnames=["id", "label", "text", "features", "segmentation"])
            for i, row in enumerate(tqdm(reader)):
                word =  row["text"]
                seg = [tok for tok in row["segmentation"].split("-") if tok not in special]
                if word in d:
                    # print(word, seg, d[word], row, tmp[word])
                    if d[word] != seg:
                        code.interact(local=locals())
                        raise AssertionError
                tmp[word] = row
                d[word] = seg
    return {k:set(spans(d[k])) for k in d}

def load_tokenizations(cache_dir, vocab_file, csp=None, dummy_prefix=None):
    vocab = load_vocab(vocab_file)
    pad_id = vocab.index("[PAD]")
    tokenizations = []
    for i in range(len(os.listdir(cache_dir))):
        with open(f"{cache_dir}/{i}.pkl", "rb") as f:
            example = pickle.load(f)
        subwords = [vocab[id] for id in example["input_ids"][3:] if id != pad_id]
        if dummy_prefix is not None:
            subwords = [subwords[0][len(dummy_prefix):]] + subwords[1:]
        if csp is not None:
            subwords = [subwords[0]] + [subword[len(csp):] for subword in subwords[1:]]
        if not "".join(subwords) == example["text"]:
            print(subwords, example)
            code.interact(local=locals())
            raise AssertionError
        tokenizations.append((example["text"], spans(subwords)))
    return tokenizations

def signature(l):
    return [len(s) for s in l]

def prefix_sum(l):
    cumsum = 0
    psum = [0]
    for s in l:
        cumsum += s
        psum.append(cumsum)
    return psum

def spans(l):
    psum = prefix_sum(signature(l))
    return list(zip(psum[:-1], psum[1:]))

def stats(tokenizations, gold):
    tp = 0
    pred = 0
    true = 0
    for text, spans in tokenizations:
        for span in spans:
            if span in gold[text]:
                tp += 1
        pred += len(spans)
        true += len(gold[text])
    return tp, pred, true

if __name__ == "__main__":
    gseg = load_gold_segmentations("/export/a01/corpora/vopt/syn/3/full/train.csv")
    # for size in [50, 100, 200, 400]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp5/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    for size in [50, 100, 134]:
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp7/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spmd-unigram-weights-{size}.txt", csp=None, dummy_prefix="▁")
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp6/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp4-11/42/{size}/768/checkpoint-600/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    # gseg = load_gold_segmentations("/export/a01/corpora/vopt/syn/4/full/all.csv")
    # print(len(gseg))
    # for size in [50, 100, 200, 400]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp2/42/768/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/4/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp6/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn4/exp4-11/42/{size}/768/checkpoint-3262/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))


    gseg = load_gold_segmentations("/export/a01/corpora/vopt/syn/3/full/train.csv")
    # for size in [50, 100, 200, 400]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp5/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    # for size in [50, 100, 134]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp7/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spmd-unigram-weights-{size}.txt", csp=None, dummy_prefix="▁")
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #

    # for size in ["gold_test"]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp22/42/1/1/24/{size}/cache/train.csv",
    #                                        f"/export/a01/corpora/vopt/syn/3/full/substring-vocab-threshold=None.txt",
    #                                        csp="@@", dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true * 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true) * 100:.2f}")

    for size in reversed([0.01, 0.1, 1.0]):
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp36/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp34-11/42/{size}/768/checkpoint-600/learned_vocab.txt", csp="@@", dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )

    for size in [50, 100, 200, 400]:
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp35/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )


    gseg = load_gold_segmentations("/export/a01/corpora/vopt/syn/4/full/train.csv")

    for size in reversed([0.01, 0.1, 1.0]):
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp36/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn4/exp34-11/42/{size}/768/checkpoint-3262/learned_vocab.txt", csp="@@", dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )

    for size in [50, 100, 200, 400]:
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp32/42/768/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/4/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )


    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp22/42/1/1/24/{size}/21/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp21/42/1/1/24/{size}/late_ent/checkpoint-10000/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))

    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp22/42/1/1/24/{size}/23/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp23/42/1/1/24/{size}/checkpoint-10000/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))

    # for size in [0.1]:
    #     for CK in [2000, 4000, 6000]:
    #         tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp22/42/1/1/24/{size}/24/{CK}/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp24/42/2/4/96/{size}/late_ent/checkpoint-{CK}/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #         tp, pred, true = stats(tokenizations, gseg)
    #         print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
