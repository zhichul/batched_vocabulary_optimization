import code
import csv
import os
import pickle
from collections import defaultdict

from tqdm import tqdm

from bopt.data.language_modeling.utils import load_segmentation_dictionary
from bopt.data.utils import load_vocab


def load_gold_segmentations(*files):
    sd = load_segmentation_dictionary(*files)
    return {k:[spans(l) for l in sd[k]] for k in sd}

def load_tokenizations(cache_dir, vocab_file, csp=None, dummy_prefix=None):
    vocab = load_vocab(vocab_file)
    pad_id = vocab.index("[PAD]")
    bos_id = vocab.index("[BOS]")
    eos_id = vocab.index("[EOS]")
    tokenizations = []
    for i in range(len(os.listdir(cache_dir))):
        with open(f"{cache_dir}/{i}.pkl", "rb") as f:
            example = pickle.load(f)
        subwords = [vocab[id] for id in example["input_ids"] if id != pad_id and id != bos_id and id != eos_id]
        if dummy_prefix is not None:
            subwords = [subwords[0][len(dummy_prefix):]] + subwords[1:]
        if csp is not None:
            subwords = [subwords[0]] + [subword[len(csp):] if subword.startswith(csp) else subword for subword in subwords[1:]]
        # if not "".join(subwords) == example["text"]:
        #     print(subwords, example)
        #     code.interact(local=locals())
        #     raise AssertionError
        yield example["text"], subwords

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

def align(text, tokens):
    words = text.split(" ")
    lengths = [len(word) for word in words]
    targets = prefix_sum(lengths)[1:]
    out_words = []
    out_subwords = []
    subwords = []
    i = 0
    ltotal = 0
    for word, target in zip(words, targets):
        while ltotal < target:
            subwords.append(tokens[i])
            ltotal += len(tokens[i])
            i += 1
        if not ltotal == target:
            code.interact(local=locals())
            assert False
        out_subwords.append(subwords)
        out_words.append(word)
        subwords = []
        if i >= len(tokens):
            assert i == len(tokens)
            break
    return list(zip(out_words, out_subwords))

def stats(tokenizations, gold):
    tp = 0
    pred = 0
    true = 0
    in_ = 0
    out = 0
    for text, tokens in tqdm(tokenizations):
        for word, subwords in align(text, tokens):
            if word in gold:
                gseg = gold[word]
                in_ += 1
            else:
                gseg = [{(0, len(word))}]
                out += 1
                continue
            sps = spans(subwords)
            for span in sps:
                if any(span in seg for seg in gseg):
                    tp += 1
            pred += len(sps)
            true += max(len(seg) for seg in gseg)
    total = in_ + out
    print(f"missed {out} / {total} = {out / total}")
    return tp, pred, true

if __name__ == "__main__":
    print("hello1")

    gseg = load_gold_segmentations("/home/blu/jhu/bopt/scripts/simple/analysis/celex_segmentation.tsv", "/home/blu/jhu/bopt/scripts/simple/analysis/celex_segmentation_mono.tsv")
    print("hello2")
    for size in [10000, 8000, 6000]:
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/ptb/exp110/44/768/{size}/cache/ptb.train.txt", f"/export/a01/corpora/ptb/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )
    for size in [0.1]:
        tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/ptb/exp123/44/4/1/384/10000/cache/ptb.train.txt", f"/export/a01/artifacts/bopt/ptb/exp20-mult/44/768/0.02/{size}/checkpoint-4000/learned_vocab.txt", csp="@@", dummy_prefix=None)
        tp, pred, true = stats(tokenizations, gseg)
        print(size, tp, pred, true, f"{tp / pred * 100:.2f} & {tp / true* 100:.2f} & {2 * (tp / pred * tp / true) / (tp / pred + tp / true)* 100:.2f}" )
    #
    # for size in [50, 100, 134]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp7/42/1/1/24/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/3/full/spmd-unigram-weights-{size}.txt", csp=None, dummy_prefix="▁")
    #     tp, pred, true = morpheme_prediction_segmentation_stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn3/exp6/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn3/exp4-11/42/{size}/768/checkpoint-600/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = morpheme_prediction_segmentation_stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    #
    # gseg = load_gold_segmentations_morpheme_prediction("/export/a01/corpora/vopt/syn/4/full/all.csv")
    # print(len(gseg))
    # for size in [50, 100, 200, 400]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp2/42/768/{size}/cache/train.csv", f"/export/a01/corpora/vopt/syn/4/full/spm-unigram-weights-{size}.txt", csp=None, dummy_prefix=None)
    #     tp, pred, true = morpheme_prediction_segmentation_stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
    # for size in [0.01, 0.1, 1.0]:
    #     tokenizations = load_tokenizations(f"/export/a01/artifacts/bopt/syn4/exp6/42/1/1/24/{size}/cache/train.csv", f"/export/a01/artifacts/bopt/syn4/exp4-11/42/{size}/768/checkpoint-3262/learned_vocab.txt", csp="@@", dummy_prefix=None)
    #     tp, pred, true = morpheme_prediction_segmentation_stats(tokenizations, gseg)
    #     print(size, tp, pred, true, tp / pred, tp / true, 2 * (tp / pred * tp / true) / (tp / pred + tp / true))
