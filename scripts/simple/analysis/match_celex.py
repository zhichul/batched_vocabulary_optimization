import code
import random
import sys
from collections import defaultdict

gold = defaultdict(set)
with open(sys.argv[1], "rt") as celex:
    for line in celex:
        if len(line.strip()) == 0:
            print("skipping empty line", file=sys.stderr)
            continue
        surf, seg = line.strip().split("\t")
        segs = seg.split(" ")
        seg_signature = tuple(len(seg) for seg in segs)
        gold[surf].add(seg_signature)
count_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

def segment(surf, signature):
    out = []
    prev = 0
    for incr in signature:
        out.append(surf[prev: prev + incr])
        prev  = prev + incr
    return out
segmentations = defaultdict(lambda : defaultdict(int))
segmentations_context = defaultdict(lambda : defaultdict(list))
with open(sys.argv[2], "rt") as forig:
    with open(sys.argv[3], "rt") as fseg:
        correct = 0
        total = 0
        contained = 0
        out_correct = 0
        for line_o, line_s in zip(forig, fseg):
            line_o = line_o.strip()
            line_s = line_s.strip()
            token_o = line_o.split(" ")
            token_s = line_s.split(" ")
            pointer = 0
            idx = 0
            for word in token_o:
                if idx >= len(token_s):
                    continue
                start = pointer
                end = pointer + len(word)
                signature = tuple()
                while pointer < end:
                    try:
                        signature += (len(token_s[idx]),)
                        pointer += len(token_s[idx])
                        idx += 1
                    except:
                        code.interact(local=locals())
                if not pointer == end:
                    print(pointer, end, word, signature, line_o, line_s)
                    assert pointer == end # true if line_s is a valid segmentation of line_o
                if word.lower() in gold:
                    # >1 morphemes
                    contained += 1
                    if signature in gold[word.lower()]:
                        correct += 1
                        count_dict["multi-correct"][word.lower()][signature] += 1
                    else:
                        count_dict["multi-incorrect"][word.lower()][signature] += 1
                else:
                    # monomorphemic
                    if len(signature) == 1:
                        out_correct += 1
                        count_dict["mono-correct"][word.lower()][signature] += 1
                    else:
                        count_dict["mono-incorrect"][word.lower()][signature] += 1
                segmentations[word.lower()]["-".join(segment(word.lower(), signature))] += 1
                segmentations_context[word.lower()]["-".join(segment(word.lower(), signature))].append(line_s)
                # print(f"{word}\t{signature}\t{gold[word.lower()]}")
                total += 1
        print(f"{sys.argv[3]:>80s} {correct} / {contained} = {correct / contained} | {out_correct} / {total - contained} = {out_correct/(total - contained)} | {out_correct + correct} / {total} = {(out_correct + correct)/ total}")
        try:
            _ = sys.argv[4]
        except:
            exit()
        denoms = {
            "mono-correct": out_correct,
            "mono-incorrect": total - contained - out_correct,
            "multi-correct": correct,
            "multi-incorrect": contained - correct
        }
        usage = defaultdict(set)
        usage_count = defaultdict(int)
        usage_count_breakdown = defaultdict(lambda:defaultdict(int))
        for correctness in ["correct", "incorrect"]:
            for split in ["mono", "multi"]:
                for word, segs in count_dict[f"{split}-{correctness}"].items():
                    for (seg, count) in segs.items():
                        pieces = segment(word, seg)
                        segmentation = "-".join(pieces)
                        for piece in pieces:
                            usage_count[piece] += count
                            usage[piece].add(segmentation)
                            usage_count_breakdown[piece][segmentation] += 1
        for correctness in ["correct", "incorrect"]:
            for split in ["mono", "multi"]:
                print(f"{split} {correctness} samples")
                sample_space = [f"({count:>6d}/{denoms[f'{split}-{correctness}']:>6d}) [{word} -> {'-'.join(segment(word, seg))}] should be one of {['-'.join(segment(word, gseg)) for gseg in gold[word]]} and the pieces have type/token usage as [{'-'.join([f'{len(usage[piece])}/{usage_count[piece]}' for piece in segment(word, seg)])}]" for word, segs in count_dict[f"{split}-{correctness}"].items() for (seg, count) in segs.items() ]
                weight = [count for word, segs in count_dict[f"{split}-{correctness}"].items() for _, count in segs.items() ]
                N = min(50, len(sample_space))
                try:
                    # for sample in sorted(list(set(random.sample(population=sample_space, k=N, counts=weight))), reverse=True):
                    #     print(sample)
                    for sample in sorted(sample_space, reverse=True)[:N]:
                        print(sample)
                except:
                    code.interact(local=locals())
                print()
        try:
            _ = sys.argv[5]
            code.interact(local=locals())
        except:
            exit()
