import math
import sys

from bopt.utils import load_vocab, load_scalar_weights

file = sys.argv[1]
vocab = load_vocab(file)
weights = load_scalar_weights(file)
# with open(file, "wt") as f:
for token, weight in zip(vocab, weights.tolist()):
    wstr = "\t".join([str(math.log(w)) for w in weight])
    assert len(vocab) == weights.size(0)
    print("\t".join([token, wstr]))