from vopt.synthetic.packaging.vocab_packaging import package_unigram_vocab, package_substring_vocab
import os



for file in ["/export/a01/corpora/simple/se.train.txt", "/export/a01/corpora/simple/se.valid.txt", "/export/a01/corpora/simple/se.test.txt"]:
    with open(file, "rt") as f:
        lines = f.readlines()
    print(file)
    lines = [l.strip().split(" ") for l in lines]
    chars = tuple(set(char for line in lines for token in line for char in token))

dir = "/export/a01/corpora/simple/"
files = [os.path.join(dir, "se.train.txt")]
cutoff = 8
package_substring_vocab(dir, files, max_unit_size=8, continuing_subword_prefix="@@", tokenizer=" ")
print(chars)
for size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000]:
    package_unigram_vocab(dir, files, chars, size=size, max_piece_length=cutoff, continuing_subword_prefix=None)
