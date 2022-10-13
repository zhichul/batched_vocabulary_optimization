import numpy as np

for file in ["/export/a01/corpora/ptb/ptb.train.txt", "/export/a01/corpora/ptb/ptb.valid.txt", "/export/a01/corpora/ptb/ptb.test.txt"]:
    with open(file, "rt") as f:
        lines = f.readlines()
    print(file)
    lines = [l.strip().split(" ") for l in lines]
    ntokens = np.array([len(line) for line in lines])
    nchars = np.array([len(token) for line in lines for token in line])
    stoken = "\n".join([f"[{np.quantile(ntokens, q)} @ {round(q*100)}% ]" for q in np.arange(0, 1.1, 0.1)])
    print(f"ntokens = {ntokens.sum()}\n{stoken}")
    schars = "\n".join([f"[{np.quantile(nchars, q)} @ {round(q*100)}%]" for q in np.arange(0, 1.1, 0.1)])
    print(f"ncahrs = {nchars.sum()}\n{schars}")
