import numpy as np

for file in ["/export/a01/corpora/ptb/ptb.train.txt", "/export/a01/corpora/ptb/ptb.valid.txt", "/export/a01/corpora/ptb/ptb.test.txt"]:
    with open(file, "rt") as f:
        lines = f.readlines()
    print(file)
    lines = [l.strip().split(" ") for l in lines]
    ntokens = np.array([len(line) for line in lines])
    nchars = np.array([len(token) for line in lines for token in line])
    nchars_per_line = np.array([sum(len(token) for token in line) for line in lines])
    stoken = "\n".join([f"[{np.quantile(ntokens, q)} @ {round(q*100)}% ]" for q in np.arange(0, 1.1, 0.1)])
    print(f"ntokens = {ntokens.sum()} / {ntokens.size} = {ntokens.mean()}\n{stoken}")
    schars = "\n".join([f"[{np.quantile(nchars, q)} @ {round(q*100)}%]" for q in np.arange(0, 1.1, 0.1)])
    print(f"ncahrs = {nchars.sum()} / {nchars.size} = {nchars.mean()}\n{schars}")
    schars_per_line = "\n".join([f"[{np.quantile(nchars_per_line, q)} @ {round(q*100)}%]" for q in np.arange(0, 1.1, 0.1)])
    print(f"ncahrs_perline = {nchars_per_line.sum()} / {nchars_per_line.size} = {nchars_per_line.mean()}\n{schars_per_line}")
