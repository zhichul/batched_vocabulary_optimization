from bopt.unigram_lm_tokenizers.encoding.forward_encoding import PADEDGE_ID, NONEDGE_ID

import torch

def get_token(id, vocabulary):
    if id == PADEDGE_ID:
        token = "-"
    elif id == NONEDGE_ID:
        token = "."
    else:
        token = vocabulary[id]
    return token

def print_lattice(encoding, vocabulary, log_potentials=None, sentences=None, exponentiate=False):
    """
    Encoding should be a BxNxMxL tensor
    """
    if exponentiate:
        log_potentials = log_potentials.exp()
    if isinstance(encoding, torch.Tensor):
        encoding = encoding.tolist()
    if isinstance(log_potentials, torch.Tensor):
        log_potentials = log_potentials.tolist()
    for b, sent in enumerate(encoding):
        print("==== ==== ==== ====")
        if sentences is not None:
            print(sentences[b])
        for n, block in enumerate(sent):
            for m, row in enumerate(block):
                for l, id in enumerate(row):
                    token = get_token(id, vocabulary)
                    print(f"{token:>8s}" + ("" if log_potentials is None else f"/{log_potentials[b][n][m][l]:<8.2f}"), end=" ")
                print()
            print()
    print("==== ==== ==== ====")