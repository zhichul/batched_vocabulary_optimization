import torch

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import PADEDGE_ID, NONEDGE_ID
from bopt.utils import increasing_roll_right


def lattice_mask(*sizes):
    """
    Returns boolean matrices with upper triangular part (including the diagonal)
    as ones.
    This marks all the possible edges of a lattice in the matrix
    encoding of it as produced by for example integrize_for_forward.
    """
    return increasing_roll_right(torch.ones(sizes, dtype=torch.bool), 0)

def expand_encodings(encodings):
    """
    This expands each lattice (M by L matrix) in encodings to L lattices
    where the jth is a sublattice of the original starting at L-1-j.
    """
    size = encodings.size()
    M,L = size[-2:] # assert M <= L
    triu_ones = torch.triu(encodings.new_ones((M, L)), diagonal=0)
    edge_mask = encodings.new_zeros((L, M, L))
    padedge_mask = encodings.new_zeros((L, M, L))
    for j in range(L):
        i = L - 1 - j  # reverse the order, so we do backward of the smallest lattice first instead of full lattice first
        edge_mask[j, :L - i, i:L] = triu_ones[:L - i, :L - i] # the indices here may exceed M but that still works
        padedge_mask[j,0,:i] = 1
    edge_mask = edge_mask[:, :M, :]
    padedge_mask = padedge_mask[:, :M, :]
    nonedge_mask = (1 - (padedge_mask + edge_mask))
    encodings = encodings[...,None,:,:] * edge_mask + nonedge_mask * NONEDGE_ID + padedge_mask * PADEDGE_ID
    return encodings

def print_lattice(encoding, vocabulary, log_potentials=None, sentences=None):
    """
    Encoding should be a BxNxMxL tensor
    """
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
                    if id == PADEDGE_ID:
                        token = "-"
                    elif id == NONEDGE_ID:
                        token = "."
                    else:
                        token = vocabulary[id]
                    print(f"{token:>8s}" + ("" if log_potentials is None else f"/{log_potentials[b][n][m][l]:<8.2f}"), end=" ")
                print()
            print()
    print("==== ==== ==== ====")