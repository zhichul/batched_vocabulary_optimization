import math

import torch

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import PADEDGE_ID, NONEDGE_ID, NONEDGE_LOGPOT, PADEDGE_LOGPOT
from bopt.utils import increasing_roll_right


def lattice_mask(*sizes):
    """
    Returns boolean matrices with upper triangular part (including the diagonal)
    as ones.
    This marks all the possible edges of a lattice in the matrix
    encoding of it as produced by for example integrize_for_forward.
    """
    return increasing_roll_right(torch.ones(sizes, dtype=torch.bool), 0)

def convert_to_backward_encoding(forward_encoding):
    return increasing_roll_right(forward_encoding.flip(-1), NONEDGE_ID)
def convert_to_forward_encoding(backward_encoding):
    return increasing_roll_right(backward_encoding.flip(-1), NONEDGE_ID)
def convert_to_backward_log_potentials(forward_log_potentials):
    return increasing_roll_right(forward_log_potentials.flip(-1), NONEDGE_LOGPOT)
def convert_to_forward_log_potentials(backward_log_potentials):
    return increasing_roll_right(backward_log_potentials.flip(-1), NONEDGE_LOGPOT)

def expansion_mask(M, L, dtype=None, device=None):
    triu_ones = torch.triu(torch.ones((M, L), dtype=dtype, device=device), diagonal=0)
    edge_mask = torch.zeros((L, M, L), dtype=dtype, device=device)
    padedge_mask =  torch.zeros((L, M, L), dtype=dtype, device=device)
    for j in range(L):
        i = L - 1 - j  # reverse the order, so we do backward of the smallest lattice first instead of full lattice first
        edge_mask[j, :L - i, i:L] = triu_ones[:L - i, :L - i] # the indices here may exceed M but that still works
        padedge_mask[j,0,:i] = 1
    edge_mask = edge_mask[:, :M, :]
    padedge_mask = padedge_mask[:, :M, :]
    nonedge_mask = (1 - (padedge_mask + edge_mask)) if dtype != torch.bool else ~(padedge_mask | edge_mask)
    return edge_mask, padedge_mask, nonedge_mask
def expand_encodings(encodings):
    """
    This expands each lattice (M by L matrix) in encodings to L lattices
    where the jth is a sublattice of the original starting at L-1-j.
    """
    output_size = encodings.size()[:-2] + (encodings.size(-1), encodings.size(-2), encodings.size(-1))
    edge_mask, padedge_mask, nonedge_mask = expansion_mask(encodings.size(-2), encodings.size(-1),
                                                           dtype=torch.bool, device=encodings.device)
    encodings = encodings[...,None,:,:].expand(*output_size)
    encodings = encodings.clone()
    encodings[nonedge_mask.expand(*output_size)] = NONEDGE_ID
    encodings = encodings.clone()
    encodings[padedge_mask.expand(*output_size)] = PADEDGE_ID
    return encodings

def expand_log_potentials(log_potentials):
    """
    This expands each lattice (M by L matrix) in encodings to L lattices
    where the jth is a sublattice of the original starting at L-1-j.
    """
    output_size = log_potentials.size()[:-2] + (log_potentials.size(-1), log_potentials.size(-2), log_potentials.size(-1))
    edge_mask, padedge_mask, nonedge_mask = expansion_mask(log_potentials.size(-2), log_potentials.size(-1),
                                                           dtype=torch.bool, device=log_potentials.device)
    log_potentials = log_potentials[...,None,:,:].expand(*output_size)
    log_potentials = log_potentials.clone()
    log_potentials[nonedge_mask.expand(*output_size)] = NONEDGE_LOGPOT
    log_potentials = log_potentials.clone()
    log_potentials[padedge_mask.expand(*output_size)] = PADEDGE_LOGPOT
    return log_potentials

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