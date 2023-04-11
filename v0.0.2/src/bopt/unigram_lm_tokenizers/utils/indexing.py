from bopt.unigram_lm_tokenizers.utils.encoding import lattice_mask

SPBINDEX_CACHE = dict()
EDGE2PREV_CACHE = dict()
EDGE2NEXT_CACHE = dict()

def linearize(encodings):
    """
    Return a flattened version of a lattice encoding.
    """
    return encodings[lattice_mask(*encodings.size())].reshape(*(encodings.size()[:-2]+(-1,)))

def serialize_by_start_position(encodings):
    M, L = encodings.size()[-2:]
    return linearize(encodings)[...,start_position_based_indexing(M,L)]

def start_position_based_indexing(M: int, L: int):
    """
    The returned indices can be used to extract ids / log potentials / conditional
    marginals from serialized edge matrices.
    """
    if (M, L) in SPBINDEX_CACHE:
        return SPBINDEX_CACHE[(M, L)]
    num_substr_per_length = list(reversed(range(1, L + 1)))
    num_substr_lt_length = [sum(num_substr_per_length[:i]) for i in range(L)]
    l = []
    for i in range(L):
        for j in range(i + 1, min(L + 1, i + M + 1)):
            l.append(num_substr_lt_length[j - i - 1] + i)
    SPBINDEX_CACHE[(M, L)] = l
    return l

def edge_to_prev_node(M: int, L: int):
    """
    The returned indices can be used to extract conditional marginals from serialized
    edge matrices (serialized version of backward_conditional_marginals).
    This assumes the serialization is reordered using start_position_based_indexing.
    """
    if (M, L) in EDGE2PREV_CACHE:
        return EDGE2PREV_CACHE[(M, L)]
    l = []
    for i in range(L):
        for j in range(i + 1, min(L + 1, i + M + 1)):
            l.append(i)
    EDGE2PREV_CACHE[(M, L)] = l
    return l

def edge_to_next_node(M: int, L: int):
    """
    The returned indices can be used to extract conditional marginals from serialized
    edge matrices (serialized version of forward_conditional_marginals).
    This assumes the serialization is reordered using start_position_based_indexing.
    """
    if (M, L) in EDGE2NEXT_CACHE:
        return EDGE2NEXT_CACHE[(M, L)]
    l = []
    for i in range(L):
        for j in range(i + 1, min(L + 1, i + M + 1)):
            l.append(j)
    EDGE2NEXT_CACHE[(M, L)] = l
    return l