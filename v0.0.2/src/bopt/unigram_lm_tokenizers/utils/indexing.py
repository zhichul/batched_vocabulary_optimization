SPBINDEX_CACHE = dict()
EDGE2NODE_CACHE = dict()

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

def edge_to_node(M: int, L: int):
    """
    The returned indices can be used to extract conditional marginals from serialized
    edge matrices.
    """
    if (M, L) in EDGE2NODE_CACHE:
        return EDGE2NODE_CACHE[(M, L)]
    l = []
    for i in range(L):
        for j in range(i + 1, min(L + 1, i + M + 1)):
            l.append(i)
    EDGE2NODE_CACHE[(M, L)] = l
    return l
