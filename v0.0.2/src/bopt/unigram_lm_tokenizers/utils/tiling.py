
def tile(conditional_marginals):
    """
    This tiles a BxNxLxMxL blockwise representation of conditional marginals
    into an attention mask of size BxExE.
    where E = M x L - (M - 1) x M // 2 is the number of possible edges in
    each block.
    """
