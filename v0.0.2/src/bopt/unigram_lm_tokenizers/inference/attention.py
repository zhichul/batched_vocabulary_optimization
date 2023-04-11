import code

import torch

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_LOGPOT
from bopt.unigram_lm_tokenizers.inference.forward_backward import conditional_marginals
from bopt.unigram_lm_tokenizers.utils.indexing import linearize, serialize_by_start_position, edge_to_prev_node, \
    edge_to_next_node


def attention_bias(attention_mask, edge_log_potentials):
    """
    Attention mask specifies the actual edges to mask in addition
    to the lattice conditional marginals computed from edge log potentials.
    """
    conditional_marginals_output = conditional_marginals(edge_log_potentials)
    attention_base = tile_attention(conditional_marginals_output)
    edge_mask = (attention_mask[:,None,:] * attention_mask[:,:,None]).to(torch.bool) # output B x NE x NE
    edge_bias = attention_base.new_ones(attention_base.size()).fill_(NONEDGE_LOGPOT)
    edge_bias[edge_mask] = 0.0
    attention_bias = attention_base + edge_bias
    return attention_bias
def tile_attention(conditional_marginals_output):
    """
    This tiles a BxNxL+1xMxL blockwise representation of conditional marginals
    into an attention mask of size BxExE.
    where E = M x L - (M - 1) x M // 2 is the number of possible edges in
    each block.

    This attention mask contains residual attention to padding-edges,
    as well as residual self-attention for non-edges and padding-edges.
    Those are masked out by the linearized attention mask in another function.
    """
    B, N, _, M, L = conditional_marginals_output.backward_conditional_marginals.size()
    bcm = serialize_by_start_position(conditional_marginals_output.backward_conditional_marginals)
    fcm = serialize_by_start_position(conditional_marginals_output.forward_conditional_marginals)
    E = bcm.size(-1)
    e2prev = edge_to_prev_node(M, L) # edge's backward attention is the same if they share the same prev node
    e2next = edge_to_next_node(M, L) # edge's forward attention is the same if they share the same next node
    backward_attention = bcm[..., e2prev, :]
    forward_attention = fcm[..., e2next, :]

    # assert (torch.tril(forward_attention.exp()) == 0).all()
    # assert (torch.triu(borward_attention.exp()) == 0).all()

    blockwise_conditional_marginals = torch.logaddexp(backward_attention, forward_attention)
    diagonal = torch.diag_embed(bcm.new_ones(E, dtype=torch.bool))
    self_base = bcm.new_ones((E,E)).fill_(NONEDGE_LOGPOT)
    self_base[diagonal] = 0.0
    blockwise_conditional_marginals = torch.logaddexp(blockwise_conditional_marginals,  # to add self-attention diagonal
                    self_base.expand_as(blockwise_conditional_marginals))
    blockwise_marginals = bcm[..., -1:, :] # last position in backward is unconditioned

    base = blockwise_marginals.reshape(B, 1, N * E)
    rows = []
    for i in range(N):
        row = base.expand(B, E, N * E).clone()
        row[...,i*E:(i+1)*E] = blockwise_conditional_marginals[:,i,...]
        rows.append(row)
    attention = torch.cat(rows, dim=-2)
    return attention