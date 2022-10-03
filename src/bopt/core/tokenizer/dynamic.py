from typing import List, Tuple

import torch

INF = 1e9

class LatticeDPMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_algorithm(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        transition_matrix: [B, M, L]
        mask: [B, M, L]
        lengths: [B]
        """
        B, M, L = transition_matrix.size()

        bmask: torch.BoolTensor = mask.to(torch.bool)
        edge_log_alphas: torch.FloatTensor = torch.ones_like(transition_matrix).fill_(-INF)
        edge_log_alphas[bmask] = 0.0
        edge_log_alphas += transition_matrix

        log_alphas: List[torch.FloatTensor] = [mask.new_zeros(B)]
        for i in range(L):
            # maski is the intersection of bmask and the ith diagonal, represents all outgoing edges at a node
            #
            # Example: for the chunk `hate`, the lattice (only drawing character edges) looks like
            #
            #   [n0] - h - [n1] - a - [n2] - t - [n3] - e [n4]
            #
            # where [nx] is the xth node in the lattice.
            #
            # The transition matrices organize the outgoing edge potentials from a given node along (shifted) diagonals,
            # and incoming edge potentials along columns.
            #
            # For example, for forward transitions, mask2 represents all outgoing edges from
            # the node before `t`.
            #   valid edges                   bmask        shifted diagonal
            # [ h a  t   e   ]             [ 1 1 1 1 ]       [ 0 0 1 0 ]
            # [      at      ]   mask2 =   [ 0 0 1 0 ]   &   [ 0 0 0 1 ]
            # [      hat ate ]             [ 0 0 1 1 ]       [ 0 0 0 0 ]
            # [          hate]             [ 0 0 0 1 ]       [ 0 0 0 0 ]
            maski = (bmask & torch.diag_embed(mask.new_ones(L - i, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)
            # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
            # for i = 2 as an example, this propagates [n1], which we aggregated last iteration
            node_to_edge = log_alphas[i][:, None, None] * maski
            edge_log_alphas = edge_log_alphas + node_to_edge

            # this update corresponds to aggregating the `incoming` edge-alphas into the alpha of a node
            # for i = 2 as an example, this aggregates [n2], which is guaranteed
            # to have all incoming edges with edge-alphas already computed
            log_alphas.append(torch.logsumexp(edge_log_alphas[:, :, i], -1))

        log_alphas = torch.gather(torch.stack(log_alphas), 0, lengths.unsqueeze(0))
        return log_alphas.squeeze(0), edge_log_alphas

    def entropy(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Essentially forward but with a more tricky semiring"""
        B, M, L= transition_matrix.size()

        bmask: torch.BoolTensor = mask.to(torch.bool)
        edge_log_alphas: torch.FloatTensor = torch.ones_like(transition_matrix).fill_(-INF)
        edge_log_alphas[bmask] = 0.0
        edge_log_alphas += transition_matrix
        edge_entropy: torch.FloatTensor = torch.zeros_like(transition_matrix) # diff
        entropy_transition_matrix: torch.FloatTensor = - transition_matrix * transition_matrix.exp() # diff -plogp

        log_alphas: List[torch.FloatTensor] = [mask.new_zeros(B)]
        entropies: List[torch.FloatTensor] = [mask.new_zeros(B)] # diff
        for i in range(L):
            maski = (bmask & torch.diag_embed(mask.new_ones(L - i, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)
            node_to_edge = log_alphas[i][:, None, None] * maski
            edge_log_alphas = edge_log_alphas + node_to_edge
            log_alphas.append(torch.logsumexp(edge_log_alphas[:, :, i], -1))

            node_to_edge_ent = ((log_alphas[i][:, None, None].exp() * maski) * (entropy_transition_matrix * maski)
                                +  (entropies[i][:, None, None] * maski) * (transition_matrix.exp() * maski))# diff: possibly underflows if low weight edges exist
            edge_entropy = edge_entropy + node_to_edge_ent
            entropies.append(torch.sum(edge_entropy[:, :, i], -1))
        uentropy = torch.gather(torch.stack(entropies), 0, lengths.unsqueeze(0))
        log_alphas = torch.gather(torch.stack(log_alphas), 0, lengths.unsqueeze(0))
        return (uentropy/ log_alphas.exp() + log_alphas).squeeze(0), edge_entropy # entropy = internal energy + free energy
