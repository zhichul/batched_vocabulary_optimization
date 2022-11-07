import code
from typing import List, Tuple

import torch

INF = 1e9

DEBUG = False
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

    def viterbi_algorithm(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, List[torch.LongTensor]]:
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
        back_pointers: List[torch.LongTensor] = []
        for i in range(L):
            maski = (bmask & torch.diag_embed(mask.new_ones(L - i, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)

            node_to_edge = log_alphas[i][:, None, None] * maski
            edge_log_alphas = edge_log_alphas + node_to_edge

            values, indices = torch.max(edge_log_alphas[:, :, i], -1)
            log_alphas.append(values)
            back_pointers.append(indices)

        log_alphas = torch.gather(torch.stack(log_alphas), 0, lengths.unsqueeze(0)).squeeze(0)
        if log_alphas.isnan().any():
            print("Nan loss detected")
            code.interact(local=locals())
        return log_alphas, edge_log_alphas, back_pointers

    def decode_backpointers(self, fwd_ids: torch.LongTensor, lengths: torch.LongTensor, back_pointers: List[torch.LongTensor]) -> List[List[int]]:
        B, M, L = fwd_ids.size()

        viterbi_sequences = []
        for b in range(B):
            viterbi_back_sequence = []
            curr_col = lengths[b]-1
            while curr_col >= 0:
                curr_row = back_pointers[curr_col][b]
                curr_id = fwd_ids[b, curr_row, curr_col].item()
                unit_length = curr_row + 1
                viterbi_back_sequence.append(curr_id)
                curr_col -= unit_length
            assert curr_col == -1
            viterbi_sequences.append(list(reversed(viterbi_back_sequence)))
        return viterbi_sequences
    def entropy(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Essentially forward but with a more tricky semiring"""
        B, M, L= transition_matrix.size()

        bmask: torch.BoolTensor = mask.to(torch.bool)
        edge_log_alphas: torch.FloatTensor = torch.ones_like(transition_matrix).fill_(-INF)
        edge_log_alphas[bmask] = 0.0
        edge_log_alphas += transition_matrix
        edge_entropy: torch.FloatTensor = torch.zeros_like(transition_matrix) # diff
        entropy_transition_matrix: torch.FloatTensor = - transition_matrix * transition_matrix.exp() * mask # diff -plogp
        code.interact(local=locals())
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
        if DEBUG: code.interact(local=locals())
        return (uentropy/ log_alphas.exp() + log_alphas).squeeze(0), edge_entropy # entropy = internal energy + free energy

    def expectation(self, transition_matrix: torch.FloatTensor, value_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Essentially forward but with a more tricky semiring"""
        B, M, L= transition_matrix.size()

        # adjusting for underflow
        bmask: torch.BoolTensor = mask.to(torch.bool)
        edge_log_alphas: torch.FloatTensor = torch.ones_like(transition_matrix).fill_(-INF)
        edge_log_alphas[bmask] = 0.0
        edge_log_alphas += transition_matrix
        edge_expected_value: torch.FloatTensor = torch.zeros_like(transition_matrix) # diff
        expected_value_transition_matrix: torch.FloatTensor = value_matrix * transition_matrix.exp() # diff pv

        log_alphas: List[torch.FloatTensor] = [mask.new_zeros(B)]
        expected_values: List[torch.FloatTensor] = [mask.new_zeros(B)] # diff
        for i in range(L):
            maski = (bmask & torch.diag_embed(mask.new_ones(L - i, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)
            node_to_edge = log_alphas[i][:, None, None] * maski
            edge_log_alphas = edge_log_alphas + node_to_edge
            log_alphas.append(torch.logsumexp(edge_log_alphas[:, :, i], -1))

            node_to_edge_ent = ((log_alphas[i][:, None, None].exp() * maski) * (expected_value_transition_matrix * maski)
                                +  (expected_values[i][:, None, None] * maski) * (transition_matrix.exp() * maski))# diff: possibly underflows if low weight edges exist
            edge_expected_value = edge_expected_value + node_to_edge_ent
            expected_values.append(torch.sum(edge_expected_value[:, :, i], -1))
        uexpected_value = torch.gather(torch.stack(expected_values), 0, lengths.unsqueeze(0))
        log_alphas = torch.gather(torch.stack(log_alphas), 0, lengths.unsqueeze(0))
        ev = (uexpected_value/ log_alphas.exp()).squeeze(0)
        if ev.sum() < 1 or ev.isnan().any():
            print("Warning: underflow may be happening")
        return ev, edge_expected_value # entropy = internal energy + free energy
