from typing import List, Dict, Tuple
from bopt.core.integerize import Integerizer

import torch.nn as nn
import torch

INF = 1e9


class CachedTensorMixin:

    def __init__(self):
        self.parallel_backward_mask_cache = dict()
        self.permutation_cache = dict()
        self.edge_initial_position_cache = dict()
        self.valid_attention_cache = dict()

    def parallel_backward_mask(self, L: int, M: int, device: str = "cpu"):
        if (L, M, device) in self.parallel_backward_mask_cache:
            return self.parallel_backward_mask_cache[(L, M, device)]
        mmask = torch.zeros(L, L, L, dtype=torch.float).to(device)
        emask = torch.zeros(L, L, L, dtype=torch.float).to(device)
        triu_ones = torch.triu(torch.ones(L, L, dtype=torch.float).to(device), diagonal=0)
        for j in range(L):
            # the i=2 backward pass should have a transition mask looking like this
            #   valid edges                   mmask            emask
            # [ e t  a   h   ]             [ 1 1 1 1 ]       [ 0 0 1 1 ]
            # [      at      ]             [ 0 0 0 1 ]       [ 0 0 0 1 ]
            # [      ate hat ]             [ 0 0 0 0 ]       [ 0 0 0 0 ]
            # [          hate]             [ 0 0 0 0 ]       [ 0 0 0 0 ]
            #
            # mmask has the diagonal and a single path leading up to the i=2th node (this represents the actual lattice)
            # emask only cuts out the edges themselves that's actually in the sub-lattice (this is for picking out weigths)
            i = L - 1 - j  # reverse the order so we do backward of the smallest lattice first instead of full lattice first
            mmask[j, :L - i, i:L] = triu_ones[:L - i, :L - i]
            emask[j, :L - i, i:L] = triu_ones[:L - i, :L - i]
            mmask[j, 0, :i] = 1
        mmask, emask = mmask[:,:M,:], emask[:,:M,:]
        self.parallel_backward_mask_cache[(L, M, device)] = (mmask, emask)
        return mmask, emask

    def valid_attention(self, L: int, M: int, device: str = "cpu"):
        if (L, M, device) in self.valid_attention_cache:
            return self.valid_attention_cache[(L, M, device)]
        E = L * (L + 1) // 2 - (L - M) * (L - M + 1) // 2
        mask = torch.zeros(E, E, dtype=torch.float).to(device)
        edges = []
        for i in range(L):
            for j in range(i + 1, min(L + 1, i + M + 1)):
                edges.append((i, j))
        for i, (si, ei) in enumerate(edges):
            for j, (sj, ej) in enumerate(edges):
                if not (si >= ej or sj >= ei):
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1
        mask = mask[:E, :E]
        self.valid_attention_cache[(L, M, device)] = mask
        return mask

    def permutation(self, L: int, M: int):
        if (L, M) in self.permutation_cache:
            return self.permutation_cache[(L, M)]
        num_substr_per_length = list(reversed(range(1, L + 1)))
        num_substr_lt_length = [sum(num_substr_per_length[:i]) for i in range(L)]
        l = []
        for i in range(L):
            for j in range(i + 1, min(L + 1, i + M + 1)):
                l.append(num_substr_lt_length[j - i - 1] + i)
        self.permutation_cache[(L, M)] = l
        return l

    def edge_initial_position(self, L: int, M: int) -> List[int]:
        if (L, M) in self.edge_initial_position_cache:
            return self.edge_initial_position_cache[(L, M)]
        l = []
        for i in range(L):
            for j in range(i + 1, min(L + 1, i + M + 1)):
                l.append(i)
        self.edge_initial_position_cache[(L, M)] = l
        return l

class Tokenizer(nn.Module, CachedTensorMixin):

    def __init__(self, vocab: List[str], weights: Dict[str, float],
                 continuing_subword_prefix: str = None, pad_token: str = "[PAD]",
                 log_space_parametrization: bool = False, max_unit_length: int = INF):
        """
        `weights` should always be in log space
        `log_space_parametrization` controls whether the parameters are in log space or real space
        """
        super(Tokenizer, self).__init__()
        super(nn.Module, self).__init__()
        self.weights_dict = weights
        self.vocab_list = vocab
        self.csp = continuing_subword_prefix
        self.pad_token = pad_token
        self.lsp = log_space_parametrization
        self.max_unit_length = min(max_unit_length, max(len(u) for u in vocab))

        # represent vocabulary
        self.vocab = Integerizer(vocab)

        # represent weight parameters
        weights_tensor = torch.FloatTensor([weights[unit] for unit in vocab]).unsqueeze(1)
        self.weights = nn.Embedding(num_embeddings=len(vocab),
                                    embedding_dim=1,
                                    padding_idx=self.vocab.index(pad_token),
                                    _weight=weights_tensor if self.lsp else weights_tensor.exp())

    def clamp_weights(self, epsilon=1e-9) -> None:
        """
        If parametrized as real space, clamp the probabilities to [0,∞) after every update,
        mostly a trick than a principled approach, (i.e. should really do constrained-opt).
        """
        if self.lsp:
            raise ValueError("clamp_weights should only be used with real space parametrization")
        self.weights.weight.data = torch.clamp(self.weights.weight.data, min=epsilon)

    def forward(self, chunks: List[str], override_M: int):
        device = self.weights.weight.device
        L = max(len(chunk) for chunk in chunks) # max length of chunk
        B = len(chunks)
        M = min(self.max_unit_length, L, override_M)

        # encode lattice as special transition matrices
        fwd_ts = []
        fwd_ms = []
        bwd_ts = []
        bwd_ms = []
        lengths = []
        for chunk in chunks:
            fwd_t, fwd_m, bwd_t, bwd_m = self.encode_transitions(chunk, L, M, device=device)
            fwd_ts.append(fwd_t)
            fwd_ms.append(fwd_m)
            bwd_ts.append(bwd_t)
            bwd_ms.append(bwd_m)
            lengths.append(len(chunk))

        # ts and ms are [B x max_length x max_length] tensors
        # lengths is a [B] tensor
        fwd_ts = torch.stack(fwd_ts) # torch.FloatTensor
        fwd_ms = torch.stack(fwd_ms) # torch.FloatTensor
        bwd_ts = torch.stack(bwd_ts) # torch.FloatTensor
        bwd_ms = torch.stack(bwd_ms) # torch.FloatTensor
        lengths = torch.tensor(lengths, dtype=torch.long, device=device) # torch.LongTensor

        # 1 forward pass
        log_alpha, edge_log_alpha = self.forward_algorithm(fwd_ts, fwd_ms, lengths, M)
        ent, edge_ent = self.entropy(fwd_ts, fwd_ms, lengths, M)

        # max_length backward passes, one from each position
        mmask, emask = self.parallel_backward_mask(L, M, device=device)
        mmask = mmask.unsqueeze(0)
        emask = emask.unsqueeze(0)
        bwd_ts = (bwd_ts.unsqueeze(1) * emask).reshape(B * L, M, L)
        bwd_ms = (bwd_ms.unsqueeze(1) * mmask).reshape(B * L, M, L)
        bwd_lengths = torch.repeat_interleave(lengths, L)
        log_betas, edge_log_betas = self.forward_algorithm(bwd_ts, bwd_ms, bwd_lengths, M)
        log_betas = log_betas.reshape(B, L)
        edge_log_betas = edge_log_betas.reshape(B, L, M, L)

        # mask out the character suffixes in backward that are just auxiliary and don't really exist
        edge_log_betas = edge_log_betas * emask + torch.ones_like(edge_log_betas).fill_(-INF) * (1-emask)
        return fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, ent, edge_ent

    def encode_transitions(self, chunk: str, L: int, M: int, device="cpu") -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        forward encoding
        [ h a  t   e   ]
        [   ha at  te  ]
        [      hat ate ]
        [          hate]

        backward encoding
        [ h   a   t  e]
        [ ha  at  te  ]
        [ hat ate     ]
        [ hate        ]
        then flip ^ left to right
        """
        if len(chunk) > L:
            raise ValueError(f"chunk length of {chunk} is greater than allowed max chunk length {L}")
        fwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        bwd_mask = torch.zeros((M, L), dtype=torch.float, device=device)  # whenever a element of a matrix is from weights, set to 1
        fwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        bwd_ids = torch.zeros((M, L), dtype=torch.int, device=device)
        for s in range(len(chunk)):
            for l in range(min(len(chunk) - s, M) + 1):
                unit = chunk[s:s + l]
                unit = unit if self.csp is None or s == 0 else self.csp + unit
                if unit in self.vocab:
                    # fwd
                    fwd_mask[l - 1, s + l - 1] = 1
                    fwd_ids[l - 1, s + l - 1] = self.vocab.index(unit)
                    # bwd
                    bwd_mask[l - 1, len(chunk) - s - 1] = 1
                    bwd_ids[l - 1, len(chunk) - s - 1] = self.vocab.index(unit)

        fwd_t = self.weights(fwd_ids).squeeze(-1)
        fwd_m = fwd_mask
        bwd_t = self.weights(bwd_ids).squeeze(-1)
        bwd_m = bwd_mask

        # if parametrized as real space, do the log conversion since forward_algorithm is in log space
        if not self.lsp:
            fwd_t = torch.log(fwd_t)
            bwd_t = torch.log(bwd_t)
        return fwd_t, fwd_m, bwd_t, bwd_m

    def forward_algorithm(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor, M: int):
        """
        transition_matrix: [B, M, L]
        mask: [B, M, L]
        lengths: [B]
        """
        B, L = transition_matrix.size(0), transition_matrix.size(2)

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

    def entropy(self, transition_matrix: torch.FloatTensor, mask: torch.FloatTensor, lengths: torch.LongTensor, M: int):
        """Essentially forward but with a more tricky semiring"""
        B, L = transition_matrix.size(0), transition_matrix.size(2)

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

    def conditionals(self, fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, M: int, device="cpu"):
        """
        fwd_ts: B, M, L
        fwd_ms: B, M, L
        log_alpha: B
        edge_log_alpha: B, M, L
        log_betas: B, L
        edge_log_betas: B, L, M, L
        """
        B, L = edge_log_alpha.size(0), edge_log_alpha.size(2)
        E = L * (L+1) // 2 - (L-M) * (L-M+1) // 2# number of all possible edges for a chunk of length L with max edge length M is L (L + 1) / 2 - (L-M) * (L-M+1) // 2

        ela = edge_log_alpha
        elb = edge_log_betas.flip(-1) # remember the backward transition matrices are ordered from last to first char
                                     # flip so that it matches the first to last order of ela
        ts = fwd_ts
        ms = fwd_ms

        # gather all the `possible` edges (not necessarily in vocab, just possible) using masked indexing
        # they exactly live in upper triangular part of the transition matrix
        triu_ones = torch.triu(torch.ones(M, L, dtype=torch.bool).to(device), diagonal=0)

        ea = ela[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, 1, -1) # [B, 1, E] where E  = L (L+1) / 2
        eb = elb[triu_ones.flip(-1)[None, None, ...].expand(B, L, -1, -1)].reshape(B, L, -1) # [B, L, E]
        td = ts[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, 1, -1) # [B, L, E]
        ms = ms[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, -1) # [B, E]

        # permute the final dimension of size E cleverly to guarantee any edge ei in position i < j
        # will always intersect ej (in which case it's not valid attention) or be in front of ej which is valid
        permutation = self.permutation(L, M)
        ea = ea[..., permutation]
        eb = eb[..., permutation]
        td = td[..., permutation]
        ms = ms[..., permutation]

        em_ = ea + eb - td # [B, L, E] this is the (log of) the sum of all paths through every edge, with L different
                            # lattice ending positions

        # make conditionals matrix, encoded as A_ij = attention from ith row (src) -> to jth column (tgt)
        # given src tgt (src always guaranteed to be in front of target)
        # compute the numerator of the conditional probability
        # u(tgt | src) = bacward_marginal(src, tgt.start_node) * beta(tgt, last_node)
        # visually it's this [n1] ... src ... [nx] tgt ... [nN] the first part is the src backward marginal, second part is tgt beta
        edge_start_pos = self.edge_initial_position(L, M) # List of size E
        edge_start_pos = torch.tensor(edge_start_pos, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(B, E, E, 1) # [B, E (dup), E (tgt), 1]
        em_expansion = em_.transpose(-1,-2).unsqueeze(2).expand(-1,-1,E,-1) # [B, E, L] -> [B, E, 1, L] -> [B, E, E (dup), L]
        em_src = torch.gather(em_expansion, -1, (edge_start_pos-1).clamp(min=0)).squeeze(-1) # [B, E (src), E (tgt)]

        eb_tgt = eb[:,-1,:].unsqueeze(1) # [B, E (dup), E (tgt)], we take the last beta value wihch corresponds to the full lattice

        ec_numerator = em_src + eb_tgt

        # the numerator matrix should be symmetric and the triu values are the correct ones so lets flip it
        triu_c = torch.triu(torch.ones(E, E, dtype=torch.float).to(device), diagonal=1)
        tril_c = torch.tril(torch.ones(E, E, dtype=torch.float).to(device), diagonal=-1)
        ec_numerator = ec_numerator * triu_c + ec_numerator.transpose(-1,-2) * tril_c

        # denominator is just the marginals of the edge corresponding to the row
        ec_denom = em_[:,-1,:] # [B, E]
        ec = ec_numerator - ec_denom.unsqueeze(-1) # [B, E, E] - [B, E, 1 (dup)]

        # create a mask for valid edges (that don't cross), intersect it with the vocabulary mask so that what remains is
        # only edge-edge attentions that are valid and from vocab item to vocab item
        cm = self.valid_attention(L, M, device=device).unsqueeze(0) * ms.unsqueeze(1) * ms.unsqueeze(2)
        ec = ec * cm + (1-cm) * -INF # mask out invalid attentions

        # here's the normalized backward marginals maybe useful?
        em = em_ - log_betas[..., None]

        # put together the matrix by taking the triu, tril, and diagonal with only non-vocab items -inf'ed out
        c = triu_c * ec + tril_c * ec + torch.diag_embed((1-ms) * -INF)

        return c, ea, eb, em_, em