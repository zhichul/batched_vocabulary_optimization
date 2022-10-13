import code
from typing import List, Tuple

import torch
INF = 1e9

class LatticeAttentionMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation_cache = dict()
        self.edge_initial_position_cache = dict()
        self.valid_attention_cache = dict()
        self.causal_mask_cache = dict()

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

    def conditionals(self, fwd_ts: torch.FloatTensor,
                     fwd_ms:  torch.FloatTensor,
                     log_alpha:  torch.FloatTensor,
                     edge_log_alpha:  torch.FloatTensor,
                     log_betas:  torch.FloatTensor,
                     edge_log_betas:  torch.FloatTensor,
                     device: str = "cpu") -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        fwd_ts: B, M, L
        fwd_ms: B, M, L
        log_alpha: B
        edge_log_alpha: B, M, L
        log_betas: B, L
        edge_log_betas: B, L, M, L
        """
        B, M, L = edge_log_alpha.size()
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
        # em = em_ - log_betas[..., None]

        # put together the matrix by taking the triu, tril, and diagonal with only non-vocab items -inf'ed out
        c = triu_c * ec + tril_c * ec + torch.diag_embed((1-ms) * -INF)

        m = em_[:, -1, :] - log_alpha.unsqueeze(1)

        return c, ea, eb, em_, m

    def tile(self, marginals: torch.FloatTensor, conditionals: torch.FloatTensor, batch_size: int, num_blocks: int, M: int, L: int , ms: torch.FloatTensor, task_mask: torch.FloatTensor = None):
        """
            (log) marginals: batch * block, E
            (log) conditionals:  batch * block, E, E
            batch_size: batch
            num_blocks: block
        """
        B = batch_size * num_blocks # number of total blocks
        E = marginals.size(-1) # number of edges per block
        if num_blocks == 1:
            return conditionals.reshape(batch_size, E, E)
        # now there's at least two blocks to merge

        triu_ones = torch.triu(torch.ones(M, L, dtype=torch.bool).to(marginals.device), diagonal=0)
        ms = ms[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, -1)[..., self.permutation(L, M)] # [B, E]
        ms = ms.reshape(batch_size, num_blocks * E)
        mask = ms.unsqueeze(2) * ms.unsqueeze(1)
        if task_mask is not None:
            mask = mask * task_mask

        conditionals = conditionals.reshape(batch_size, num_blocks, E, E)
        marginals = marginals.reshape(batch_size, num_blocks, 1, E).expand(batch_size, num_blocks, (num_blocks -1) * E, E)
        attention = torch.cat([conditionals, marginals], dim=2)
        columns = [attention[:,i,:,:].roll(i * E, 1) for i in range(num_blocks)]
        attention = torch.cat(columns, dim=-1).reshape(-1, num_blocks * E, num_blocks * E)
        attention = attention * mask + (-INF) * (1-mask)
        return attention

    def tile_node(self, marginals: torch.FloatTensor, conditionals: torch.FloatTensor, batch_size: int, num_blocks: int, M: int, L: int):
        """
            (log) marginals: batch * block, E
            (log) conditionals:  batch * block, L, E
            batch_size: batch
            num_blocks: block
        """
        B = batch_size * num_blocks # number of total blocks
        E = marginals.size(-1) # number of edges per block
        if num_blocks == 1:
            return conditionals.reshape(batch_size, L, E)
        # now there's at least two blocks to merge
        conditionals = conditionals.reshape(batch_size, num_blocks, L, E)
        marginals = marginals.reshape(batch_size, num_blocks, 1, E).expand(batch_size, num_blocks, (num_blocks -1) * L, E)
        attention = torch.cat([conditionals, marginals], dim=2)
        columns = [attention[:,i,:,:].roll(i * L, 1) for i in range(num_blocks)]
        attention = torch.cat(columns, dim=-1).reshape(-1, num_blocks * L, num_blocks * E)
        return attention

    def tile_lm(self, edge_marginals: torch.FloatTensor, log_betas: torch.FloatTensor, marginals: torch.FloatTensor, attentions: torch.FloatTensor, batch_size: int, num_blocks: int, M: int, L: int, mask: torch.FloatTensor):
        em = edge_marginals - log_betas[..., None]
        bottom_left = self.tile_node(marginals, em, batch_size, num_blocks, M, L)#marginals.reshape(batch_size, 1, -1).expand(batch_size, num_blocks * L, attentions.size(-1)) # batch NL NE
        bottom_left = bottom_left.roll(1, -2)
        top_right = torch.ones_like(bottom_left).reshape(batch_size, bottom_left.size(-1), bottom_left.size(-2)) * -INF # botch NE NL
        bottom_right = (1 - torch.eye(num_blocks * L).unsqueeze(0).expand(batch_size, num_blocks * L, num_blocks * L).to(attentions.device)) * -INF # batch NL NL
        full_attention = torch.cat([torch.cat([attentions, top_right],dim=2), torch.cat([bottom_left, bottom_right], dim=2)],dim=1)
        full_attention = full_attention * mask + (-INF) * (1-mask)
        return full_attention

    def block_tril(self, mat: torch.Tensor, N: int, diagonal_mat: torch.Tensor = None, shift: int = 0):
        """
        Tiles the lower triangular part of a N x N grid with `mat`
        """
        if diagonal_mat is None: diagonal_mat = mat
        zeros = torch.zeros_like(mat)
        grid = [[mat if row - col > shift else (diagonal_mat if row - col == shift else zeros) for col in range(N)] for row in range(N)]
        return torch.cat([torch.cat(grid_row, dim=1) for grid_row in grid], dim=0)

    def causal_mask(self, N: int, L:int, M:int, device="cpu"):
        """
        Returns a NxE + NxL by NxE + NxL matrix representing the causal mask.
        """
        # make the within-block causal mask
        if (N, L, M, device) not in self.causal_mask_cache:
            triu = torch.triu(torch.ones((M, L), dtype=torch.bool))
            trius = torch.triu(torch.ones((L, M, L), dtype=torch.bool))
            for i in range(L):
                trius[i, :, i:] = 0

            permutation = self.permutation(L, M)
            node_causal_mask = trius[triu.unsqueeze(0).expand(L, M, L)].reshape(L, -1) # [L, E]
            node_causal_mask = node_causal_mask[..., permutation]

            edge_causal_mask = torch.stack([node_causal_mask[s] for s in range(L) for l in range(1, 1 + min(L-s, M))])

            top_left = self.block_tril(torch.ones_like(edge_causal_mask), N, diagonal_mat=edge_causal_mask)
            bottom_left = self.block_tril(torch.ones_like(node_causal_mask), N, diagonal_mat=node_causal_mask)
            top_right = torch.zeros_like(bottom_left).transpose(0,1)
            bottom_right = torch.eye(bottom_left.size(0))
            causal_mask = torch.cat([torch.cat([top_left, top_right], dim=1), torch.cat([bottom_left, bottom_right], dim=1)], dim=0)
            causal_mask[torch.eye(causal_mask.size(0), dtype=torch.bool)] = 1
            self.causal_mask_cache[(N, L, M, device)] = causal_mask.to(device)
        return self.causal_mask_cache[(N, L, M, device)]

