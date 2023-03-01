import math
from typing import Dict, Tuple, List

import torch
import code
from torch import nn

from bopt.core.tokenizer.attention import LatticeAttentionMixin
from bopt.core.tokenizer.dynamic import LatticeDPMixin
from bopt.core.tokenizer.tokenization import TokenizationMixin
from bopt.core.utils import increasing_roll_left, increasing_roll_right

INF = 1e9

class Tokenizer(TokenizationMixin, LatticeDPMixin, LatticeAttentionMixin, nn.Module):

    def __init__(self, *args,
                 weights: Dict[str, float] = None,
                 log_space_parametrization: bool = False,
                 log_space_parametrization_multiplier: float = 10,
                 mixture_count = 1,
                 **kwargs):
        """
        `weights` should always be in log space
        `log_space_parametrization` controls whether the parameters are in log space or real space
        """
        super().__init__(*args, **kwargs)
        self.weights_dict = weights

        self.lsp = log_space_parametrization
        self.lsp_multipler = log_space_parametrization_multiplier
        self.mixture_count = mixture_count

        # represent weight parameters
        weights_tensor = torch.FloatTensor([weights[unit] for unit in self.vocab]).unsqueeze(1) if mixture_count == 1 else torch.FloatTensor([weights[unit] for unit in self.vocab])
        self.weights = nn.Embedding(num_embeddings=len(self.vocab),
                                    embedding_dim=mixture_count,
                                    padding_idx=self.vocab.index(self.pad_token),
                                    _weight=weights_tensor if self.lsp else weights_tensor.exp())
        self.reset_padding_weight()

    def clamp_weights(self, epsilon=1e-6) -> None:
        """
        If parametrized as real space, clamp the probabilities to [0,∞) after every update,
        mostly a trick than a principled approach, (i.e. should really do constrained-opt).
        """
        if self.lsp:
            raise ValueError("clamp_weights should only be used with real space parametrization")
        self.weights.weight.data = torch.clamp(self.weights.weight.data, min=epsilon)

    def reset_padding_weight(self) -> None:
        self.weights.weight.data[self.pad_index] = 1.0 if not self.lsp else 0.0

    def reset_specials_weight(self) -> None:
        self.weights.weight.data[self.specials_indices] = 1.0 if not self.lsp else 0.0

    def reset_singleton_weight(self) -> None:
        self.weights.weight.data[self.singleton_indices] = 1.0 if not self.lsp else 0.0

    def get_singleton_weight(self) -> torch.FloatTensor:
        return self.weights.weight.data[self.singleton_indices]

    def set_singleton_weight(self, singleton_weight) -> None:
        self.weights.weight.data[self.singleton_indices] = singleton_weight
    def get_weights(self, ids: torch.LongTensor):
        # get the weights from ids
        ts = self.weights(ids)
        if self.mixture_count == 1:
            ts = ts.squeeze(-1)
        # if parametrized as real space, do the log conversion since forward_algorithm is in log space
        if not self.lsp:
            ts = torch.log(ts)
        else:
            ts *= self.lsp_multipler
        return ts
    def forward(self, fwd_ids: torch.LongTensor,
                fwd_ms: torch.FloatTensor,
                lengths: torch.LongTensor,
                bwd_ids: torch.LongTensor,
                bwd_ms: torch.FloatTensor,
                bwd_lengths:  torch.LongTensor,
                mmask: torch.FloatTensor,
                emask: torch.FloatTensor,
                tmask: torch.FloatTensor=None,
                lm: bool=False,
                lm_mask: torch.FloatTensor=None,
                fwd_ts: torch.FloatTensor=None,
                bwd_ts: torch.FloatTensor=None,
                marginal_temperature: float=None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """

        fwd_ids: num_batch, num_block, max_unit_length, max_block_length
        fwd_ms: num_batch, num_block, max_unit_length, max_block_length
        lengths: num_batch, num_block
        bwd_ids: num_batch, num_block, max_block_length, max_unit_length, max_block_length
        bwd_ms: num_batch, num_block, max_block_length, max_unit_length, max_block_length
        bwd_lengths: num_batch, num_block
        mmask: num_batch, num_block, max_block_length, max_unit_length, max_block_length
        emask: num_batch, num_block, max_block_length, max_unit_length, max_block_length
        """
        # reshape inputs to have only one batch dimension
        num_batch, num_block = fwd_ids.size()[:2]
        fwd_ids = fwd_ids.reshape(-1, *fwd_ids.size()[2:])
        fwd_ms = fwd_ms.reshape(-1, *fwd_ms.size()[2:])
        lengths = lengths.reshape(-1, *lengths.size()[2:])
        bwd_ids = bwd_ids.reshape(-1, *bwd_ids.size()[2:])
        bwd_ms = bwd_ms.reshape(-1, *bwd_ms.size()[2:])
        bwd_lengths = bwd_lengths.reshape(-1, *bwd_lengths.size()[2:])
        mmask = mmask[0] # mmask and emask should be the same for all examples, so just take the first one (it already has an extra dimension)
        emask = emask[0] # mmask and emask should be the same for all examples, so just take the first one (it already has an extra dimension)
        device = self.weights.weight.data.device

        if fwd_ts is None:
            fwd_ts = self.get_weights(fwd_ids)
        else:
            fwd_ts = fwd_ts.reshape(-1, *fwd_ts.size()[2:])
        if bwd_ts is None:
            bwd_ts = self.get_weights(bwd_ids)
        else:
            bwd_ts = bwd_ts.reshape(-1, *bwd_ts.size()[2:])

        # handle mixture of lattices
        if self.mixture_count > 1:
            fwd_ts = fwd_ts.permute(fwd_ts.dim()-1, *range(fwd_ts.dim()-1)).reshape(-1, *fwd_ts.size()[1:fwd_ts.dim()-1]) # [B, M, L, K] -> [K, B, M, L] -> [KB, M, L]
            bwd_ts = bwd_ts.permute(bwd_ts.dim()-1, *range(bwd_ts.dim()-1)).reshape(-1, *bwd_ts.size()[1:bwd_ts.dim()-1])  # [B, M, L, K] -> [K, B, M, L] -> [KB, M, L]
            fwd_ms = fwd_ms[None, ...].expand(self.mixture_count, *fwd_ms.size()).reshape(-1, *fwd_ms.size()[1:])
            lengths = lengths[None, ...].expand(self.mixture_count, *lengths.size()).reshape(-1, *lengths.size()[1:])
            bwd_ms = bwd_ms[None, ...].expand(self.mixture_count, *bwd_ms.size()).reshape(-1, *bwd_ms.size()[1:])
            bwd_lengths = bwd_lengths[None, ...].expand(self.mixture_count, *bwd_lengths.size()).reshape(-1, *bwd_lengths.size()[1:])
            if tmask is not None:
                tmask = tmask[None, ...].expand(self.mixture_count, *tmask.size()).reshape(-1, *tmask.size()[1:])
            if lm_mask is not None:
                lm_mask = lm_mask[None, ...].expand(self.mixture_count, *lm_mask.size()).reshape(-1, *lm_mask.size()[1:])
            # mmask and emask are broadcast-ready so no need to expand they'll just broadcast themselves

        B, M, L = fwd_ts.size() # this is the effective B now, which is really self.mixture_count * num_batch * num_block

        # 1 forward pass
        log_alpha, edge_log_alpha, node_log_alpha = self.forward_algorithm(fwd_ts, fwd_ms, lengths, return_nodes=True)
        ent, edge_ent = self.entropy(fwd_ts, fwd_ms, lengths)

        # max_length backward passes, one from each position
        log_betas, edge_log_betas, node_log_betas_rev = self.forward_algorithm(bwd_ts, bwd_ms, bwd_lengths, return_nodes=True)
        log_betas = log_betas.reshape(B, L)
        edge_log_betas = edge_log_betas.reshape(B, L, M, L)

        # mask out the character suffixes in backward that are just auxiliary and don't really exist
        edge_log_betas = edge_log_betas * emask + torch.ones_like(edge_log_betas).fill_(-INF) * (1-emask)

        # compute conditionals
        c, ea, eb, em_, m = self.conditionals(fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, marginal_temperature=marginal_temperature, device=device)
        c = c.reshape(self.mixture_count * num_batch, num_block, c.size(-2), c.size(-1))
        m = m.reshape(self.mixture_count *num_batch, num_block, m.size(-1))
        a = self.tile(m, c, self.mixture_count * num_batch, num_block, M, L, fwd_ms, task_mask=tmask)
        ent = ent.reshape(self.mixture_count * num_batch, -1).sum(-1)
        if lm:
            a = self.tile_lm(em_, log_betas, m, a, self.mixture_count * num_batch, num_block, M, L, lm_mask)

        if (ent < -1e-3).any() or (ent.isnan().any()) or (ent.isinf().any()):
            print(f"Bug detected in entropy! Negative entropy! {ent}")
            # code.interact(local=locals())
        if (a > 1e-3).any():
            print("Bug detected in entropy! Greater than one marginals!")
            code.interact(local=locals())
        ent = torch.maximum(ent, torch.zeros_like(ent))
        a = torch.minimum(a, torch.zeros_like(a))

        # handle mixture of lattices
        if self.mixture_count > 1:
            normalized_n = (node_log_alpha + node_log_betas_rev.flip(-1).reshape(B, L, -1)[:, -1, :])[:,:-1] - log_alpha[:, None]
            marginal_n = normalized_n.reshape(self.mixture_count, num_batch * num_block, *normalized_n.size()[1:]).logsumexp(dim=0) - math.log(self.mixture_count)

            normalized_m = m
            marginal_m = normalized_m.reshape(self.mixture_count, num_batch * num_block, *normalized_m.size()[1:]).logsumexp(dim=0) - math.log(self.mixture_count)
            marginal_m = marginal_m[..., self.inverse_permutation(L, M)]

            triu_ones = torch.triu(torch.ones(M, L, dtype=torch.bool).to(device), diagonal=0)[None, ...].expand(num_batch * num_block, -1, -1)
            marginal_m_matrix = fwd_ts.new_zeros(num_batch * num_block, *fwd_ts.size()[1:]) * 0.0
            marginal_m_matrix[triu_ones] = marginal_m.reshape(-1)
            marginal_c_matrix = increasing_roll_right(increasing_roll_left(marginal_m_matrix, padding_value=-INF) - marginal_n[:, None, :], padding_value=-INF)

            marginal_m_matrix[~fwd_ms.reshape(self.mixture_count, num_batch * num_block, M, L)[0].to(torch.bool)] = -INF
            marginal_c_matrix[~fwd_ms.reshape(self.mixture_count, num_batch * num_block, M, L)[0].to(torch.bool)] = -INF

            marginal_ent = -(marginal_m_matrix[triu_ones].double().exp() * marginal_c_matrix[triu_ones].double()).reshape(num_batch * num_block, -1).sum(-1)
            # normalized_fwd_ts = self.forward_normalize(fwd_ts, fwd_ms, lengths, bwd_ts, bwd_ms, bwd_lengths, mmask, emask, device=device, m=m)
            return ent.reshape(self.mixture_count, num_batch * num_block, *ent.size()[1:]), \
                   a.reshape(self.mixture_count, num_batch * num_block, *a.size()[1:]), \
                   m.reshape(self.mixture_count, num_batch * num_block, *m.size()[1:]), \
                   c.reshape(self.mixture_count, num_batch * num_block, *c.size()[1:]), \
                   marginal_ent, \
                   marginal_c_matrix
        return ent, a, m, c,

    def forward_normalize(self,
                fwd_ts: torch.FloatTensor,
                fwd_ms: torch.FloatTensor,
                lengths: torch.LongTensor,
                bwd_ts: torch.FloatTensor,
                bwd_ms: torch.FloatTensor,
                bwd_lengths:  torch.LongTensor,
                mmask: torch.FloatTensor,
                emask: torch.FloatTensor,
                device="cpu",
                m=None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if m is None:
            B, M, L = fwd_ts.size()  # this is the effective B now, which is really self.mixture_count * num_batch * num_block

            # 1 forward pass
            log_alpha, edge_log_alpha = self.forward_algorithm(fwd_ts, fwd_ms, lengths)
            ent, edge_ent = self.entropy(fwd_ts, fwd_ms, lengths)

            # max_length backward passes, one from each position
            log_betas, edge_log_betas = self.forward_algorithm(bwd_ts, bwd_ms, bwd_lengths)
            log_betas = log_betas.reshape(B, L)
            edge_log_betas = edge_log_betas.reshape(B, L, M, L)

            # mask out the character suffixes in backward that are just auxiliary and don't really exist
            edge_log_betas = edge_log_betas * emask + torch.ones_like(edge_log_betas).fill_(-INF) * (1 - emask)

            _,_,_,_,m = self.conditionals(fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, device=device)

        B, M, L = fwd_ts.size()
        m = m[..., self.inverse_permutation(L, M)]
        triu_ones = torch.triu(torch.ones(M, L, dtype=torch.bool).to(device), diagonal=0)
        normalized_fwd_ts = torch.zeros_like(fwd_ts)
        normalized_fwd_ts[triu_ones[None, ...].expand(B, -1, -1)] = m.reshape(-1)

        bmask: torch.BoolTensor = fwd_ms.to(torch.bool)
        edge_log_alphas: torch.FloatTensor = torch.ones_like(fwd_ts).fill_(-INF)
        edge_log_alphas[bmask] = 0.0
        edge_log_alphas += fwd_ts

        log_alphas: List[torch.FloatTensor] = [fwd_ms.new_zeros(B)]
        for i in range(L):
            maski = (bmask & torch.diag_embed(fwd_ms.new_ones(L - i, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(
                torch.float)
            node_to_edge = log_alphas[i][:, None, None] * maski
            edge_log_alphas = edge_log_alphas + node_to_edge
            normalized_fwd_ts = normalized_fwd_ts - node_to_edge # divide the edge marginals m by alpha_node

            log_alphas.append(torch.logsumexp(edge_log_alphas[:, :, i], -1))
        return normalized_fwd_ts