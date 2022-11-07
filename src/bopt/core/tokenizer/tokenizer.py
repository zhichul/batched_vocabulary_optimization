from typing import Dict, Tuple

import torch
import code
from torch import nn

from bopt.core.tokenizer.attention import LatticeAttentionMixin
from bopt.core.tokenizer.dynamic import LatticeDPMixin
from bopt.core.tokenizer.tokenization import TokenizationMixin

INF = 1e9

class Tokenizer(TokenizationMixin, LatticeDPMixin, LatticeAttentionMixin, nn.Module):

    def __init__(self, *args,
                 weights: Dict[str, float] = None,
                 log_space_parametrization: bool = False,
                 log_space_parametrization_multiplier: float = 10,
                 **kwargs):
        """
        `weights` should always be in log space
        `log_space_parametrization` controls whether the parameters are in log space or real space
        """
        super().__init__(*args, **kwargs)
        self.weights_dict = weights

        self.lsp = log_space_parametrization
        self.lsp_multipler = log_space_parametrization_multiplier

        # represent weight parameters
        weights_tensor = torch.FloatTensor([weights[unit] for unit in self.vocab]).unsqueeze(1)
        self.weights = nn.Embedding(num_embeddings=len(self.vocab),
                                    embedding_dim=1,
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
        ts = self.weights(ids).squeeze(-1)
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


        B, M, L = fwd_ts.size()

        # 1 forward pass
        log_alpha, edge_log_alpha = self.forward_algorithm(fwd_ts, fwd_ms, lengths)
        ent, edge_ent = self.entropy(fwd_ts, fwd_ms, lengths)

        # max_length backward passes, one from each position
        log_betas, edge_log_betas = self.forward_algorithm(bwd_ts, bwd_ms, bwd_lengths)
        log_betas = log_betas.reshape(B, L)
        edge_log_betas = edge_log_betas.reshape(B, L, M, L)

        # mask out the character suffixes in backward that are just auxiliary and don't really exist
        edge_log_betas = edge_log_betas * emask + torch.ones_like(edge_log_betas).fill_(-INF) * (1-emask)

        # compute conditionals
        c, ea, eb, em_, m = self.conditionals(fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, marginal_temperature=marginal_temperature, device=device)
        c = c.reshape(num_batch, num_block, c.size(-2), c.size(-1))
        m = m.reshape(num_batch, num_block, m.size(-1))
        a = self.tile(m, c, num_batch, num_block, M, L, fwd_ms, task_mask=tmask)
        ent = ent.reshape(num_batch, -1).sum(-1)
        if lm:
            a = self.tile_lm(em_, log_betas, m, a, num_batch, num_block, M, L, lm_mask)

        if (ent < -1e-3).any() or (ent.isnan().any()) or (ent.isinf().any()):
            print(f"Bug detected in entropy! Negative entropy! {ent}")
            # code.interact(local=locals())
        if (a > 1e-3).any():
            print("Bug detected in entropy! Greater than one marginals!")
            code.interact(local=locals())
        ent = torch.maximum(ent, torch.zeros_like(ent))
        a = torch.minimum(a, torch.zeros_like(a))
        return ent, a, m, c