from typing import Dict, Tuple

import torch
from torch import nn

from bopt.core.tokenizer.attention import LatticeAttentionMixin
from bopt.core.tokenizer.dynamic import LatticeDPMixin
from bopt.core.tokenizer.tokenization import TokenizationMixin

INF = 1e9

class Tokenizer(TokenizationMixin, LatticeDPMixin, LatticeAttentionMixin, nn.Module):

    def __init__(self, *args,
                 weights: Dict[str, float] = None,
                 log_space_parametrization: bool = False,
                 **kwargs):
        """
        `weights` should always be in log space
        `log_space_parametrization` controls whether the parameters are in log space or real space
        """
        super().__init__(*args, **kwargs)
        self.weights_dict = weights

        self.lsp = log_space_parametrization

        # represent weight parameters
        weights_tensor = torch.FloatTensor([weights[unit] for unit in self.vocab]).unsqueeze(1)
        self.weights = nn.Embedding(num_embeddings=len(self.vocab),
                                    embedding_dim=1,
                                    padding_idx=self.vocab.index(self.pad_token),
                                    _weight=weights_tensor if self.lsp else weights_tensor.exp())

    def clamp_weights(self, epsilon=1e-9) -> None:
        """
        If parametrized as real space, clamp the probabilities to [0,∞) after every update,
        mostly a trick than a principled approach, (i.e. should really do constrained-opt).
        """
        if self.lsp:
            raise ValueError("clamp_weights should only be used with real space parametrization")
        self.weights.weight.data = torch.clamp(self.weights.weight.data, min=epsilon)


    def forward(self, fwd_ids: torch.LongTensor,
                fwd_ms: torch.FloatTensor,
                lengths: torch.LongTensor,
                bwd_ids: torch.LongTensor,
                bwd_ms: torch.FloatTensor,
                bwd_lengths:  torch.LongTensor,
                mmask: torch.FloatTensor,
                emask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = self.weights.weight.data.device

        # get the weights from ids
        fwd_ts = self.weights(fwd_ids).squeeze(-1)
        bwd_ts = self.weights(bwd_ids).squeeze(-1)

        # if parametrized as real space, do the log conversion since forward_algorithm is in log space
        if not self.lsp:
            fwd_ts = torch.log(fwd_ts)
            bwd_ts = torch.log(bwd_ts)

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
        c, ea, eb, em_, em = self.conditionals( fwd_ts, fwd_ms, log_alpha, edge_log_alpha, log_betas, edge_log_betas, device=device)
        return ent, log_alpha, c