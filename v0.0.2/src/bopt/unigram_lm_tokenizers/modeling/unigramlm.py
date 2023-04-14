import math
from typing import List

import torch
import torch.nn as nn

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID, NONEDGE_LOGPOT, PADEDGE_LOGPOT


class UnigramLM(nn.Module):

    def __init__(self, vocabulary_size, pretrained_log_potentials=None):
        super().__init__()
        self.edge_log_potentials = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=1,
                                    _weight=pretrained_log_potentials)
        if pretrained_log_potentials is None:
            self.edge_log_potentials.weight.data[...] = 0

    def forward(self, lattice_encoding: torch.Tensor):
        """
        Given a batch of edge id matrices (forward or backward),
        return a batch of edge log potential matrices, where NONEDGE_IDs have
        weight -inf, PADEDGE_IDs have weight 0, and the rest have weight based
        on self.edge_log_potentials
        """
        nonedge_mask = lattice_encoding == NONEDGE_ID
        padedge_mask = lattice_encoding == PADEDGE_ID
        edge_mask = (~(nonedge_mask | padedge_mask)).to(torch.long)
        edge_log_potentials = self.edge_log_potentials(edge_mask * lattice_encoding).squeeze(-1)

        edge_log_potentials[nonedge_mask] = NONEDGE_LOGPOT
        edge_log_potentials[padedge_mask] = PADEDGE_LOGPOT

        return edge_log_potentials

    @property
    def device(self):
        return self.edge_log_potentials.weight.data.device

    def set(self, indices: List[int], value: List[float]):
        for i, v in zip(indices, value):
            self.edge_log_potentials.weight.data[0, i] = v

    def l1(self, avoid_indices=list()):
        return ((self.edge_log_potentials.weight.exp().sum() -
                self.edge_log_potentials.weight[avoid_indices,:].exp().sum())
                / (self.edge_log_potentials.weight.size(1) * (self.edge_log_potentials.weight.size(0) - len(avoid_indices))))

    def unigram_p(self):
        return torch.softmax(self.edge_log_potentials.weight.data[:,0])