import math

import torch
import torch.nn as nn

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID


class UnigramLM(nn.Module):

    def __init__(self, vocabulary_size, pretrained_log_potentials=None):
        super().__init__()
        self.edge_log_potentials = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=1,
                                    _weight=pretrained_log_potentials)

    def forward(self, lattice_encoding: torch.Tensor):
        """
        Given a batch of edge id matrices (forward or backward),
        return a batch of edge log potential matrices, where NONEDGE_IDs have
        weight -inf, PADEDGE_IDs have weight 0, and the rest have weight based
        on self.edge_log_potentials
        """
        nonedge_mask = (lattice_encoding == NONEDGE_ID).to(torch.long)
        nonedge_mask_b = nonedge_mask.to(torch.bool)
        padedge_mask = (lattice_encoding == PADEDGE_ID).to(torch.long)
        padedge_mask_b = padedge_mask.to(torch.bool)
        edge_mask = (1 - (nonedge_mask + padedge_mask))
        edge_mask_f = edge_mask.to(torch.float)

        base_weight = torch.zeros_like(lattice_encoding, dtype=torch.float)
        base_weight[nonedge_mask_b] = -math.inf
        base_weight[padedge_mask_b] = 0.0
        edge_weight = edge_mask_f * self.edge_log_potentials(edge_mask * lattice_encoding).squeeze(-1)

        return base_weight + edge_weight