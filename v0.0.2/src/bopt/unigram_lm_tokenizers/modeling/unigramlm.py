import code
import math
import os
from typing import List

import torch
import torch.nn as nn

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID, NONEDGE_LOGPOT, PADEDGE_LOGPOT


class UnigramLM(nn.Module):

    def __init__(self, vocabulary_size, pretrained_log_potentials=None, log_space_parametrization=False):
        super().__init__()
        self.edge_log_potentials = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=1, _weight=pretrained_log_potentials)
        self.log_space_parametrization = log_space_parametrization

        if pretrained_log_potentials is None:
            self.edge_log_potentials.weight.data[...] = 0
        if not log_space_parametrization:
            self.edge_log_potentials.weight.data = self.edge_log_potentials.weight.data.exp()
            self.clamp_weights()

    def forward(self, lattice_encoding: torch.Tensor, temperature=1.0):
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

        if not self.log_space_parametrization:
            edge_log_potentials = edge_log_potentials.log()

        edge_log_potentials[nonedge_mask] = NONEDGE_LOGPOT
        edge_log_potentials[padedge_mask] = PADEDGE_LOGPOT

        return edge_log_potentials * temperature

    @property
    def device(self):
        return self.edge_log_potentials.weight.data.device

    def set(self, indices: List[int], value: List[float]):
        for i, v in zip(indices, value):
            self.edge_log_potentials.weight.data[0, i] = v

    def clamp_weights(self):
        # if not log space parametrization make sure edge potentials are nonnegative
        if not self.log_space_parametrization:
            self.edge_log_potentials.weight.data.clamp_(min=1e-6)

    def l1(self, avoid_indices=list()):
        if self.log_space_parametrization:
            real_weights = self.edge_log_potentials.weight.exp()
        else:
            real_weights = self.edge_log_potentials.weight
        return ((real_weights.sum() -
                real_weights[avoid_indices,:].sum())
                / (real_weights.size(1) * (real_weights.size(0) - len(avoid_indices))))

    def unigram_p(self, temperature=1.0):
        if self.log_space_parametrization:
            return torch.softmax(self.edge_log_potentials.weight.data[:,0] * temperature, -1)
        else:
            return torch.softmax(self.edge_log_potentials.weight.data[:,0].log() * temperature, -1)

    def log_weights(self):
        if self.log_space_parametrization:
            log_weights = self.edge_log_potentials.weight.data
        else:
            log_weights = self.edge_log_potentials.weight.data.log()
        return log_weights

    def save_to_folder(self, folder, vocabulary):
        with open(os.path.join(folder, "learned_vocab.txt"), "wt") as f:
            log_weights = self.log_weights().tolist()
            for v, w in zip(vocabulary, log_weights):
                weght_str = '\t'.join([f'{i}' for i in w])
                print(f"{v}\t{weght_str}", file=f)
