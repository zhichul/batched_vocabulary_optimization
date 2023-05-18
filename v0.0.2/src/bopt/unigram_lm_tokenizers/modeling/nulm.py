import os
from typing import List

import torch
import torch.nn as nn

from .unigramlm import UnigramLM


class NeuralUnigramLM(nn.Module):

    def __init__(self, vocabulary_size, num_hidden_layers=1, hidden_size=768):
        """
        Currently only supports one unigram distribution at a time, and
        always parametrizes the edge weights in log space.
        This is NOT really a subclass of UnigramLM but the intention is
        to not have duplicate code. Inherits only forward and reuses save.
        """
        super().__init__()
        self.edge_embeddings = nn.Embedding(num_embeddings=vocabulary_size,
                                    embedding_dim=hidden_size)
        self.MLP = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.MLP.append(nn.Linear(hidden_size, hidden_size))
            self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(hidden_size, 1))
        self.index = torch.arange(vocabulary_size)
        self.log_space_parametrization = True

    def edge_log_potentials(self, ids):
        output = self.edge_embeddings(ids)
        for layer in self.MLP:
            output = layer(output)
        return output

    def edge_log_potentials_all(self):
        if self.index.device != self.device:
            self.index = self.index.to(self.device)
        return self.edge_log_potentials(self.index)

    def forward(self, *args, **kwargs):
        outputs = UnigramLM.forward(self, *args, **kwargs)
        return outputs

    @property
    def device(self):
        return self.edge_embeddings.weight.data.device

    def l1(self, avoid_indices=list()):
        real_weights = self.edge_log_potentials_all().exp()
        return ((real_weights.sum() - real_weights[avoid_indices].sum())
                / (real_weights.size(0) - len(avoid_indices)))

    def unigram_p(self, temperature=1.0):
        return torch.softmax(self.edge_log_potentials_all() * temperature, -1)

    def log_weights(self):
        log_weights = self.edge_log_potentials_all()
        return log_weights

    def save_to_folder(self, folder, vocabulary):
        UnigramLM.save_to_folder(self, folder, vocabulary)
        torch.save(self.state_dict(), os.path.join(folder, "nulm_model.bin"))

    # to early fail on bugs
    def set(self, indices: List[int], value: List[float]):
        raise NotImplementedError

    def clamp_weights(self):
        pass # do nothing
