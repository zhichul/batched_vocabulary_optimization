from dataclasses import dataclass
from typing import List, Union

import torch
import torch.nn as nn

@dataclass
class UnigramLMTokenizerOutput:
    input_ids: torch.Tensor = None          # always set
    attention_mask: torch.Tensor = None     # always set
    position_ids: torch.Tensor = None       # always set
    type_ids: torch.Tensor = None           # always set
    attention_bias: torch.Tensor = None     # set when in full lattice mode
    weights: torch.Tensor = None            # set when each input sentence produces multiple tokenizations
    entropy: torch.Tensor = None            # set when requested
class AbstractUnigramLMTokenizer(nn.Module):

    def forward(self, sentences: Union[List[str], List[List[str]]]):
        """
        This method tokenizes a batch of sentences or sentence lists. It returns
        tensors that match the API of common downstream neural models.
        """
        input_ids, attention_mask, position_ids, type_ids, attention_bias, weights, entropy = None, None, None, None, None, None, None, None
        return UnigramLMTokenizerOutput(input_ids, attention_mask, position_ids, type_ids, attention_bias, weights, entropy)


