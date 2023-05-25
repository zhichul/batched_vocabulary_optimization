from dataclasses import dataclass
from typing import List, Union, Optional

import torch
import torch.nn as nn

@dataclass
class UnigramLMTokenizerOutput:
    input_ids: Optional[torch.Tensor] = None          # always set
    attention_mask: Optional[torch.Tensor] = None     # always set
    position_ids: Optional[torch.Tensor] = None       # always set
    type_ids: Optional[torch.Tensor] = None           # always set
    attention_bias: Optional[torch.Tensor] = None     # set when in full lattice mode
    weights: Optional[torch.Tensor] = None            # set when each input sentence produces multiple tokenizations
    entropy: Optional[torch.Tensor] = None            # set when requested
    nchars: Optional[torch.Tensor] = None             # set when requested
    edge_log_potentials: Optional[torch.Tensor] = None  # set when requested
    forward_encodings: Optional[torch.Tensor] = None  # set when requested

class AbstractUnigramLMTokenizer(nn.Module):

    def forward(self, sentences: Union[List[str], List[List[str]]]):
        """
        This method tokenizes a batch of sentences or sentence lists. It returns
        tensors that match the API of common downstream neural models.
        """
        input_ids, attention_mask, position_ids, type_ids, attention_bias, weights, entropy, nchars, edge_log_potentials, forward_encodings = None, None, None, None, None, None, None, None, None, None, None, None
        return UnigramLMTokenizerOutput(input_ids, attention_mask, position_ids, type_ids, attention_bias, weights, entropy, nchars, edge_log_potentials, forward_encodings)


