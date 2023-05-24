import os
from dataclasses import dataclass
from typing import Optional, Any

import torch

from bopt.modeling.modeling_bert import BertConfig, BertForMaskedLM


def load_model(config, pad_token_id=0, saved_model=None, bias_mode=None, ignore=None, include=None):
    # if include is set will first filter by include then filter by ignore
    if saved_model is not None:
        # allows overriding configuration file
        state_dict = torch.load(os.path.join(saved_model, "pytorch_model.bin"))
        if include is not None:
            state_dict = {k: v for k, v in state_dict.items() if k in include}
        if ignore is not None:
            state_dict = {k:v for k,v in state_dict.items() if k not in ignore}
        model = BertForMaskedLM.from_pretrained(None, state_dict=state_dict, config=config)
        config = model.config
    else:
        config = BertConfig.from_json_file(config)
        config.pad_token_id = pad_token_id
        model = BertForMaskedLM(config)

    model.bias_mode = bias_mode
    return model, config

@dataclass
class Regularizers:

    l1:Optional[Any] = None
    entropy:Optional[Any] = None
    nchars:Optional[int] = None