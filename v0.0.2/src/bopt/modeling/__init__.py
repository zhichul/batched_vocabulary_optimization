import os
from dataclasses import dataclass
from typing import Optional, Any

from bopt.modeling.modeling_bert import BertConfig, BertForMaskedLM


def load_model(config, pad_token_id=0, saved_model=None, bias_mode=None):
    if saved_model is not None:
        config = BertConfig.from_json_file(os.path.join(saved_model, "config.json"))
        model = BertForMaskedLM.from_pretrained(saved_model)
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