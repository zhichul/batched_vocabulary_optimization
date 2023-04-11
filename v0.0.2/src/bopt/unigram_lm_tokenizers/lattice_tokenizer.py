from bopt.unigram_lm_tokenizers.inference.entropy import entropy
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, length
from bopt.unigram_lm_tokenizers.encoding.linearized_encoding import extract_input_ids, extract_position_ids, \
    extract_attention_mask
from bopt.unigram_lm_tokenizers.inference.attention import attention_bias
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM

from typing import Union, List

import torch.nn as nn
import torch

from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput


class LatticeTokenizer(nn.Module):

    def __init__(self, vocabulary, pretrained_log_potentials=None):
        super().__init__()
        self.unigramlm = UnigramLM(len(vocabulary), pretrained_log_potentials=pretrained_log_potentials)
        self.vocabulary = vocabulary

    @property
    def device(self):
        return self.unigramlm.device

    def forward(self, sentences: Union[List[str], List[List[str]]],
                max_blocks = 1,
                max_unit_length = 12,
                max_block_length = 8,
                space_character: str = "▁",
                split_on_space: bool = True,
                add_dummy_space_start: bool = True,
                remove_space: bool = False):
        B, N, M, L, K = len(sentences), max_blocks, max_unit_length, max_block_length, 1
        if isinstance(sentences[0], list):
            K = len(sentences[0])
            sentences = sum(sentences, [])
        forward_encodings = integerize_for_forward(sentences, N, M, L, self.vocabulary,
                                                   space_character=space_character,
                                                   split_on_space=split_on_space,
                                                   add_dummy_space_start=add_dummy_space_start,
                                                   remove_space=remove_space).to(self.device).reshape(B, K*N, M, L) # B x KN x M x L

        # extract linearized ids
        input_ids = extract_input_ids(forward_encodings, padding_id=0) #TODO: make depend on vocab B x KNE
        position_ids = extract_position_ids(forward_encodings) # B x KNE
        attention_mask = extract_attention_mask(forward_encodings) # B x KNE
        NE = input_ids.size(-1) // (K)
        type_ids = torch.arange(K, dtype=torch.long, device=self.device)[None,:,None].expand(B,K,1).reshape(B*K,-1).expand(B*K, NE).reshape(B, K*NE) # B x KNE

        # compute attention
        edge_log_potentials = self.unigramlm(forward_encodings) # B x KN x M x L
        attention = attention_bias(attention_mask, edge_log_potentials) # B x KNE x KNE

        # compute and normalize entropy
        ent = entropy(edge_log_potentials) # B x KN
        lengths = length(forward_encodings) # B x KN

        ent_scalar = ent.sum() / lengths.sum() # 1
        return UnigramLMTokenizerOutput(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        type_ids=type_ids,
                                        attention_bias=attention,
                                        entropy=ent_scalar)





