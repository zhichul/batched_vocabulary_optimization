import os

from bopt.unigram_lm_tokenizers.inference.entropy import entropy
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, length, NONEDGE_LOGPOT
from bopt.unigram_lm_tokenizers.encoding.linearized_encoding import extract_input_ids, extract_position_ids, \
    extract_attention_mask, extract_type_ids
from bopt.unigram_lm_tokenizers.inference.attention import attention_bias
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM

from typing import Union, List

import torch.nn as nn
import torch

from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput


class LatticeTokenizer(nn.Module):

    def __init__(self, unigramlm, vocabulary):
        super().__init__()
        self.unigramlm = unigramlm
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
                remove_space: bool = False,
                memoizer = None,
                sentence_ids = None,
                specials=set(),
                try_word_initial_when_unk=False,
                pad_token_id=0,
                subsample_vocab=None,
                temperature=1.0,
                collapse_padding=False,
                output_inputs=False):
        if memoizer is None != sentence_ids is None: raise ValueError(
            "memoizer and sentence_ids have to be set at the same time")
        forward_encodings, input_ids, position_ids, attention_mask, type_ids, B, N, M, L, K = self.extract_encodings(
            sentences=sentences,
            max_blocks = max_blocks,
            max_unit_length = max_unit_length,
            max_block_length = max_block_length,
            space_character = space_character,
            split_on_space = split_on_space,
            add_dummy_space_start = add_dummy_space_start,
            remove_space = remove_space,
            memoizer = memoizer,
            sentence_ids = sentence_ids,
            specials = specials,
            try_word_initial_when_unk = try_word_initial_when_unk,
            pad_token_id = pad_token_id,
            subsample_vocab = subsample_vocab,
            temperature = temperature
            )
        # compute attention
        edge_log_potentials = self.unigramlm(forward_encodings, temperature=temperature) # B x KN x M x L
        attention = attention_bias(attention_mask, edge_log_potentials) # B x KNE x KNE

        # efficiency improvement
        if collapse_padding:
            max_length = attention_mask.sum(dim=-1).max()
            new_input_ids = input_ids.new_zeros(input_ids.size(0), max_length)
            new_position_ids = input_ids.new_zeros(input_ids.size(0), max_length)
            new_attention_mask = input_ids.new_zeros(input_ids.size(0), max_length)
            new_type_ids = input_ids.new_zeros(input_ids.size(0), max_length)
            new_attention = attention.new_zeros(input_ids.size(0), max_length, max_length).fill_(NONEDGE_LOGPOT)
            new_attention[:, list(range(max_length)), list(range(max_length))] = 0.0
            for i in range(input_ids.size(0)):
                nonpad = attention_mask[i].to(torch.bool)
                l = nonpad.sum().item()
                new_input_ids[i, :l] = input_ids[i, nonpad]
                new_position_ids[i, :l] = position_ids[i, nonpad]
                new_attention_mask[i, :l] = attention_mask[i, nonpad]
                new_type_ids[i, :l] = type_ids[i, nonpad]
                new_attention[i, :l, :l] = attention[i, nonpad][:, nonpad]
            input_ids, position_ids, attention_mask, type_ids, attention = new_input_ids, new_position_ids, new_attention_mask, new_type_ids, new_attention

        # compute and normalize entropy
        ent = entropy(edge_log_potentials) # B x KN
        lengths = length(forward_encodings) # B x KN

        ent_scalar = ent.sum() / lengths.sum() # 1
        return UnigramLMTokenizerOutput(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        type_ids=type_ids,
                                        attention_bias=attention,
                                        entropy=ent_scalar,
                                        nchars=lengths.sum().item(),
                                        edge_log_potentials=edge_log_potentials if output_inputs else None,
                                        forward_encodings=forward_encodings if output_inputs else None)

    def extract_encodings(self, sentences: Union[List[str], List[List[str]]],
                max_blocks = 1,
                max_unit_length = 12,
                max_block_length = 8,
                space_character: str = "▁",
                split_on_space: bool = True,
                add_dummy_space_start: bool = True,
                remove_space: bool = False,
                memoizer = None,
                sentence_ids = None,
                specials=set(),
                try_word_initial_when_unk=False,
                pad_token_id=0,
                subsample_vocab=None,
                temperature=1.0):
        B, N, M, L, K = len(sentences), max_blocks, max_unit_length, max_block_length, 1
        if isinstance(sentences[0], list):
            K = len(sentences[0])
            sentences = sum(sentences, [])
            if sentence_ids:
                sentence_ids = sum(sentence_ids, [])
        if self.training and subsample_vocab is not None:
            # only subsample in training
            vocab = self.vocabulary.subsample(subsample_vocab,self.unigramlm.unigram_p(temperature=temperature).tolist())
        else:
            vocab = self.vocabulary
        forward_encodings = integerize_for_forward(sentences, N, M, L, vocab,
                                                   space_character=space_character,
                                                   split_on_space=split_on_space,
                                                   add_dummy_space_start=add_dummy_space_start,
                                                   remove_space=remove_space,
                                                   memoizer=memoizer,
                                                   sentence_ids=sentence_ids,
                                                   specials=specials,
                                                   try_word_initial_when_unk=try_word_initial_when_unk).to(self.device).reshape(B, K*N, M, L) # B x KN x M x L

        # extract linearized ids
        input_ids = extract_input_ids(forward_encodings, padding_id=pad_token_id) # B x KNE
        position_ids = extract_position_ids(forward_encodings) # B x KNE
        attention_mask = extract_attention_mask(forward_encodings) # B x KNE
        type_ids = extract_type_ids(forward_encodings.reshape(B,K,N,M,L)) # B x KNE
        return forward_encodings, input_ids, position_ids, attention_mask, type_ids, B, N, M, L, K

    def l1(self, avoid_tokens=tuple()):
        return self.unigramlm.l1(avoid_indices=[self.vocabulary.index(token) for token in avoid_tokens])

    def clamp_weights(self):
        self.unigramlm.clamp_weights()
    def save_to_folder(self, folder):
        self.unigramlm.save_to_folder(folder, self.vocabulary)

