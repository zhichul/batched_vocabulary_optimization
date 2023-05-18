import code
import os

from bopt.unigram_lm_tokenizers import LatticeTokenizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, length, NONEDGE_ID, PADEDGE_ID
from bopt.unigram_lm_tokenizers.encoding.linearized_encoding import extract_input_ids, extract_position_ids, \
    extract_attention_mask, extract_token_encoding
from bopt.unigram_lm_tokenizers.inference.entropy import entropy
from bopt.unigram_lm_tokenizers.inference.viterbi import viterbi_nbest

from typing import Union, List

import torch.nn as nn
import torch

from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput


class NBestTokenizer(LatticeTokenizer):

    def forward(self, sentences: Union[List[str], List[List[str]]],
                n=1,
                use_lattice_position_ids=False,
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
            try_word_initial_when_unk=try_word_initial_when_unk,
            pad_token_id = pad_token_id,
            subsample_vocab=subsample_vocab,
            temperature=temperature
            )

        # compute nbest
        edge_log_potentials = self.unigramlm(forward_encodings, temperature=temperature) # B x KN x M x L
        viterbi_nbest_output = viterbi_nbest(edge_log_potentials, n=n) # B x KN x n x M x L
        nbest_forward_encodings = forward_encodings.unsqueeze(2).expand_as(viterbi_nbest_output.mask).clone() # materialize so no list bugs
        nbest_forward_encodings[~viterbi_nbest_output.mask] = NONEDGE_ID
        nbest_forward_encodings = nbest_forward_encodings.transpose(1,2).reshape(B,n,K,N,M,L) # B x n x K x N x M x L

        # extract integer ids
        nbest_input_ids, nbest_attention_mask, nbest_position_ids, nbest_type_ids = extract_token_encoding(nbest_forward_encodings,
                                                                    use_lattice_position_ids=use_lattice_position_ids) # B x n x seq_length
        weight = viterbi_nbest_output.weight.sum(1) # B x KN x n -> B x n

        # compute and normalize entropy
        ent = entropy(edge_log_potentials) # B x KN
        lengths = length(forward_encodings) # B x KN

        ent_scalar = ent.sum() / lengths.sum() # 1
        return UnigramLMTokenizerOutput(input_ids=nbest_input_ids,
                                        attention_mask=nbest_attention_mask,
                                        position_ids=nbest_position_ids,
                                        type_ids=nbest_type_ids,
                                        attention_bias=None,
                                        entropy=ent_scalar,
                                        nchars=lengths.sum().item(),
                                        weights=weight)


