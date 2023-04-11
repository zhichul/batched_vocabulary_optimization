import math

import torch

from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, NONEDGE_ID
from bopt.unigram_lm_tokenizers.encoding.linearized_encoding import extract_input_ids, extract_attention_mask
from bopt.unigram_lm_tokenizers.inference.attention import attention_bias
from bopt.unigram_lm_tokenizers.inference.forward_backward import conditional_marginals
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.utils.encoding import convert_to_forward_encoding, expand_encodings, \
    convert_to_backward_encoding
from bopt.unigram_lm_tokenizers.utils.printing import print_lattice, print_attention


def test_attention():
    print("Test Forward Backard")
    vocabulary = Integerizer(
        [
            "[UNK]",
            "h",
            "a",
            "t",
            "e",
            "hat",
            "hate",
            "at",
            "ate",
        ]
    )
    log_potentials = torch.tensor([math.log(2.0)] * len(vocabulary)).unsqueeze(-1)
    encoding = integerize_for_forward(["hate hat"], 2, 4, 5, vocabulary, space_character=" ", split_on_space=True,
                                       add_dummy_space_start=False, remove_space=True)
    print_lattice(encoding, vocabulary)
    unigramlm = UnigramLM(len(vocabulary), log_potentials)
    output_potentials = unigramlm(encoding)
    print("Edge (log) potentials")
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)
    attn = attention_bias(extract_attention_mask(encoding), output_potentials)
    print_attention(extract_input_ids(encoding, padding_id=-1), vocabulary, attn)
if __name__ == "__main__":
    test_attention()