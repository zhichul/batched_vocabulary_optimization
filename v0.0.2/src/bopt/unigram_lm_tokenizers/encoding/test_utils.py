from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.backward_encoding import integerize_for_backward
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
from bopt.unigram_lm_tokenizers.inference.forward_backward import forward_algorithm
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.encoding.utils import print_lattice, expand_encodings

import torch

def test():
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
    log_potentials = torch.tensor([0.0] * len(vocabulary)).unsqueeze(-1)
    unigramlm = UnigramLM(len(vocabulary), log_potentials)
    encoding1 = integerize_for_forward(["hate"], 1, 5, 4, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False)
    encoding2 = integerize_for_backward(["hate"], 1, 5, 4, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False)
    for encoding in [expand_encodings(encoding1), expand_encodings(encoding2)]:
        expanded_encoding = encoding.reshape(-1, 1, 4, 5)
        print_lattice(encoding.reshape(-1, 1, 4, 5), vocabulary, log_potentials=unigramlm(expanded_encoding))

if __name__ == "__main__":
    test()