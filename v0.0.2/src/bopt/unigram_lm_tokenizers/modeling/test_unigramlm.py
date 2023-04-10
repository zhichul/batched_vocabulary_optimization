from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
import traceback
import torch

from bopt.unigram_lm_tokenizers.encoding.utils import print_lattice, convert_to_backward_encoding
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM


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

    # forward encoding
    encoding = integerize_for_forward(["hate"], 1, 5, 4, vocabulary,
                                      space_character=" ", split_on_space=False, add_dummy_space_start=False)
    print_lattice(encoding, vocabulary)
    print(encoding.size())
    output_potentials = unigramlm(encoding)
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)

    # backward encoding
    encoding = convert_to_backward_encoding(encoding)
    print_lattice(encoding, vocabulary)
    print(encoding.size())
    output_potentials = unigramlm(encoding)
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)

def test1():
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

    # forward encoding
    encoding = integerize_for_forward(["hate hat ate a at"], 2, 10, 4, vocabulary,
                                      space_character=" ", split_on_space=True, remove_space=True, add_dummy_space_start=False)
    print_lattice(encoding, vocabulary)
    print(encoding.size())
    output_potentials = unigramlm(encoding)
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)

    # backward encoding
    encoding = convert_to_backward_encoding(encoding)
    print_lattice(encoding, vocabulary)
    print(encoding.size())
    output_potentials = unigramlm(encoding)
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)

if __name__ == "__main__":
    test()
    test1()