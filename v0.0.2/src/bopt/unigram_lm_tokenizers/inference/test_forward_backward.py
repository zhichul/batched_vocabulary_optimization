from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.backward_encoding import integerize_for_backward
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
from bopt.unigram_lm_tokenizers.inference.forward_backward import forward_algorithm
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.encoding.utils import print_lattice

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
    encoding1 = integerize_for_forward(["hate"], 1, 4, 4, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False)
    encoding2 = integerize_for_backward(["hate"], 1, 4, 4, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False)
    for encoding in [encoding1, encoding2]:
        print_lattice(encoding, vocabulary)
        unigramlm = UnigramLM(len(vocabulary), log_potentials)
        print("id_matrix_size", encoding.size())
        output_potentials = unigramlm(encoding)
        print("Edge (log) potentials")
        print_lattice(encoding, vocabulary, log_potentials=output_potentials)
        forward_output = forward_algorithm(output_potentials)
        print("Edge (log) alphas")
        print_lattice(encoding, vocabulary, log_potentials=forward_output.edge_log_alphas)
        print(forward_output.last_node_log_alphas)

if __name__ == "__main__":
    test()