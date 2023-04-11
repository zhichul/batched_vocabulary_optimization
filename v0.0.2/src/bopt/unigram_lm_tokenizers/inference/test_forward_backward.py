import math

from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, NONEDGE_ID
from bopt.unigram_lm_tokenizers.inference.forward_backward import forward_algorithm, conditional_marginals
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.utils.encoding import convert_to_backward_encoding, convert_to_forward_encoding, \
    expand_encodings
from bopt.unigram_lm_tokenizers.utils.indexing import serialize_by_start_position, edge_to_prev_node, edge_to_next_node
from bopt.unigram_lm_tokenizers.utils.printing import print_lattice

import torch

def test_forward():
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
    encoding2 = convert_to_backward_encoding(encoding1)
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


def test_forward_backward():
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
    cmo = conditional_marginals(output_potentials)
    bcm = cmo.backward_conditional_marginals
    fcm = cmo.forward_conditional_marginals
    print("Edge backward (log) conditional marginals")
    print(bcm.size())
    lattice = convert_to_forward_encoding(expand_encodings(convert_to_backward_encoding(encoding)))
    lattice = torch.cat([encoding.new_zeros(encoding.size()).fill_(NONEDGE_ID)[...,None,:,:], lattice], dim=-3)
    print_lattice(lattice.reshape(-1 ,1,4,5), vocabulary, log_potentials=bcm.reshape(-1, 1, 4, 5), exponentiate=True)
    print("Edge forward (log) conditional marginals")
    print(fcm.size())
    lattice = expand_encodings(encoding, longest_first=True)
    lattice = torch.cat([encoding[...,None,:,:], lattice], dim=-3)
    print_lattice(lattice.reshape(-1,1,4,5), vocabulary, log_potentials=fcm.reshape(-1, 1, 4, 5), exponentiate=True)

    bcms = serialize_by_start_position(bcm)
    fcms = serialize_by_start_position(fcm)
    e2prev = edge_to_prev_node(4, 5)
    e2next = edge_to_next_node(4, 5)
    backward_attention = bcms[..., e2prev, :]
    forward_attention = fcms[..., e2next, :]
    assert (torch.tril(forward_attention.exp()) == 0).all()
    assert (torch.triu(backward_attention.exp()) == 0).all()

if __name__ == "__main__":
    test_forward()
    test_forward_backward()