import math

import torch

from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, NONEDGE_ID
from bopt.unigram_lm_tokenizers.inference.viterbi import viterbi_nbest
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.utils.printing import print_lattice


def test():
    print("Test Viterbi")
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
    encoding = integerize_for_forward(["hate"], 1, 4, 5, vocabulary, space_character=" ", split_on_space=True,
                                      add_dummy_space_start=False, remove_space=True)
    print_lattice(encoding, vocabulary)
    unigramlm = UnigramLM(len(vocabulary), log_potentials)
    output_potentials = unigramlm(encoding)
    print("Edge (log) potentials")
    print_lattice(encoding, vocabulary, log_potentials=output_potentials)
    vout = viterbi_nbest(output_potentials, n=6)
    print(vout.mask)
    nbest_encoding = encoding.expand_as(vout.mask).clone()
    nbest_encoding[~vout.mask] = NONEDGE_ID
    print_lattice(nbest_encoding.reshape(-1, 1, 4, 5), vocabulary)
    print(vout.weight)
    print(vout.mask[0,0,-1])

if __name__ == "__main__":
    test()