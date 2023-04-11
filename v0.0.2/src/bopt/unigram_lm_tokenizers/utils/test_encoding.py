from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
from bopt.unigram_lm_tokenizers.modeling.unigramlm import UnigramLM
from bopt.unigram_lm_tokenizers.utils.encoding import convert_to_backward_encoding, expand_encodings, \
    convert_to_backward_log_potentials, expand_log_potentials, serialize
from bopt.unigram_lm_tokenizers.utils.indexing import start_position_based_indexing
from bopt.unigram_lm_tokenizers.utils.printing import print_lattice, get_token

import torch


def test_conversion_expansion():
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
    encoding1 = integerize_for_forward(["hate"], 1, 4, 5, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False)
    encoding2 = convert_to_backward_encoding(encoding1)
    for encoding in [expand_encodings(encoding1), expand_encodings(encoding2)]:
        expanded_encoding = encoding.reshape(-1, 1, 4, 5)
        print_lattice(encoding.reshape(-1, 1, 4, 5), vocabulary, log_potentials=unigramlm(expanded_encoding))

    print_lattice(expand_encodings(encoding2).reshape(-1, 1, 4, 5), vocabulary, log_potentials=expand_log_potentials(convert_to_backward_log_potentials(unigramlm(encoding1))).reshape(-1, 1, 4, 5))
    if not (expand_log_potentials(convert_to_backward_log_potentials(unigramlm(encoding1))).reshape(-1, 1, 4, 5) == unigramlm(expand_encodings(encoding2).reshape(-1, 1, 4, 5))).all():
        raise AssertionError

def test_indexing():
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
    encoding1 = integerize_for_forward(["hate"], 1, 4, 5, vocabulary, space_character=" ", split_on_space=False, add_dummy_space_start=False) # 1x1x4x5
    print(serialize(encoding1)[...,start_position_based_indexing(4, 5)].size())
    print([get_token(id, vocabulary) for id in serialize(encoding1)[...,start_position_based_indexing(4, 5)].reshape(-1)])

if __name__ == "__main__":
    test_conversion_expansion()
    test_indexing()