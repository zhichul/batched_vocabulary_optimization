import math

import torch

from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.lattice_tokenizer import LatticeTokenizer
from bopt.unigram_lm_tokenizers.utils.printing import print_attention
from experiments.utils.memoizer import OnDiskTensorMemoizer


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
    log_potentials = torch.tensor([math.log(2.0)] * len(vocabulary)).unsqueeze(-1)

    tokenizer = LatticeTokenizer(vocabulary, pretrained_log_potentials=log_potentials)
    memoizer = OnDiskTensorMemoizer("/tmp/bopt/test_lattice_tokenizer", overwrite=True, debug=True)
    sentence_ids = [["1-1", "1-2"],["1-1", "1-2"]]
    output = tokenizer([["hate", "hat"],["hate", "hat"]],
                max_blocks = 2,
                max_unit_length = 4,
                max_block_length = 5,
                space_character = " ",
                split_on_space = True,
                add_dummy_space_start = False,
                remove_space=True,
                memoizer=memoizer,
                sentence_ids=sentence_ids)
    print(output.input_ids)
    print(output.attention_mask)
    print(output.position_ids)
    print(output.type_ids)
    marked_input_ids = output.input_ids.clone()
    marked_input_ids[marked_input_ids == 0] = -1
    print_attention(marked_input_ids, vocabulary, output.attention_bias)
    print(output.entropy)



if __name__ == "__main__":
    test()