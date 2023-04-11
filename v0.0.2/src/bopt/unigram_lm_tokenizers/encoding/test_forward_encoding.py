from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
from bopt.unigram_lm_tokenizers.utils.printing import print_lattice

import traceback
import torch


def test1():
    vocabulary = Integerizer(
        [
            "[UNK]",
            "▁",
            "▁h",
            "▁hat",
            "▁hate",
            "▁hate",
            "▁a",
            "▁ate",
            "▁at",
            "t",
            "a",
            "e",
            "h"
        ]
    )
    sentences = ["hate", "ate hat", "ate hate hat"]
    max_unit_length = 3
    max_block_length = 4
    max_blocks = 3
    try:
        integerize_for_forward(sentences, max_blocks, max_unit_length, max_block_length, vocabulary)
    except ValueError as e:
        traceback.print_exc()

def test2():
    vocabulary = Integerizer(
        [
            "[UNK]",
            "▁",
            "▁h",
            "▁hat",
            "▁hate",
            "▁hate",
            "▁a",
            "▁ate",
            "▁at",
            "t",
            "a",
            "e",
            "h",
            "at",
            "ate",
            "hat",
            "[PAD]",
            "[NON]"
        ]
    )
    sentences = ["hate", "ate hat", "ate hote hat"]
    max_unit_length = 3
    max_block_length = 5
    max_blocks = 3
    output = integerize_for_forward(sentences, max_blocks, max_unit_length, max_block_length, vocabulary)
    assert (output == torch.tensor([[[[ 1, 11,  9,  8, 10],
      [-1,  2, -1, 12, -1],
      [-1, -1, -1, 14, 13]],

     [[-2, -2, -2, -2, -2],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1]],

     [[-2, -2, -2, -2, -2],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1]]],


    [[[ 1,  9,  8, 10, -2],
      [-1,  5, 12, -1, -1],
      [-1, -1,  7, 13, -1]],

     [[ 1, 11,  9,  8, -2],
      [-1,  2, -1, 12, -1],
      [-1, -1, -1, 14, -1]],

     [[-2, -2, -2, -2, -2],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1]]],


    [[[ 1,  9,  8, 10, -2],
      [-1,  5, 12, -1, -1],
      [-1, -1,  7, 13, -1]],

     [[ 1, 11,  0,  8, 10],
      [-1,  2, -1, -1, -1],
      [-1, -1, -1, -1, -1]],

     [[ 1, 11,  9,  8, -2],
      [-1,  2, -1, 12, -1],
      [-1, -1, -1, 14, -1]]]], dtype=torch.long)).all().item()
    print_lattice(output, vocabulary)

if __name__ == "__main__":
    test1()
    test2()