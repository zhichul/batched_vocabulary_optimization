from typing import List
from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward, PADEDGE_ID, NONEDGE_ID
from bopt.utils import increasing_roll_right

import torch



def integerize_for_backward(sentences: List[str],
                           max_blocks: int,
                           max_block_length: int,
                           max_unit_length: int,
                           vocabulary: Integerizer,
                           space_character : str = "▁",
                           split_on_space : bool = True,
                           add_dummy_space_start : bool = True) -> torch.Tensor:
    forward_ids = integerize_for_forward(sentences,
                                         max_blocks,
                                         max_block_length,
                                         max_unit_length,
                                         vocabulary,
                                         space_character=space_character,
                                         split_on_space=split_on_space,
                                         add_dummy_space_start=add_dummy_space_start)
    return increasing_roll_right(forward_ids.flip(-1), NONEDGE_ID)