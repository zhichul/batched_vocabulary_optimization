from bopt.integerize import Integerizer
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import integerize_for_forward
from bopt.unigram_lm_tokenizers.encoding.linearized_encoding import extract_input_ids, extract_attention_mask, extract_position_ids

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
    encoding1 = integerize_for_forward(["hate hat"], 2, 4, 5, vocabulary, space_character=" ", split_on_space=True,
                                       add_dummy_space_start=False, remove_space=True)  # 1x1x4x5
    position_ids = extract_position_ids(encoding1)
    print(position_ids)
    input_ids = extract_input_ids(encoding1, padding_id=-1)
    print(input_ids)
    attention_mask = extract_attention_mask(encoding1)
    print(attention_mask)

if __name__ == "__main__":
    test()
