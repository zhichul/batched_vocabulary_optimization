import code
import json
import sys

from tqdm import tqdm

from bopt.tokenization import TokenizationSetup
from bopt.tokenization.utils import display

from bopt.tokenization.utils import tokens_to_tokenization
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import len_c


def tokenization_loop(setup: TokenizationSetup):
    for line in tqdm(sys.stdin):
        text = line.strip()
        if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
            n = setup.args.n if setup.args.input_tokenizer_mode == "nbest" else 1

            tokenization_output = setup.tokenizer([line],
                      n=n,
                      max_blocks=setup.args.max_blocks,
                      max_unit_length=setup.args.max_unit_length,
                      max_block_length=setup.args.max_block_length,
                      space_character=setup.args.space_character,
                      split_on_space=setup.args.split_on_space,
                      add_dummy_space_start=setup.args.add_dummy_space_start,
                      remove_space=setup.args.remove_space,
                      specials=setup.specials,
                      pad_token_id=-1) # pad should never be used in single sentence mode, so this is a fail check
            tokenizations = []
            for ids in tokenization_output.input_ids.squeeze(0).tolist(): # 1 x n x seq_length
                tokens = [setup.tokenizer.vocabulary[id] for id in ids]
                # this space removal step below is to make sure that gold
                # segementations expressed without dummy spaces can be properly
                # matched against tokenizers with prefixed space
                tokens = [token.lstrip(setup.args.space_character) for token in tokens if token != setup.args.space_character]
                tokenization = tokens_to_tokenization(tokens, specials=setup.specials)
                tokenizations.append(tokenization)
            if setup.args.input_tokenizer_mode == "nbest":
                weights = tokenization_output.weights.softmax(-1).view(-1).tolist()
            else:
                weights = [1.0]
        display(text, tokenizations, weights, display_mode=setup.args.display_mode)
