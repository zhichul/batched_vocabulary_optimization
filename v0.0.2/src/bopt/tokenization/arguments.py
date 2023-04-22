from argparse import ArgumentParser


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument("--seed", required=False, type=int, default=42)
    # vocab & tokenization
    parser.add_argument("--input_vocab", required=True, type=str,
                        help="a tab separated file where first column contain the tokens")
    parser.add_argument("--input_tokenizer_model", required=True, type=str, choices=["unigram", "nulm"])
    parser.add_argument("--input_tokenizer_mode", required=True, type=str, choices=["lattice", "sample", "nbest", "1best"])
    parser.add_argument("--input_tokenizer_weights", required=False, type=str)
    parser.add_argument("--log_space_parametrization", action="store_true")
    parser.add_argument("--special_tokens", required=True, nargs="+", default=["[PAD]", "[UNK]", "[SP1]", "[SP2]", "[SP3]"])
    parser.add_argument("--pad_token", required=True, default="[PAD]")
    parser.add_argument("--n", required=False, type=int, default=5, help="n for nbest or sample")
    parser.add_argument("--temperature", required=False, type=float, default=1.0, help="hyperparameter to flatten the distribution over tokenizations")

    # lattice tokenizer parameters
    parser.add_argument("--max_blocks", required=True, type=int)
    parser.add_argument("--max_unit_length", required=True, type=int)
    parser.add_argument("--max_block_length", required=True, type=int)
    parser.add_argument("--space_character", required=True, type=str, default="▁")
    parser.add_argument("--remove_space", action="store_true")
    parser.add_argument("--split_on_space", action="store_true")
    parser.add_argument("--add_dummy_space_start", action="store_true")

    # display options
    parser.add_argument("--display_mode", default="json", choices=["json", "pretty_json"])
    return parser.parse_args()
