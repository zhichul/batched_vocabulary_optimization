from argparse import ArgumentParser

from bopt.arguments import add_tokenizer_arguments


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument("--seed", required=False, type=int, default=42)
    # vocab & tokenization
    add_tokenizer_arguments(parser, mode="tokenize")
    # display options
    parser.add_argument("--display_mode", default="json", choices=["json", "pretty_json"])
    return parser.parse_args()
