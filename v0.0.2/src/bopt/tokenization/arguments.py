from argparse import ArgumentParser

from bopt.arguments import add_tokenizer_arguments


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument("--seed", required=False, type=int, default=42)
    # vocab & tokenization
    add_tokenizer_arguments(parser, mode="tokenize")
    # display options
    parser.add_argument("--display_mode", default="json", choices=["json", "pretty_json"])
    parser.add_argument("--input_mode", default="txt", choices=["txt", "json"])
    parser.add_argument("--report_reference", action="store_true")
    args = parser.parse_args()
    if args.report_reference and not args.input_mode == "json":
        raise ValueError("report reference requires json input containing reference")
    return args
