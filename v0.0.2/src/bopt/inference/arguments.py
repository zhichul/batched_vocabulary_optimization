from argparse import ArgumentParser

from bopt.arguments import add_model_arguments, add_tokenizer_arguments, add_training_arguments, add_device_arguments, add_logging_parameters


def parse_arguments():

    parser = ArgumentParser()

    # data parameters
    parser.add_argument("--task", required=True, type=str, choices=["classification"])
    parser.add_argument("--domain", required=True, type=str, choices=["morpheme_prediction", "superbizarre_prediction"])
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--data_num_workers", required=True, type=int, default=1)

    # model parameters
    add_model_arguments(parser, mode="infer")

    # vocab & tokenization
    add_tokenizer_arguments(parser, mode="infer")

    # gpu
    add_device_arguments(parser)

    # saving
    add_logging_parameters(parser, mode="infer")
    return parser.parse_args()