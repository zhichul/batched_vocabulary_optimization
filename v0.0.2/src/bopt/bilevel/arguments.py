from argparse import ArgumentParser

from bopt.arguments import add_model_arguments, add_tokenizer_arguments, add_training_arguments, add_device_arguments, \
    add_logging_parameters, add_task_arguments


def parse_arguments():

    parser = ArgumentParser()
    # data parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_inner_dataset", required=True, type=str)
    parser.add_argument("--train_outer_dataset", required=True, type=str)
    parser.add_argument("--dev_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)
    parser.add_argument("--data_num_workers", required=True, type=int, default=1)

    add_task_arguments(parser)
    add_model_arguments(parser)
    add_tokenizer_arguments(parser)
    add_training_arguments(parser, mode="bilevel")
    add_device_arguments(parser, mode="bilevel")
    add_logging_parameters(parser, mode="bilevel")
    args = parser.parse_args()
    return args