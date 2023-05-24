from argparse import ArgumentParser

from bopt.arguments import add_model_arguments, add_tokenizer_arguments, add_training_arguments, add_device_arguments, add_logging_parameters


def parse_arguments():

    parser = ArgumentParser()

    # data parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", required=True, type=str, choices=["classification"])
    parser.add_argument("--domain", required=True, type=str, choices=["morpheme_prediction", "superbizarre_prediction"])
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--dev_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)
    parser.add_argument("--data_num_workers", required=True, type=int, default=1)

    # # model parameters
    # parser.add_argument('--bias_mode', type=str, choices=["albo", "mult_then_renorm"], default="mult_then_renorm")
    # parser.add_argument("--config", type=str)
    # parser.add_argument("--pretrained_model", type=str)
    # parser.add_argument("--pretrained_ignore", type=str, nargs="+")
    #
    # # vocab & tokenization
    # parser.add_argument("--input_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
    # parser.add_argument("--output_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
    # parser.add_argument("--input_tokenizer_model", required=True, type=str, choices=["unigram", "nulm", "bert"])
    # parser.add_argument("--input_tokenizer_mode", required=True, type=str, choices=["lattice", "sample", "nbest", "1best", "bert"])
    # parser.add_argument("--input_tokenizer_weights", required=False, type=str)
    # parser.add_argument("--nulm_num_hidden_layers", required=False, type=int, default=None)
    # parser.add_argument("--nulm_hidden_size", required=False, type=int, default=None)
    # parser.add_argument("--nulm_tie_embeddings", action="store_true", help="forces the transformer to use the same embeddings as the nulm")
    # parser.add_argument("--log_space_parametrization", action="store_true")
    # parser.add_argument("--special_tokens", required=True, nargs="+", default=None)
    # parser.add_argument("--try_word_initial_when_unk", action="store_true")
    # parser.add_argument("--pad_token", required=True, default=None)
    # parser.add_argument("--n", required=False, type=int, default=None, help="n for nbest or sample")
    # parser.add_argument("--use_lattice_position_ids", action="store_true", help="whether to use token based position ids or lattice based (char based)")
    # parser.add_argument("--subsample_vocab", required=False, default=None, type=float, help="how much to subsample vocabulary at training")
    # parser.add_argument("--temperature", required=False, default=1.0, type=float, help="hyperparameter for flattening the distribution over tokenizations")
    #
    # # lattice tokenizer parameters
    # parser.add_argument("--max_blocks", required=False, type=int, default=None)
    # parser.add_argument("--max_unit_length", required=False, type=int, default=None)
    # parser.add_argument("--max_block_length", required=False, type=int, default=None)
    # parser.add_argument("--space_character", required=False, type=str, default="▁")
    # parser.add_argument("--remove_space", action="store_true")
    # parser.add_argument("--split_on_space", action="store_true")
    # parser.add_argument("--add_dummy_space_start", action="store_true")
    # parser.add_argument("--collapse_padding", action="store_true")
    #
    # # training
    # parser.add_argument("--task_model_learning_rate", required=True, type=float, default=6.25e-5)
    # parser.add_argument("--input_tokenizer_learning_rate", required=False, type=float, default=None)
    # parser.add_argument("--train_batch_size", required=True, type=int, default=32)
    # parser.add_argument("--train_steps", required=True, type=int, default=10_000)
    # parser.add_argument("--patience", required=True, type=int, default=5, help="number of epochs where loss didn't improve to reduce lr")
    # parser.add_argument("--lr_adjustment_window_size", required=True, type=int)
    # parser.add_argument("--reduce_factor", required=True, type=float, default=0.25, help="factor to reduce lr by when loss plateus")
    #
    # # evaluation
    # parser.add_argument("--eval_steps", required=True, type=float, default=1000)
    #
    # # losses
    # parser.add_argument("--annealing", required=False, type=float, default=0.0)
    # parser.add_argument("--annealing_start_steps", required=False, type=int, default=0)
    # parser.add_argument("--annealing_end_steps", required=False, type=int, default=0)
    # parser.add_argument("--L1", required=False, type=float, default=0.0)
    #
    # # gpu
    # parser.add_argument("--gpu_batch_size", required=True, type=int, default=4)
    # parser.add_argument("--device", required=True, type=str, default="cuda")
    #
    # # caching
    # parser.add_argument("--train_tokenization_cache", required=True)
    # parser.add_argument("--dev_tokenization_cache", required=True)
    # parser.add_argument("--test_tokenization_cache", required=True)
    # parser.add_argument("--overwrite_cache", action="store_true")
    #
    # # saving
    # parser.add_argument("--output_directory", required=True)
    # parser.add_argument("--overwrite_output_directory", action="store_true")

    add_model_arguments(parser)
    add_tokenizer_arguments(parser)
    add_training_arguments(parser)
    add_device_arguments(parser)
    add_logging_parameters(parser)
    return parser.parse_args()