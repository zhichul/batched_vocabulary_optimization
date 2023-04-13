from argparse import ArgumentParser

def parse_arguments():

    parser = ArgumentParser()

    # data parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", required=True, type=str, choices=["classification"])
    parser.add_argument("--domain", required=True, type=str, choices=["morpheme_prediction"])
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--dev_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)
    parser.add_argument("--data_num_workers", required=True, type=int, default=1)

    # model parameters
    parser.add_argument('--bias_mode', type=str, choices=["albo", "mult_then_renorm"], default="mult_then_renorm")
    parser.add_argument("--config", type=str)

    # vocab & tokenization
    parser.add_argument("--input_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
    parser.add_argument("--output_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
    parser.add_argument("--input_tokenizer_model", required=True, type=str, choices=["unigram", "nulm"])
    parser.add_argument("--input_tokenizer_mode", required=True, type=str, choices=["1best", "lattice", "1sample", "nbest"])
    parser.add_argument("--input_tokenizer_weights", required=False, type=str)
    parser.add_argument("--special_tokens", required=True, nargs="+", default=["[PAD]", "[UNK]", "[SP1]", "[SP2]", "[SP3]"])
    parser.add_argument("--pad_token", required=True, default="[PAD]")

    # lattice tokenizer parameters
    parser.add_argument("--max_blocks", required=True, type=int)
    parser.add_argument("--max_unit_length", required=True, type=int)
    parser.add_argument("--max_block_length", required=True, type=int)
    parser.add_argument("--space_character", required=True, type=str, default="▁")
    parser.add_argument("--remove_space", action="store_true")
    parser.add_argument("--split_on_space", action="store_true")
    parser.add_argument("--add_dummy_space_start", action="store_true")

    # training
    parser.add_argument("--task_model_learning_rate", required=True, type=float, default=6.25e-5)
    parser.add_argument("--input_tokenizer_learning_rate", required=True, type=float, default=0.02)
    parser.add_argument("--train_batch_size", required=True, type=int, default=32)
    parser.add_argument("--train_steps", required=True, type=int, default=10_000)
    parser.add_argument("--patience", required=True, type=int, default=5, help="number of epochs where loss didn't improve to reduce lr")
    parser.add_argument("--lr_adjustment_window_size", required=True, type=int)
    parser.add_argument("--reduce_factor", required=True, type=float, default=0.25, help="factor to reduce lr by when loss plateus")

    # evaluation
    parser.add_argument("--eval_steps", required=True, type=float, default=1000)

    # losses
    parser.add_argument("--annealing", required=True, type=float, default=0.0)
    parser.add_argument("--annealing_start_steps", required=True, type=int, default=0.0)
    parser.add_argument("--annealing_end_steps", required=True, type=float, default=0.0)
    parser.add_argument("--L1", required=True, type=float, default=0.0)

    # gpu
    parser.add_argument("--gpu_batch_size", required=True, type=int, default=4)
    parser.add_argument("--device", required=True, type=str, default="cuda")

    # caching
    parser.add_argument("--train_tokenization_cache", required=True)
    parser.add_argument("--dev_tokenization_cache", required=True)
    parser.add_argument("--test_tokenization_cache", required=True)
    parser.add_argument("--overwrite_cache", action="store_true")

    # saving
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--overwrite_output_directory", action="store_true")
    return parser.parse_args()