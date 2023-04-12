from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()
    # data parameters
    parser.add_argument("--train_dataset", required=True, type=str)
    parser.add_argument("--dev_dataset", required=True, type=str)
    parser.add_argument("--test_dataset", required=True, type=str)

    # vocab & tokenization
    parser.add_argument("--input_vocab", required=True, type=str)
    parser.add_argument("--output_vocab", required=True, type=str)
    parser.add_argument("--input_tokenizer_model", required=True, type=str, choices=["unigram", "nulm"])
    parser.add_argument("--input_tokenizer_mode", required=True, type=str, choices=["1best", "lattice", "1sample", "nbest"])
    parser.add_argument("--input_tokenizer_weights", required=False, type=str)

    # training
    parser.add_argument("--task_model_learning_rate", required=True, type=float, default=6.25e-5)
    parser.add_argument("--input_tokneizer_learning_rate", required=True, type=float, default=0.02)
    parser.add_argument("--train_batch_size", required=True, type=int, default=32)
    parser.add_argument("--train_steps", required=True, type=int, default=10_000)

    # evaluation
    parser.add_argument("--eval_steps", required=True, type=float, default=1000)
    parser.add_argument("--eval_steps", required=True, type=float, default=1000)

    # losses
    parser.add_argument("--annealing", required=True, type=float, default=0.0)
    parser.add_argument("--annealing_start_steps", required=True, type=int, default=0.0)
    parser.add_argument("--annealing_end_steps", required=True, type=float, default=0.0)
    parser.add_argument("--L1", required=True, type=float, default=0.0)

    # gpu
    parser.add_argument("--gpu_batch_size", required=True, type=int, default=4)

    return parser.parse_args()