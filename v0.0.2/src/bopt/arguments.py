

def add_task_arguments(parser):
    parser.add_argument("--task", required=True, type=str, choices=["classification"])
    parser.add_argument("--domain", required=True, type=str, choices=["morpheme_prediction", "superbizarre_prediction"])

def add_model_arguments(parser, mode="train"):
    # model parameters
    parser.add_argument('--bias_mode', type=str, choices=["albo", "mult_then_renorm"], default="mult_then_renorm")
    parser.add_argument("--use_lattice_position_ids", action="store_true", help="whether to use token based position ids or lattice based (char based)")
    parser.add_argument("--collapse_padding", action="store_true")
    if mode=="train":
        parser.add_argument("--config", type=str)
        parser.add_argument("--pretrained_model", type=str)
        parser.add_argument("--pretrained_ignore", type=str, nargs="+", default=None)
        parser.add_argument("--pretrained_include", type=str, nargs="+", default=None)
    elif mode=="infer":
        parser.add_argument("--model_path", type=str, required=True)


def add_tokenizer_arguments(parser, mode="train"):
    # vocab & tokenization
    parser.add_argument("--input_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
    parser.add_argument("--input_tokenizer_model", required=True, type=str, choices=["unigram", "nulm", "bert"])
    parser.add_argument("--input_tokenizer_mode", required=True, type=str, choices=["lattice", "sample", "nbest", "1best", "bert"])
    parser.add_argument("--input_tokenizer_weights", required=False, type=str)
    parser.add_argument("--nulm_num_hidden_layers", required=False, type=int, default=None)
    parser.add_argument("--nulm_hidden_size", required=False, type=int, default=None)
    parser.add_argument("--log_space_parametrization", action="store_true")
    parser.add_argument("--special_tokens", required=True, nargs="+", default=None)
    parser.add_argument("--try_word_initial_when_unk", action="store_true")
    parser.add_argument("--pad_token", required=True, default=None)
    parser.add_argument("--n", required=False, type=int, default=None, help="n for nbest or sample")
    parser.add_argument("--temperature", required=False, default=1.0, type=float, help="hyperparameter for flattening the distribution over tokenizations")
    if mode=="train" :
        parser.add_argument("--output_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
        parser.add_argument("--nulm_tie_embeddings", action="store_true", help="forces the transformer to use the same embeddings as the nulm")
        parser.add_argument("--subsample_vocab", required=False, default=None, type=float, help="how much to subsample vocabulary at training")
    elif mode=="infer":
        parser.add_argument("--output_vocab", required=True, type=str, help="a tab separated file where first column contain the tokens")
        parser.add_argument("--subsample_vocab", required=False, default=None, type=float, help="how much to subsample vocabulary at training") # this is only to support API should never set
        parser.add_argument("--nulm_tie_embeddings", action="store_true", help="forces the transformer to use the same embeddings as the nulm") # this is only to support API should never set

    # lattice tokenizer parameters
    parser.add_argument("--max_blocks", required=False, type=int, default=None)
    parser.add_argument("--max_unit_length", required=False, type=int, default=None)
    parser.add_argument("--max_block_length", required=False, type=int, default=None)
    parser.add_argument("--space_character", required=False, type=str, default="▁")
    parser.add_argument("--remove_space", action="store_true")
    parser.add_argument("--split_on_space", action="store_true")
    parser.add_argument("--add_dummy_space_start", action="store_true")

def add_training_arguments(parser, mode="normal"):
    # training
    parser.add_argument("--task_model_learning_rate", required=True, type=float, default=6.25e-5)
    parser.add_argument("--task_model_embedding_learning_rate", required=False, type=float, default=None)
    parser.add_argument("--input_tokenizer_learning_rate", required=False, type=float, default=None)
    parser.add_argument("--train_batch_size", required=True, type=int, default=1024)
    parser.add_argument("--train_steps", required=True, type=int, default=10_000)
    parser.add_argument("--patience", required=True, type=int, default=5, help="number of epochs where loss didn't improve to reduce lr")
    parser.add_argument("--lr_adjustment_window_size", required=True, type=int)
    parser.add_argument("--reduce_factor", required=True, type=float, default=0.25, help="factor to reduce lr by when loss plateus")

    # evaluation
    parser.add_argument("--eval_steps", required=True, type=float, default=1000)

    # losses
    parser.add_argument("--annealing", required=False, type=float, default=0.0)
    parser.add_argument("--annealing_start_steps", required=False, type=int, default=0)
    parser.add_argument("--annealing_end_steps", required=False, type=int, default=0)
    parser.add_argument("--L1", required=False, type=float, default=0.0)

    # flags for dynamics logging
    parser.add_argument("--log_learning_dynamics", action="store_true")
    if mode=="bilevel":
        parser.add_argument("--inner_optimizer", choices=["Adam", "SGD"], default="Adam")
        parser.add_argument("--bilevel_optimization_scheme", choices=["unroll", "ift", "reversible-learning"], default="ift")
        parser.add_argument("--train_steps_inner", required=True, type=int, default=30, help="how many inner steps to take per outer/ per unroll")
        parser.add_argument("--train_trajectory_inner", required=False, type=int, default=30, help="how many inner unroll units to take as part of the inner trajectory (only with unroll)")
        parser.add_argument("--train_batch_size_inner", required=False, type=int, default=1024, help="inner batch size")
        parser.add_argument("--train_steps_warmup", required=True, type=int, default=100, help="how many inner steps to take before any outer step")
        parser.add_argument("--inner_threshold", required=False, type=float, default=1e-3, help="this is to exit early if inner converged")
        parser.add_argument("--neumann_iterations", required=False, type=int, default=10, help="number of terms to use in neumann series approximation of vH-1")
        parser.add_argument("--neumann_alpha", required=False, type=float, default=0.01,  help="preconditioner")
        parser.add_argument("--neumann_threshold", required=False, type=float, default=1e-3, help="this is to exit early if neumann estimate of vH-1 is changing slowly (in terms of norm)")
        parser.add_argument("--indirect_gradient_only", action="store_true", help="this is to choose to not use the direct gradient from the outer loss")
        parser.add_argument("--random_restarts", required=False, type=int, default=1, help="number of restarts to use to estimate the outer gradient")
        parser.add_argument("--eval_random_restarts", required=False, type=int, default=1, help="number of restarts to use to estimate the outer gradient")
        parser.add_argument("--fix_transformer_initialization", required=False, action="store_true")
        parser.add_argument("--momentum_coefficient", type=float, help="by default precision is 10e-2")
        parser.add_argument("--momentum_coefficient_precision", type=lambda x:int(float(1/x)), help="precision value, but gets converted to denominator for the code", default=100)
        parser.add_argument("--eval_train_steps", required=True, type=int, default=600)

def add_device_arguments(parser, mode="normal"):
    # gpu
    parser.add_argument("--gpu_batch_size", required=True, type=int, default=128)
    parser.add_argument("--device", required=True, type=str, default="cuda")
    if mode == "bilevel":
        parser.add_argument("--gpu_batch_size_inner", required=True, type=int, default=128)

def add_logging_parameters(parser, mode="train"):
    if mode=="train":
        parser.add_argument("--train_tokenization_cache", required=True)
    elif mode=="bilevel":
        parser.add_argument("--train_inner_tokenization_cache", required=True)
        parser.add_argument("--train_outer_tokenization_cache", required=True)
    if mode == "train" or mode == "bilevel":
        parser.add_argument("--dev_tokenization_cache", required=True)
        parser.add_argument("--test_tokenization_cache", required=True)
        parser.add_argument("--overwrite_cache", action="store_true")
    # saving
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--overwrite_output_directory", action="store_true")