from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None,  help='pretrained model name')
    parser.add_argument('--config', type=str)
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.", required=True)
    parser.add_argument('--task', type=str, choices=["morpheme_prediction", "language_modeling"], default="morpheme_prediction", help='name of the task', required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')

    parser.add_argument('--input_vocab', type=str, default=None, required=True)
    parser.add_argument('--continuing_subword_prefix', type=str)
    parser.add_argument('--output_vocab', type=str, default=None, help="If not the same as input vocab.")
    parser.add_argument('--weights_file', type=str, default=None, help="If not using default initialization.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_tokenize", action='store_true', help="Whether to run tokenization.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--warmup_epochs', type=float, default=2)
    parser.add_argument('--train_epochs', type=int)
    parser.add_argument('--eval_epochs', type=int)
    parser.add_argument('--save_epochs', type=int)
    parser.add_argument('--train_dataset', type=str, default=None, required=False)
    parser.add_argument('--eval_dataset', type=str, default=None, required=False)
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--gpu_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--data_num_workers', type=int, default=1)

    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--weights_learning_rate', type=float, default=0.02)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')

    parser.add_argument('--main_loss_multiplier', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--entropic', type=float, default=0.0)
    parser.add_argument('--entropy_start', type=int, default=200)
    parser.add_argument('--entropy_end', type=int, default=200)

    parser.add_argument('--bias_mode', type=str, choices=["albo"], default="albo")
    parser.add_argument('--vopt', action='store_true')
    parser.add_argument('--debug_viterbi_lattice', action='store_true')
    parser.add_argument('--debug_node_unigram', action='store_true')
    parser.add_argument('--debug_fixed_point', action='store_true')
    parser.add_argument('--normalize_by_tokens', action='store_true')
    parser.add_argument('--normalize_by_expected_length', action='store_true')
    parser.add_argument('--no_normalization', action='store_true')
    parser.add_argument('--constant_normalization', type=float, default=None)
    parser.add_argument('--output_viterbi', action='store_true')
    parser.add_argument('--length_normalized_initialization', action='store_true')
    parser.add_argument('--log_space', action='store_true')
    parser.add_argument('--marginal_temperature', type=float, default=1.0)
    parser.add_argument('--quiet', action='store_true')

    parser.add_argument('--max_blocks', type=int)
    parser.add_argument('--max_block_length', type=int)
    parser.add_argument('--max_unit_length', type=int, default=1e10)
    parser.add_argument('--max_length', type=int, help="used for non-lattice")
    parser.add_argument('--specials', type=str, nargs="+", default=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[WBD]", "[SP1]", "[SP2]", "[SP3]", "[SP4]", "[SP5]", "[BOS]", "[EOS]"])
    parser.add_argument('--pad_token', type=str, default="[PAD]")

    parser.add_argument('--log_lattice', type=str, nargs="+")
    parser.add_argument('--log_lattice_key', type=str)
    parser.add_argument('--log_lattice_file', type=str)
    return check_args(parser.parse_args())

def check_args(args):
    # check args
    if not args.do_train and not args.do_eval and not args.do_tokenize:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_tokenize` must be True.")

    # check existence and overwriting of output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if len(os.listdir(args.output_dir)) != 0 and not args.overwrite_output_dir:
            raise ValueError("Output dir exists and is non-empty, please set overwrite_output_dir to True")
    return args