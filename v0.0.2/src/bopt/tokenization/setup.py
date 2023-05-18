from bopt.tokenization import TokenizationSetup
from bopt.unigram_lm_tokenizers.loading import load_input_tokenizer
from bopt.utils import load_vocab
from experiments.utils.seeding import seed


def setup(args):
    seed(args.seed)

    # load vocabularies
    input_vocab = load_vocab(args.input_vocab)
    input_vocab.specials = set(args.special_tokens)

    # tokenizer
    input_tokenizer = load_input_tokenizer(args.input_tokenizer_model, args.input_tokenizer_mode, input_vocab, log_space_parametrization=args.log_space_parametrization, weight_file=args.input_tokenizer_weights,
                                           num_hidden_layers=args.nulm_num_hidden_layers, hidden_size=args.nulm_hidden_size)
    specials = set(args.special_tokens)

    return TokenizationSetup(args=args,
                            tokenizer=input_tokenizer,
                            specials=specials)