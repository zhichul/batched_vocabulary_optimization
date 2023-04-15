from bopt.unigram_lm_tokenizers.label_tokenizers import LatticeLabelTokenizer, NBestLabelTokenizer
from bopt.unigram_lm_tokenizers.lattice_tokenizer import LatticeTokenizer
from bopt.unigram_lm_tokenizers.nbest_tokenizer import NBestTokenizer
from bopt.utils import load_scalar_weights


def load_input_tokenizer(tokenizer_model, tokenizer_mode, vocabulary, log_space_parametrization=False, weight_file=None):
    if tokenizer_model == "unigram":
        if tokenizer_mode == "lattice":
            if weight_file is None: return LatticeTokenizer(vocabulary, log_space_parametrization=log_space_parametrization)
            else: return LatticeTokenizer(vocabulary, pretrained_log_potentials=load_scalar_weights(weight_file), log_space_parametrization=log_space_parametrization)
        elif tokenizer_mode == "nbest" or tokenizer_mode == "1best":
            if weight_file is None: return NBestTokenizer(vocabulary, log_space_parametrization=log_space_parametrization)
            else: return NBestTokenizer(vocabulary, pretrained_log_potentials=load_scalar_weights(weight_file) , log_space_parametrization=log_space_parametrization)

def load_label_tokenizer(tokenizer_mode, vocabulary):
    if tokenizer_mode == "lattice":
        return LatticeLabelTokenizer(vocabulary)
    elif tokenizer_mode == "nbest" or tokenizer_mode == "1best":
        return NBestLabelTokenizer(vocabulary)