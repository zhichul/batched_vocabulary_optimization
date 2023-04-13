from bopt.unigram_lm_tokenizers.label_tokenizers import LatticeLabelTokenizer
from bopt.unigram_lm_tokenizers.lattice_tokenizer import LatticeTokenizer
from bopt.utils import load_scalar_weights


def load_input_tokenizer(tokenizer_model, tokenizer_mode, vocabulary, weight_file=None):
    if tokenizer_model == "unigram":
        if tokenizer_mode == "lattice":
            if weight_file is None: return LatticeTokenizer(vocabulary)
            else: return LatticeTokenizer(vocabulary, load_scalar_weights(weight_file))

def load_label_tokenizer(tokenizer_mode, vocabulary):
    if tokenizer_mode == "lattice":
        return LatticeLabelTokenizer(vocabulary)