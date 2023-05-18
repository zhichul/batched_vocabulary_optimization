import torch
from tokenizers import Tokenizer

from bopt.unigram_lm_tokenizers import UnigramLM, NeuralUnigramLM, LatticeLabelTokenizer, NBestLabelTokenizer, \
    LatticeTokenizer, NBestTokenizer
from bopt.utils import load_scalar_weights


def load_input_tokenizer(tokenizer_model, tokenizer_mode, vocabulary,
                         num_hidden_layers=None, hidden_size=None,  # for NeuralUnigramLM
                         log_space_parametrization=False, # for UnigramLM
                         weight_file=None, # for both
                         tie_embeddings=None,
                         model=None): # for NueralUnigramLM tie embedding
    if tokenizer_model == "unigram":
        if weight_file is not None:
            pretrained_log_potentials = load_scalar_weights(weight_file)
            unigramlm = UnigramLM(len(vocabulary), pretrained_log_potentials=pretrained_log_potentials, log_space_parametrization=log_space_parametrization)
        else:
            unigramlm = UnigramLM(len(vocabulary), log_space_parametrization=log_space_parametrization)
    elif tokenizer_model == "nulm":
        unigramlm = NeuralUnigramLM(len(vocabulary), num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        if weight_file is not None:
            unigramlm.load_state_dict(torch.load(weight_file))
        if tie_embeddings:
            # check size matches
            if hidden_size != model.config.hidden_size: raise ValueError(
                f"Embeddings size does not match nulm:{hidden_size} transformer:{model.config.hidden_size}, cannot tie embeddings")
            model.bert.embeddings.word_embeddings = unigramlm.edge_embeddings
    elif tokenizer_model == "bert":
        pass
    else:
        raise ValueError(f"Unknown tokenizer model: {tokenizer_model}")
    if tokenizer_mode == "lattice":
        return LatticeTokenizer(unigramlm, vocabulary)
    elif tokenizer_mode == "nbest" or tokenizer_mode == "1best":
        return NBestTokenizer(unigramlm, vocabulary)
    elif tokenizer_mode == "bert":
        return Tokenizer.from_file(weight_file)



def load_label_tokenizer(tokenizer_mode, vocabulary):
    if tokenizer_mode == "lattice":
        return LatticeLabelTokenizer(vocabulary)
    elif tokenizer_mode == "nbest" or tokenizer_mode == "1best":
        return NBestLabelTokenizer(vocabulary)
    elif tokenizer_mode == "bert":
        return NBestLabelTokenizer(vocabulary)