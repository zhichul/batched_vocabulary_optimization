import torch

from bopt.training.setup import ClassificationSetup
from bopt.unigram_lm_tokenizers.tokenizers import UnigramLMTokenizerOutput
from experiments.utils.functions import ramp_function


def lattice_classification(step, epoch, batch, setup:ClassificationSetup, mode="train"):

    entropy_coeff = ramp_function(0, setup.args.annealing, setup.args.annealing_start_steps, setup.args.annealing_end_steps)(step)
    ids, texts, labels = batch
    tokenizer_output: UnigramLMTokenizerOutput = setup.input_tokenizer(texts,
                setup.args.max_blocks,
                setup.args.max_unit_length,
                setup.args.max_block_length,
                setup.args.space_character,
                setup.args.split_on_space,
                setup.args.add_dummy_space_start,
                setup.args.remove_space,
                setup.train_tokenization_memoizer
                          if mode == "train" else
                          (setup.dev_tokenization_memoizer
                           if mode == "dev" else setup.test_tokenization_memoizer),
                ids,
                specials=setup.specials)
    labels = setup.label_tokenizer(labels,
                                   setup.args.max_unit_length,
                                   tokenizer_output.input_ids.size(-1),
                                   setup.train_label_memoizer
                                   if mode == "train" else
                                   (setup.dev_label_memoizer
                                    if mode == "dev" else setup.test_label_memoizer),
                                   ids)
    losses = setup.model(input_ids=tokenizer_output.input_ids, position_ids=tokenizer_output.position_ids, labels=labels, attn_bias=tokenizer_output.attention_bias)
    loss = losses[0] + entropy_coeff * tokenizer_output.entropy + setup.args.L1 * setup.input_tokenizer.unigramlm.edge_log_potentials.weight.exp().mean()
    predictions = losses[1].argmax(dim=-1)
    return predictions, loss