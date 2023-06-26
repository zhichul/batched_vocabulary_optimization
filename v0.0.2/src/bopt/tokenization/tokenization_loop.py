import code
import json
import math
import sys

from tqdm import tqdm

from bopt.tokenization import TokenizationSetup
from bopt.tokenization.utils import display

from bopt.tokenization.utils import tokens_to_tokenization
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import len_c


def tokenization_loop(setup: TokenizationSetup, batch_mode=True, batch_size=128, device="cuda"):
    if batch_mode:
        references = []
        texts = []
        for line in tqdm(sys.stdin):
            line = line.strip()
            if setup.args.input_mode == "json":
                line = json.loads(line)
                text = line["text"]
                reference = [unit for unit, end in line["tokenizations"][0]]
                references.append(reference)
                texts.append(text)
            else:
                text = line
                texts.append(text)
        start = 0
        log_probs = None
        if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
            setup.tokenizer.to(device)

        with tqdm(total=math.ceil(len(texts) / batch_size)) as pbar:
            while start < len(texts):
                batch_texts = texts[start:start + batch_size]
                batch_references = references[start:start + batch_size]
                if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
                    n = setup.args.n if setup.args.input_tokenizer_mode == "nbest" else 1
                    tokenization_output = setup.tokenizer(batch_texts,
                                                          n=n,
                                                          max_blocks=setup.args.max_blocks,
                                                          max_unit_length=setup.args.max_unit_length,
                                                          max_block_length=setup.args.max_block_length,
                                                          space_character=setup.args.space_character,
                                                          split_on_space=setup.args.split_on_space,
                                                          add_dummy_space_start=setup.args.add_dummy_space_start,
                                                          remove_space=setup.args.remove_space,
                                                          specials=setup.specials,
                                                          try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                                                          pad_token_id=-1,
                                                          temperature=setup.args.temperature,
                                                          output_forward_alpha=setup.args.report_reference)  # pad should never be used in single sentence mode, so this is a fail check
                    if setup.args.report_reference:
                        reference_tokenization_output = setup.tokenizer(batch_texts,
                                                                        n=n,
                                                                        max_blocks=setup.args.max_blocks,
                                                                        max_unit_length=setup.args.max_unit_length,
                                                                        max_block_length=setup.args.max_block_length,
                                                                        space_character=setup.args.space_character,
                                                                        split_on_space=setup.args.split_on_space,
                                                                        add_dummy_space_start=setup.args.add_dummy_space_start,
                                                                        remove_space=setup.args.remove_space,
                                                                        specials=setup.specials,
                                                                        try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                                                                        pad_token_id=-1,
                                                                        temperature=setup.args.temperature,
                                                                        output_forward_alpha=setup.args.report_reference,
                                                                        references=batch_references)
                        log_probs = (reference_tokenization_output.forward_alpha - tokenization_output.forward_alpha).view(-1).tolist()

                    for j in range(len(batch_texts)):
                        tokenizations = []
                        for ids in tokenization_output.input_ids[j].tolist():  # 1 x n x seq_length
                            tokens = [setup.tokenizer.vocabulary[id] for id in ids]
                            # this space removal step below is to make sure that gold
                            # segementations expressed without dummy spaces can be properly
                            # matched against tokenizers with prefixed space
                            tokens = [token.lstrip(setup.args.space_character) for token in tokens if
                                      token != setup.args.space_character and token != setup.args.pad_token]
                            tokenization = tokens_to_tokenization(tokens, specials=setup.specials)
                            tokenizations.append(tokenization)
                        if setup.args.input_tokenizer_mode == "nbest":
                            weights = tokenization_output.weights[j].softmax(-1).view(-1).tolist()
                        else:
                            weights = [1.0]
                        if log_probs is not None:
                            log_prob = log_probs[j]
                        else:
                            log_prob = None
                        display(texts[start + j], tokenizations, weights, log_prob=log_prob, display_mode=setup.args.display_mode)


                if setup.args.input_tokenizer_mode == "bert":
                    tokenizer_output = setup.tokenizer.encode_batch(texts[start:start + batch_size])
                    for j in range(len(batch_texts)):
                        tokens = [setup.vocab[id] for id in tokenizer_output[j].ids[1:-1]]
                        tokens = [token.lstrip(setup.args.space_character) for token in tokens if
                                  token != setup.args.space_character]
                        tokenizations = [tokens_to_tokenization(tokens)]
                        weights = [1.0]
                        display(texts[start + j], tokenizations, weights, display_mode=setup.args.display_mode)


                start += batch_size
                pbar.update(1)
    else:
        for line in tqdm(sys.stdin):
            line = line.strip()
            if setup.args.input_mode == "json":
                line = json.loads(line)
                text = line["text"]
                reference = [unit for unit, end in line["tokenizations"][0]]
            else:
                text = line
            log_prob = None
            if setup.args.input_tokenizer_mode == "nbest" or setup.args.input_tokenizer_mode == "1best":
                n = setup.args.n if setup.args.input_tokenizer_mode == "nbest" else 1

                tokenization_output = setup.tokenizer([text],
                          n=n,
                          max_blocks=setup.args.max_blocks,
                          max_unit_length=setup.args.max_unit_length,
                          max_block_length=setup.args.max_block_length,
                          space_character=setup.args.space_character,
                          split_on_space=setup.args.split_on_space,
                          add_dummy_space_start=setup.args.add_dummy_space_start,
                          remove_space=setup.args.remove_space,
                          specials=setup.specials,
                          try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                          pad_token_id=-1,
                          temperature=setup.args.temperature,
                          output_forward_alpha=setup.args.report_reference) # pad should never be used in single sentence mode, so this is a fail check
                if setup.args.report_reference:
                    reference_tokenization_output = setup.tokenizer([text],
                                                          n=n,
                                                          max_blocks=setup.args.max_blocks,
                                                          max_unit_length=setup.args.max_unit_length,
                                                          max_block_length=setup.args.max_block_length,
                                                          space_character=setup.args.space_character,
                                                          split_on_space=setup.args.split_on_space,
                                                          add_dummy_space_start=setup.args.add_dummy_space_start,
                                                          remove_space=setup.args.remove_space,
                                                          specials=setup.specials,
                                                          try_word_initial_when_unk=setup.args.try_word_initial_when_unk,
                                                          pad_token_id=-1,
                                                          temperature=setup.args.temperature,
                                                          output_forward_alpha=setup.args.report_reference,
                                                          references=[reference])
                    log_prob = (reference_tokenization_output.forward_alpha - tokenization_output.forward_alpha).view(-1).item()
                tokenizations = []
                for ids in tokenization_output.input_ids.squeeze(0).tolist(): # 1 x n x seq_length
                    tokens = [setup.tokenizer.vocabulary[id] for id in ids]
                    # this space removal step below is to make sure that gold
                    # segementations expressed without dummy spaces can be properly
                    # matched against tokenizers with prefixed space
                    tokens = [token.lstrip(setup.args.space_character) for token in tokens if token != setup.args.space_character]
                    tokenization = tokens_to_tokenization(tokens, specials=setup.specials)
                    tokenizations.append(tokenization)
                if setup.args.input_tokenizer_mode == "nbest":
                    weights = tokenization_output.weights.softmax(-1).view(-1).tolist()
                else:
                    weights = [1.0]
            if setup.args.input_tokenizer_mode == "bert":
                tokenizer_output = setup.tokenizer.encode(text)
                tokens = [setup.vocab[id] for id in tokenizer_output.ids[1:-1]]
                tokens = [token.lstrip(setup.args.space_character) for token in tokens if token != setup.args.space_character]
                tokenizations = [tokens_to_tokenization(tokens)]
                weights = [1.0]
            display(text, tokenizations, weights, log_prob=log_prob, display_mode=setup.args.display_mode)
