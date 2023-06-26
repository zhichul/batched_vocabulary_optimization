import json

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import len_c


def display(text, tokenizations, weights, log_prob=None, display_mode="json"):
    output = {}
    output["text"] = text
    output["tokenizations"] = tokenizations
    output["weights"] = weights
    if log_prob is not None:
        output["reference_log_prob"] = log_prob
    if display_mode == "json":
        print(json.dumps(output))
    elif display_mode == "pretty_json":
        print(json.dumps(output, indent=4))

def tokens_to_tokenization(tokens, specials=set()):
    tokenization = []
    rboundary = 0
    for token in tokens:
        rboundary += len_c(token, specials)
        tokenization.append((token, rboundary))
    return tokenization