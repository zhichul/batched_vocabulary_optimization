import glob
import os
from typing import List

from bopt.core.tokenizer import Tokenizer
from bopt.core.tokenizer.tokenization import PackedChunk


def clear_cache(cache_dir: str):
    for f in glob.glob(f'{cache_dir}/*'):
        os.remove(f)


def pretokenize(text_str: str) -> List[str]:
    return ["[BOS]"] + text_str.split(" ") + ["[EOS]"]

def truncated_and_pad_packed_chunks(tokenizer: Tokenizer, packed_chunks: List[PackedChunk], max_blocks: int = None, max_chars: int = None, types=False) -> List[PackedChunk]:
    block_count = 0
    char_count = 0
    kept_chunks = []
    for chunk in packed_chunks:
        if types: chunk_length = sum(tokenizer.len_type(subword_type) for subword_type in chunk)
        else: chunk_length = tokenizer.len_p(chunk)
        if (max_blocks is not None and block_count + 1 > max_blocks) or \
                (max_chars is not None and char_count + chunk_length > max_chars):
            print(f"[WARNING] Truncating {packed_chunks} to {' '.join(sum(kept_chunks, []))}")
            break
        kept_chunks.append(chunk)
        char_count += chunk_length
        block_count += 1

    if max_blocks is not None and len(kept_chunks) < max_blocks:
        kept_chunks += [[]] * (max_blocks - len(kept_chunks)) # add padding if needed
    return kept_chunks

def viterbi_tokenize(tokenizer: Tokenizer, tokens: List[str], remove_csp=False) -> List[List[str]]:
    """
    Returns a list of lists, one for each token, containing its viterbi segmentation, with continuing subword prefix added
    """
    fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask = tokenizer.encode_batch(tokens, tokenizer.max_unit_length)
    fwd_ts = tokenizer.get_weights(fwd_ids)
    max_log_alpha, _, backpointers = tokenizer.viterbi_algorithm(fwd_ts, fwd_ms, lengths)
    log_alpha, _ = tokenizer.forward_algorithm(fwd_ts, fwd_ms, lengths)
    word_ids = tokenizer.decode_backpointers(fwd_ids, lengths, backpointers)
    return [[tokenizer.id2str(id, remove_csp=remove_csp) for id in word_id] for word_id in word_ids]

def pack_viterbi_chunks(kept_chunks, input_tokenizations) -> List[PackedChunk]:
    """

    Args:
        kept_chunks: the truncated and padded chunks without viterbification
        input_tokenizations: the viterbi segmentation of each token (without truncation or padding)

    Returns:
        a list of new chunks each of which contains not tokens but subwords for those tokens
    """
    pos = 0
    viterbi_packed_chunks = []
    for chunk in kept_chunks:
        new_packed_chunk = []
        for token in chunk:
            new_packed_chunk.extend(input_tokenizations[pos])
            pos += 1
        viterbi_packed_chunks.append(new_packed_chunk)
    return viterbi_packed_chunks

def prefix_sum(l):
    cumsum = 0
    out = []
    for item in l:
        cumsum += item
        out.append(cumsum)
    return out