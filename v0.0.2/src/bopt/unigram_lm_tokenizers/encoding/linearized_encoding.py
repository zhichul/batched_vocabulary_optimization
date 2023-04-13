import code

from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID
from bopt.unigram_lm_tokenizers.utils.indexing import start_position_based_indexing, linearize, \
    serialize_by_start_position
from bopt.utils import increasing_roll_right, product

import torch

"""
All of these methodds maintains arbitrary size prefixes ... x N x M x L => ... x NE
"""
def extract_position_ids(forward_encoding):
    size = forward_encoding.size()
    dtype = forward_encoding.dtype
    device = forward_encoding.device
    N, M, L = size[-3:]
    block_internal_positions = torch.arange(0, L, dtype=dtype, device=device).expand(*size)
    block_positions = L * torch.arange(0, N, dtype=dtype, device=device).expand(*size[:-2])[...,None,None].expand(size)
    position_mapping = increasing_roll_right(block_internal_positions + block_positions, -1)

    blockwise_position_embedding = serialize_by_start_position(position_mapping)
    serialized_position_embedding = blockwise_position_embedding.reshape(*(size[:-3] + (-1,)))
    return serialized_position_embedding

def extract_input_ids(forward_encoding, padding_id=0):
    size = forward_encoding.size()
    N, M, L = size[-3:]
    blockwise_input_ids = serialize_by_start_position(forward_encoding)
    serialized_input_ids = blockwise_input_ids.reshape(*(size[:-3] + (-1,)))
    serialized_input_ids[serialized_input_ids == NONEDGE_ID] = padding_id
    serialized_input_ids[serialized_input_ids == PADEDGE_ID] = padding_id
    return serialized_input_ids

def extract_attention_mask(forward_encoding):
    input_ids = extract_input_ids(forward_encoding, padding_id=-1)
    return (input_ids != -1).to(torch.long)

def extract_type_ids(forward_encoding):
    K,N,M,L = forward_encoding.size()[-4:]
    device = forward_encoding.device
    E = M * L - (M - 1) * M // 2
    NE = N * E
    B = product(forward_encoding.size()[:-4])
    type_ids = torch.arange(K, dtype=torch.long, device=device)[None, :, None].expand(B, K, 1).reshape(B * K, -1).expand(B * K, NE).reshape(B, K * NE)  # B x KNE
    return type_ids.reshape(*(forward_encoding.size()[:-4] + (-1,)))

def extract_token_encoding(forward_encoding, use_lattice_position_ids=False, padding_id=0):
    size = forward_encoding.size()
    device = forward_encoding.device
    K, N, M, L = size[-4:]
    size_prefix = size[:-4]
    B = product(size_prefix)
    forward_encoding = forward_encoding.reshape(-1, K, N, M, L)
    token_mask = (forward_encoding != NONEDGE_ID) & (forward_encoding != PADEDGE_ID)  # ... x K x N x M x L
    serialized_token_mask = serialize_by_start_position(token_mask).reshape(B, -1) # ... x KNE

    input_ids = extract_input_ids(forward_encoding).reshape(B, -1)
    attention_mask = extract_attention_mask(forward_encoding).reshape(B, -1)
    type_ids = extract_type_ids(forward_encoding).reshape(B, -1)
    if use_lattice_position_ids:
        position_ids = extract_position_ids(forward_encoding.reshape(B, K*N, M, L)).reshape(B, -1)

    ntokens = serialized_token_mask.sum(-1)  # B (sums over the KN blocks)
    max_tokens = ntokens.max().item()  # 1
    sparse_input_ids = input_ids[serialized_token_mask]  # whatever
    sparse_attention_mask = attention_mask[serialized_token_mask]  # whatever
    sparse_type_ids = type_ids[serialized_token_mask]  # whatever
    if use_lattice_position_ids:
        sparse_position_ids = position_ids[serialized_token_mask]  # whatever

    input_ids = torch.zeros((B, max_tokens), dtype=torch.long, device=device).fill_(padding_id)  # ..., max_tokens
    attention_mask = torch.zeros((B, max_tokens), dtype=torch.long, device=device)  # ..., max_tokens
    type_ids = torch.zeros((B, max_tokens), dtype=torch.long, device=device).fill_(padding_id)  # ..., max_tokens
    if use_lattice_position_ids:
        position_ids = torch.zeros((B, max_tokens), dtype=torch.long, device=device).fill_(padding_id)  # ..., max_tokens
    else:
        # use increasing position id
        position_ids = torch.arange(max_tokens, dtype=torch.long, device=device).expand(B, max_tokens)  # ..., max_tokens
    start = 0
    for b in range(B):
        ntoken = ntokens[b].item()
        input_ids[b, :ntoken] = sparse_input_ids[start:start + ntoken]
        attention_mask[b, :ntoken] = sparse_attention_mask[start:start + ntoken]
        type_ids[b, :ntoken] = sparse_type_ids[start:start + ntoken]
        if use_lattice_position_ids:
            position_ids[b, :ntoken] = sparse_position_ids[start:start + ntoken]
        start = start + ntoken
    assert start == sparse_input_ids.numel()
    input_ids = input_ids.reshape(*(size_prefix + (-1,)))
    attention_mask = attention_mask.reshape(*(size_prefix + (-1,)))
    position_ids = position_ids.reshape(*(size_prefix + (-1,)))
    type_ids = type_ids.reshape(*(size_prefix + (-1,)))
    return input_ids, attention_mask, position_ids, type_ids