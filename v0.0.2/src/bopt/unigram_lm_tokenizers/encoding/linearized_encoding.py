from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_ID, PADEDGE_ID
from bopt.unigram_lm_tokenizers.utils.encoding import serialize
from bopt.unigram_lm_tokenizers.utils.indexing import start_position_based_indexing
from bopt.utils import increasing_roll_right

import torch

def extract_position_ids(forward_encoding):
    size = forward_encoding.size()
    dtype = forward_encoding.dtype
    device = forward_encoding.device
    N, M, L = size[-3:]
    block_internal_positions = torch.arange(0, L, dtype=dtype, device=device).expand(*size)
    block_positions = L * torch.arange(0, N, dtype=dtype, device=device).expand(*size[:-2])[...,None,None].expand(size)
    position_mapping = increasing_roll_right(block_internal_positions + block_positions, -1)

    blockwise_position_embedding = serialize(position_mapping)[...,start_position_based_indexing(M, L)]
    serialized_position_embedding = blockwise_position_embedding.reshape(*(size[:-3] + (-1,)))
    return serialized_position_embedding

def extract_input_ids(forward_encoding, padding_id=0):
    size = forward_encoding.size()
    N, M, L = size[-3:]
    blockwise_input_ids = serialize(forward_encoding)[...,start_position_based_indexing(M, L)]
    serialized_input_ids = blockwise_input_ids.reshape(*(size[:-3] + (-1,)))
    serialized_input_ids[serialized_input_ids == NONEDGE_ID] = padding_id
    serialized_input_ids[serialized_input_ids == PADEDGE_ID] = padding_id
    return serialized_input_ids

def extract_attention_mask(forward_encoding):
    input_ids = extract_input_ids(forward_encoding, padding_id=-1)
    return (input_ids != -1).to(torch.long)