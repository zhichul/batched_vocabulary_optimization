import code
import math
from dataclasses import dataclass
from typing import Optional, Any

import torch

@dataclass
class ViterbiOutput:
    mask:Optional[Any] = None
    weight:Optional[Any] = None

def viterbi_nbest(edge_log_potentials: torch.FloatTensor, n=1):
    size = edge_log_potentials.size()
    device = edge_log_potentials.device
    N, M, L = size[-3:]
    size_prefix = size[:-2]
    edge_log_alphas = edge_log_potentials.reshape(-1, 1, M, L).clone() # this is the edge log alphas without the node contribution

    # forward algorithm (converts internally to B' x n x M x L where B' collapses all the dimesions in size_prefix into a single one)
    B = edge_log_potentials.numel() // (M * L)
    Bh = edge_log_potentials.numel() // (M * L * N)
    back_edge_indices = []
    node_nbest_log_alphas = [torch.zeros(B, n, device=device, dtype=torch.float)]
    node_nbest_log_alphas[0][:,1:] = -math.inf
    for i in range(L):
        # this is to select the outgoing edges of the ith node
        maski = (torch.diag_embed(torch.ones(L - i, device=device, dtype=torch.bool), offset=i)[:M][None,None,...]).expand(B, n, M, L)


        # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
        node_to_edge = maski.new_zeros(maski.size(), dtype=torch.float)
        node_to_edge[maski] = node_nbest_log_alphas[i][..., None, None].expand_as(node_to_edge)[maski]
        edge_log_alphas = edge_log_alphas + node_to_edge

        # this update corresponds to aggregating the `incoming` edge-alphas into the alpha of a node
        new_log_alphas, new_back_edge_index = edge_log_alphas[..., i].reshape(B, n * M).topk(k=n, dim=-1) # B x n
        print(new_log_alphas, new_back_edge_index, edge_log_alphas[..., i].reshape(B, n * M))
        back_edge_indices.append(new_back_edge_index)
        node_nbest_log_alphas.append(new_log_alphas)
    d1_indices = torch.arange(B, dtype=torch.long, device=device)[:,None,None].expand(B, n, n)
    d2_indices = torch.arange(n, dtype=torch.long, device=device)[None,:,None].expand(B, n, n)
    node_mask = torch.zeros(B, n, L, n, device=device, dtype=torch.bool) # batch, nbest, position, position_nbest
    node_mask[:,list(range(n)),-1,list(range(n))] = 1
    edge_mask = torch.zeros(B, n, M, L, device=device, dtype=torch.bool)
    for i in reversed(range(L)):
        self_mask = node_mask[...,i,:] # B x n x n
        previous_n = (back_edge_indices[i] / M).to(torch.long).reshape(B, 1, n).expand(B, n, n) # B x n x n
        edge_length =  1 + (back_edge_indices[i] % M).reshape(B, 1, n).expand(B, n, n) # B x n x n
        previous_i = i - edge_length # B x n x n
        nonnegative_mask = previous_i[self_mask] >= 0
        node_mask[d1_indices[self_mask][nonnegative_mask], d2_indices[self_mask][nonnegative_mask], previous_i[self_mask][nonnegative_mask], previous_n[self_mask][nonnegative_mask]] = True
        edge_mask[d1_indices[self_mask], d2_indices[self_mask], edge_length[self_mask]-1, i] = True

    last_node_nbest_log_alpha = node_nbest_log_alphas[-1].reshape(*((Bh, N) + node_nbest_log_alphas[-1].size()[1:])) # Bh x N x n
    edge_mask = edge_mask.reshape(*((Bh, N) + edge_mask.size()[1:])) # Bh x N x n x M x L

    # run another viterbi along block dimension treating it as a unigram model
    block_nbest_log_alphas = [torch.zeros((Bh, n), device=device, dtype=torch.float)]
    block_nbest_log_alphas[0][:,1:] = -math.inf
    block_choice_indices = []
    for i in range(N):
        current = last_node_nbest_log_alpha[:, i:i+1, :]  # Bh x 1 x n
        prev = block_nbest_log_alphas[i].unsqueeze(-1) # Bh x n x 1
        block_nbest, block_choice = (current + prev).reshape(Bh, -1).topk(k=n, dim=-1) # Bh x n
        block_choice_indices.append(block_choice)
        block_nbest_log_alphas.append(block_nbest)
    Bh_index = torch.arange(Bh, dtype=torch.long, device=device).repeat_interleave(n)
    selection = torch.arange(n, dtype=torch.long, device=device).repeat(Bh)
    block_nbest_indices = []
    for i in reversed(range(N)):
        current_choice = block_choice_indices[i][Bh_index, selection].reshape(Bh, n)
        current_indices = current_choice % n
        selection = (current_choice / n).to(torch.long).reshape(-1)
        block_nbest_indices.append(current_indices)
    block_nbest_indices = list(reversed(block_nbest_indices)) # N copies of Bh x n
    block_nbest_indices = torch.stack(block_nbest_indices, dim=1) # Bh x N x n
    block_nbest_log_alpha = torch.gather(last_node_nbest_log_alpha, -1, block_nbest_indices)
    block_nbest_edge_mask = torch.gather(edge_mask, 2, block_nbest_indices[...,None,None].expand_as(edge_mask))
    return ViterbiOutput(mask=block_nbest_edge_mask, weight=block_nbest_log_alpha)
