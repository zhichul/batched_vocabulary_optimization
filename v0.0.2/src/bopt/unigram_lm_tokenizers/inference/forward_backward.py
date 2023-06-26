import code
from dataclasses import dataclass
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_LOGPOT
from bopt.unigram_lm_tokenizers.utils.encoding import convert_to_backward_log_potentials, expand_log_potentials, \
    convert_to_forward_log_potentials, expansion_mask, lattice_mask
from bopt.utils import increasing_roll_right, col_shift
import torch


@dataclass
class ForwardAlgorithmOutput:
    last_node_log_alphas: torch.Tensor = None                   # always set
    edge_log_alphas: torch.Tensor = None                        # always set
    edge_log_alphas_node_contribution: torch.Tensor = None      # optionally set
    all_node_log_alphas: torch.Tensor = None                    # optionally set

@dataclass
class ConditionalMarginalsOutput:
    backward_conditional_marginals: torch.Tensor = None         # always set
    forward_conditional_marginals: torch.Tensor = None          # optionally set

def forward_algorithm(edge_log_potentials: torch.FloatTensor, return_node_contribution=False):
    size = edge_log_potentials.size()
    device = edge_log_potentials.device
    M, L = size[-2:]
    size_prefix = size[:-2]
    edge_log_alphas = edge_log_potentials.reshape(-1, M, L).clone() # this is the edge log alphas without the node contribution
    node_contributions = 0 if return_node_contribution else None

    # forward algorithm (converts internally to B' x M x L where B' collapses all the dimesions in size_prefix into a single one)
    node_log_alphas = [torch.zeros(edge_log_potentials.numel() // (M * L), device=device, dtype=torch.float)]
    for i in range(L):
        # this is to select the outgoing edges of the ith node
        maski = (torch.diag_embed(torch.ones(L - i, device=device, dtype=torch.bool), offset=i)[:M].unsqueeze(0)) #.to(torch.float)

        # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
        # node_to_edge = node_log_alphas[i][:, None, None] * maski
        node_to_edge = node_log_alphas[i][:, None, None].expand_as(edge_log_alphas).clone()
        node_to_edge[~maski.expand_as(edge_log_alphas)] = 0
        edge_log_alphas = edge_log_alphas + node_to_edge

        # this is bookkeeping to record the node contributions without adding in the edge potentials
        if return_node_contribution:
            node_contributions = node_contributions + node_to_edge

        # this update corresponds to aggregating the `incoming` edge-alphas into the alpha of a node
        node_log_alphas.append(edge_log_alphas[..., i].logsumexp(-1))

    # converts back to prefix_size
    edge_log_alphas = edge_log_alphas.reshape(*(size_prefix + (M, L)))
    last_node_log_alphas = node_log_alphas[-1].reshape(*size_prefix)
    if return_node_contribution:
        node_contributions = node_contributions.reshape(*(size_prefix + (M,L)))

    return ForwardAlgorithmOutput(last_node_log_alphas, edge_log_alphas, edge_log_alphas_node_contribution=node_contributions)

def conditional_marginals(edge_log_potentials: torch.FloatTensor, return_forward=True):
    """
    Given a lattice encoded as an M by L edge matrix, compute backward
    conditional marginals of edges BEFORE and AFTER {the node after the ith character},
    conditioning on that the path goes through {the node after the ith character}.
    i goes from 0 to L. Returns one (or two) size_prefix x L+1 x M x L matrices.
    """
    M, L = edge_log_potentials.size()[-2:]
    size_prefix = edge_log_potentials.size()[:-2]
    forward_o = forward_algorithm(edge_log_potentials, return_node_contribution=True)
    backward_o = forward_algorithm(expand_log_potentials(convert_to_backward_log_potentials(edge_log_potentials)), return_node_contribution=True)
    expanded_edge_log_potentials = edge_log_potentials[..., None, :, :]  # edge potential                               ... x 1 x M x L
    edge_complete_betas = convert_to_forward_log_potentials(backward_o.edge_log_alphas_node_contribution)[...,-1:,:,:] # all edge complete betas           ... x 1 x M x L
    char_complete_betas = convert_to_forward_log_potentials(backward_o.edge_log_alphas_node_contribution)[...,-1,0,:,None,None] # character complete betas ... x L x 1 x 1
    char_all_betas = col_shift(increasing_roll_right(convert_to_forward_log_potentials(backward_o.edge_log_alphas_node_contribution)[...,:1,:].transpose(-1,-3).expand(*(size_prefix + (L,M,L))), NONEDGE_LOGPOT), 1, NONEDGE_LOGPOT) # character all betas      ... x L x M x L
    # backward conditionals
    backward_Z = backward_o.last_node_log_alphas[...,None,None]
    backward_conditional_marginals = ( convert_to_forward_log_potentials(
                              (expand_log_potentials(convert_to_backward_log_potentials(forward_o.edge_log_alphas_node_contribution))  # forward prob of previous node
                            + backward_o.edge_log_alphas_node_contribution)) # backward prob of next node
                            + expanded_edge_log_potentials
                            - backward_Z) # normalization

    if return_forward:
        # forward conditionals
        forward_Z = char_complete_betas # + char_complete_alphas + char_edge_potentials (cancelled out with numerator)
        forward_conditional_marginals = ( char_all_betas
                                + expanded_edge_log_potentials
                                + edge_complete_betas
                                # + char_complete_alphas + char_edge_potentials (cancelled out with denominator)
                                - forward_Z) # normalization

    # set invalid positions to -inf
    edge_mask, padedge_mask, nonedge_mask = expansion_mask(M, L, dtype=torch.bool, device=edge_log_potentials.device)
    padedge_mask = increasing_roll_right(padedge_mask.flip(-1), False)
    nonedge_mask = increasing_roll_right(nonedge_mask.flip(-1), False)
    backward_conditional_marginals = backward_conditional_marginals.clone()
    backward_conditional_marginals[nonedge_mask.expand(*backward_conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
    backward_conditional_marginals = backward_conditional_marginals.clone()
    backward_conditional_marginals[padedge_mask.expand(*backward_conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
    backward_padding = backward_conditional_marginals.new_zeros((size_prefix + (1, M, L))).fill_(NONEDGE_LOGPOT)
    backward_conditional_marginals = torch.cat([backward_padding, backward_conditional_marginals], dim=-3)

    if return_forward:
        edge_mask, padedge_mask, nonedge_mask = expansion_mask(M, L, dtype=torch.bool, device=edge_log_potentials.device, longest_first=True)
        nonedge_mask = col_shift(nonedge_mask, 1, 1)
        padedge_mask = col_shift(padedge_mask, 1, 0)
        forward_conditional_marginals = forward_conditional_marginals.clone()
        forward_conditional_marginals[nonedge_mask.expand(*forward_conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
        forward_conditional_marginals = forward_conditional_marginals.clone()
        forward_conditional_marginals[padedge_mask.expand(*forward_conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
        forward_padding = backward_conditional_marginals[...,-1:,:,:] # just the unconditional marginals
        forward_conditional_marginals = torch.cat([forward_padding, forward_conditional_marginals], dim=-3)

    return ConditionalMarginalsOutput(backward_conditional_marginals, forward_conditional_marginals=forward_conditional_marginals if return_forward else None)