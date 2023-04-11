from dataclasses import dataclass
from bopt.unigram_lm_tokenizers.encoding.forward_encoding import NONEDGE_LOGPOT
from bopt.unigram_lm_tokenizers.encoding.utils import convert_to_backward_log_potentials, expand_log_potentials, \
    convert_to_forward_log_potentials, expansion_mask
from bopt.utils import increasing_roll_right
import torch


@dataclass
class ForwardAlgorithmOutput:
    last_node_log_alphas: torch.Tensor = None                   # always set
    edge_log_alphas: torch.Tensor = None                        # always set
    edge_log_alphas_node_contribution: torch.Tensor = None      # optionally set
    all_node_log_alphas: torch.Tensor = None                    # optionally set

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
        maski = (torch.diag_embed(torch.ones(L - i, device=device, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)

        # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
        node_to_edge = node_log_alphas[i][:, None, None] * maski
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

def conditional_marginals(edge_log_potentials: torch.FloatTensor):
    """
    Given a lattice encoded as an M by L edge matrix, compute conditional marginals
    of edges BEFORE {the node after the ith character}, conditioning on
    that the path goes through {the node after the ith character}.
    """
    M, L = edge_log_potentials.size()[-2:]
    forward_o = forward_algorithm(edge_log_potentials, return_node_contribution=True)
    backward_o = forward_algorithm(expand_log_potentials(convert_to_backward_log_potentials(edge_log_potentials)), return_node_contribution=True)
    Z = backward_o.last_node_log_alphas[...,None,None]
    conditional_marginals = ( convert_to_forward_log_potentials(
                              (expand_log_potentials(convert_to_backward_log_potentials(forward_o.edge_log_alphas_node_contribution))  # forward prob of previous node
                            + backward_o.edge_log_alphas_node_contribution)) # backward prob of next node
                            + edge_log_potentials[...,None,:,:] # edge potential
                            - Z) # normalization

    # set invalid positions to -inf
    edge_mask, padedge_mask, nonedge_mask = expansion_mask(M, L, dtype=torch.bool, device=edge_log_potentials.device)
    padedge_mask = increasing_roll_right(padedge_mask.flip(-1), False)
    nonedge_mask = increasing_roll_right(nonedge_mask.flip(-1), False)
    conditional_marginals = conditional_marginals.clone()
    conditional_marginals[nonedge_mask.expand(*conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
    conditional_marginals = conditional_marginals.clone()
    conditional_marginals[padedge_mask.expand(*conditional_marginals.size())] = NONEDGE_LOGPOT # force both padding and nonedge to be -inf
    return conditional_marginals