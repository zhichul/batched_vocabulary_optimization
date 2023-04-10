from dataclasses import dataclass
import torch

@dataclass
class ForwardAlgorithmOutput:
    last_node_log_alphas: torch.Tensor = None       # always set
    edge_log_alphas: torch.Tensor = None            # always set
    all_node_log_alphas: torch.Tensor = None        # optionally set

def forward_algorithm(edge_log_potentials: torch.FloatTensor):
    size = edge_log_potentials.size()
    device = edge_log_potentials.device
    M, L = size[-2:]
    size_prefix = size[:-2]
    edge_log_alphas = edge_log_potentials.reshape(-1, M, L) # this is the edge log alphas without the node contribution

    # forward algorithm
    node_log_alphas = [torch.zeros(size_prefix, device=device, dtype=torch.float)]
    for i in range(L):
        # this is to select the outgoing edges of the ith node
        maski = (torch.diag_embed(torch.ones(L - i, device=device, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)

        # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
        node_to_edge = node_log_alphas[i][:, None, None] * maski
        edge_log_alphas = edge_log_alphas + node_to_edge

        # this update corresponds to aggregating the `incoming` edge-alphas into the alpha of a node
        node_log_alphas.append(edge_log_alphas[..., i].logsumexp(-1))

    edge_log_alphas = edge_log_alphas.reshape(*(size_prefix + (M, L)))
    last_node_log_alphas = node_log_alphas[-1].reshape(*size_prefix)
    return ForwardAlgorithmOutput(last_node_log_alphas, edge_log_alphas)

def conditional_marginals(edge_log_potentials: torch.FloatTensor):
    pass