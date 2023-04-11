import torch


def entropy(edge_log_potentials):
    size = edge_log_potentials.size()
    device = edge_log_potentials.device
    M, L = size[-2:]
    size_prefix = size[:-2]
    edge_log_potentials = edge_log_potentials.reshape(-1, M, L).double()
    edge_log_alphas = edge_log_potentials.clone() # this is the edge log alphas without the node contribution

    edge_entropy_aggregate = torch.zeros_like(edge_log_potentials).double() # this is the edge log alphas without the node contribution
    edge_entropy_individual = - torch.clamp(edge_log_potentials,min=-1e9) * edge_log_potentials.exp()  # -plogp

    # forward algorithm (converts internally to B' x M x L where B' collapses all the dimesions in size_prefix into a single one)
    node_log_alphas = [torch.zeros(edge_log_potentials.numel() // (M * L), device=device, dtype=torch.double)]
    node_entropies = [torch.zeros(edge_log_potentials.numel() // (M * L), device=device, dtype=torch.double)]
    for i in range(L):
        # this is to select the outgoing edges of the ith node
        maski = (torch.diag_embed(torch.ones(L - i, device=device, dtype=torch.bool), offset=i)[:M].unsqueeze(0)).to(torch.float)

        # this update corresponds to the propagation of alpha from a node to all `outgoing` edges
        node_to_edge = node_log_alphas[i][:, None, None] * maski
        edge_log_alphas = edge_log_alphas + node_to_edge
        node_to_edge_ent = ((node_log_alphas[i][:, None, None].exp() * maski) * (edge_entropy_individual * maski)
                            + (node_entropies[i][:, None, None] * maski) * (edge_log_potentials.exp() * maski))
        edge_entropy_aggregate = edge_entropy_aggregate + node_to_edge_ent

        # this update corresponds to aggregating the `incoming` edge-alphas into the alpha of a node
        node_log_alphas.append(edge_log_alphas[..., i].logsumexp(-1))
        node_entropies.append(torch.sum(edge_entropy_aggregate[:, :, i], -1))

    # converts back to prefix_size
    # edge_log_alphas = edge_log_alphas.reshape(*(size_prefix + (M, L)))
    last_node_log_alphas = node_log_alphas[-1].reshape(*size_prefix)
    last_node_entropy_aggregate = node_entropies[-1].reshape(*size_prefix)

    return (last_node_entropy_aggregate / last_node_log_alphas.exp() + last_node_log_alphas)

