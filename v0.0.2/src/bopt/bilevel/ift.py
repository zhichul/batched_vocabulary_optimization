import math

import torch
from tqdm import tqdm

from bopt.bilevel.classification_loss_inner import train_loss_inner


def approx_inverse_vhp(v, f, w, I, scaling, threshold=1e-3):
    """
    Algorithm 3 of Lorraine et al. 2019
    """
    p = tuple(u.clone() for u in v)
    bar = tqdm(range(I))
    vinitial = math.sqrt(sum(u.norm() ** 2 for u in v).item())
    for i in bar:
        jvp = torch.autograd.grad(f, w, grad_outputs=v, create_graph=False, retain_graph=True)
        v = tuple(u - scaling * w for u,w in zip(v, jvp))
        p = tuple(u + w for u, w in zip(p, v))
        vnorm = math.sqrt(sum(u.norm() ** 2 for u in v).item())
        pnorm = math.sqrt(sum(u.norm() ** 2 for u in p).item())
        if vnorm / pnorm < threshold: # relative threshold
            break
        bar.set_description_str(f"Approx Inv VHP: dnorm / norm: {vnorm / pnorm:.2f} shrink: {vnorm:.2} / {vinitial:.2} = {vnorm / vinitial if vinitial != 0 else math.inf:.2f}")
        # code.interact(local=locals())
    return tuple(scaling * u for u in p)

def hyper_gradient(inner_loss, inner_params, outer_params, neumann_iterations, neumann_alpha, neumann_threshold):
    """
    Assume the outer grad is aggregated at the grad field of inner_params
    and outer_params respectively.
    Hence this does not require the outer loss.
    The inner loss may be a stochastic estimate since the entire training set
    may not fit on the gpu.
    """
    # the train gradient computation graph, there might be some unused bert parameters so allow_unused=True
    dLtdw = torch.autograd.grad(inner_loss, inner_params, create_graph=True, retain_graph=True, allow_unused=True)
    used_inner_params_mask = tuple(grad is not None for grad in dLtdw)
    used_inner_params = tuple(param for param, m in zip(inner_params, used_inner_params_mask) if m)
    dLtdw = tuple(grad for grad, m in zip(dLtdw, used_inner_params_mask) if m) # this is to match the masking

    # Algorithm 2 of lorraine et al. 2019
    v1 = tuple(p.grad for p in used_inner_params)  # dLVdw
    # code.interact(local=locals())
    v2 = approx_inverse_vhp(v1, dLtdw, used_inner_params, neumann_iterations, neumann_alpha, threshold=neumann_threshold)
    v3 = torch.autograd.grad(dLtdw, outer_params, grad_outputs=v2, allow_unused=True)
    return tuple(u * -1 for u in v3)

def hyper_step(setup):
    Lt = train_loss_inner(setup)  # train loss graph
    indirect_grad = hyper_gradient(Lt,
                          list(setup.classifier.model.parameters()),
                          list(setup.classifier.input_tokenizer.parameters()),
                          setup.args.neumann_iterations,
                          setup.args.neumann_alpha,
                          setup.args.neumann_threshold)
    assert len(indirect_grad) == len(setup.classifier.input_tokenizer.parameters())
    indirect_gradient_norm = 0
    direct_gradient_norm = 0
    total_gradient_norm = 0
    for param, ind_grad in zip(setup.classifier.input_tokenizer.parameters(), indirect_grad):
        indirect_gradient_norm += (ind_grad ** 2).sum()
        direct_gradient_norm += (param.grad ** 2).sum()
        total_gradient_norm += ((param.grad + ind_grad) ** 2).sum()
        param.grad += ind_grad
    indirect_gradient_norm = math.sqrt(indirect_gradient_norm.item())
    direct_gradient_norm = math.sqrt(direct_gradient_norm.item())
    total_gradient_norm = math.sqrt(total_gradient_norm.item())
    return {"indirect_gradient_norm": indirect_gradient_norm, "direct_gradient_norm": direct_gradient_norm, "total_gradient_norm": total_gradient_norm}
