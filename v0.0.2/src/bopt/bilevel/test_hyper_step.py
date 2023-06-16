import torch
import torch.nn as nn

from bopt.bilevel.classification_train_outer import hyper_gradient


def test1():
    # this is the situation where you initialize at w*(lambda)
    param_inner = nn.Parameter(torch.tensor(3.6))
    param_outer = nn.Parameter(torch.tensor(3.6))
    # loss = 1/2|param_inner - param_outer|^2, hessian should be 1
    inner_loss = 0.5 * (param_inner - param_outer) **2
    outer_loss = 0.5 * (param_inner - 1.6) ** 2
    outer_loss.backward()
    # dLvdw should be 3.6 - 1.6 = 2,
    # dLtdwdlambda should be -1
    # so indirect_grad = dLvdw @ - invH @ dLtdwdlambda = 2
    (indirect_grad,) = hyper_gradient(inner_loss, (param_inner,), (param_outer,), 10000, 0.001, -1)
    print(indirect_grad.item())
    if abs(indirect_grad.item()- 2) > 1e-3:
        raise AssertionError

def test2():
    # this is the situation where you initialize at w*(lambda)
    param_inner = nn.Parameter(torch.tensor(3.5))
    param_outer = nn.Parameter(torch.tensor(1.8))
    # loss = 1/2|param_inner - param_outer|^2, hessian should be 2
    inner_loss = (param_inner - 2 * param_outer) **2
    outer_loss = 0.5 * (param_inner - 1.6) ** 2
    outer_loss.backward()
    # dLvdw should be 3.5 - 1.6 = 1.90,
    # dLtdwdlambda should be -4
    # so indirect_grad = dLvdw @ - invH @ dLtdwdlambda = 3.8
    (indirect_grad,) = hyper_gradient(inner_loss, (param_inner,), (param_outer,), 10000, 0.001, -1)
    print(indirect_grad.item())
    if abs(indirect_grad.item()- 3.8) > 1e-3:
        raise AssertionError
if __name__ == "__main__":
    test1()
    test2()