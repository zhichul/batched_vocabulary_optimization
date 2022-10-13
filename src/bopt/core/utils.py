import torch

def increasing_roll_left(mat: torch.Tensor, padding_value, shift=False):
    size = mat.size()
    if not len(size) > 2:
        raise ValueError(mat.size())
    R, C = size[-2:]
    padding = torch.zeros(size[:-2] + (R,), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat.reshape(*(size[:-2] + (-1,))), padding], dim=-1).reshape(*(size[:-2] + (R, C + 1)))
    out = rolled[..., :C]
    return out

def increasing_roll_right(mat: torch.Tensor, padding_value):
    size = mat.size()
    if not len(size) > 2:
        raise ValueError(mat.size())
    R, C = size[-2:]
    padding = torch.zeros(size[:-2] + (R,1), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat, padding], dim=-1).reshape(*(size[:-2] + (R * (C + 1),)))[...,:-R].reshape(*(size[:-2] + (R,C)))
    return rolled