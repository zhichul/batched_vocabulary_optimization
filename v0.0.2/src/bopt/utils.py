import torch


def increasing_roll_left(mat: torch.Tensor, padding_value):
    """
    This function rolls the rows of the input matrix (or tensor) by increasing
    amounts to the left. This is not exactly the inverse to increasing_roll_right
    but close. The reason it is not an inverse is that increasing_roll_right
    throws away elements of the matrix and pads. Rolling left moves those
    thrown away elements back into position but their value is replaced by
    padding.

    For example: the following matrix was produced by increasing_roll_right

        [[1,2,3,4],
        [-1,5,6,7],
        [-1,-1,9,10]]

    would be rolled back into

        [[1,2,3,4],
        [5,6,7,-1],
        [9,10,0,0]]

    where 0 is the padding value.

    Usually it is used in conjunction with increasing_roll_right with the same
    padding value.
    """
    size = mat.size()
    if not len(size) > 2:
        raise ValueError(mat.size())
    rows, cols = size[-2:]
    size_prefix = size[:-2]
    padding = torch.zeros(size_prefix + (rows,), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat.reshape(*(size_prefix + (-1,))), padding], dim=-1).reshape(*(size_prefix + (rows, cols + 1)))
    out = rolled[..., :cols]
    return out

def increasing_roll_right(mat: torch.Tensor, padding_value):
    """
    This function rolls the rows of the input matrix (or tensor) by increasing
    amounts to the right.

    For example:

        [[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]]

    would be rolled into

        [[1,2,3,4],
        [0,5,6,7],
        [0,0,9,10]]

    where 0 is the padding value.

    """
    size = mat.size()
    if not len(size) >= 2:
        raise ValueError(mat.size())
    rows, cols = size[-2:]
    size_prefix = size[:-2]
    padding = torch.zeros(size[:-2] + (rows,1), dtype=mat.dtype, device=mat.device)
    padding.fill_(padding_value)
    rolled = torch.cat([mat, padding], dim=-1).reshape(*(size_prefix + (rows * (cols + 1),)))[...,:-rows].reshape(*(size_prefix+ (rows,cols)))
    return rolled

