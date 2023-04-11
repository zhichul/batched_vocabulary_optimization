import torch

from bopt.utils import increasing_roll_right, increasing_roll_left


def test_rolling():
    ones = torch.ones((1,1,4,5))
    print(increasing_roll_right(ones, 0))
    print(increasing_roll_left(ones, 0))

if __name__=="__main__":
    test_rolling()