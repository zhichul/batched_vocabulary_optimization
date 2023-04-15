import torch.autograd

from bopt.tokenization.arguments import parse_arguments
from bopt.tokenization.setup import setup
from bopt.tokenization.tokenization_loop import tokenization_loop


def main():

    args = parse_arguments()
    s = setup(args)
    tokenization_loop(s)



if __name__ == "__main__":
    main()