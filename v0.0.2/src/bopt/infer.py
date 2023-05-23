import os

import torch.autograd

from bopt.inference.arguments import parse_arguments
from bopt.inference.setup import setup_classification
from bopt.inference.classification_eval_loop import eval_classification


def main():

    args = parse_arguments()
    s = setup_classification(args)
    eval_classification(s, os.path.join(args.output_directory, args.name))


if __name__ == "__main__":
    main()