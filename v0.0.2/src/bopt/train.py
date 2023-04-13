import torch.autograd

from bopt.training.arguments import parse_arguments
from bopt.training.setup import setup_classification
from bopt.training.classification_train_loop import train_classification


def main():

    args = parse_arguments()
    if args.task == "classification":
        setup = setup_classification(args)
        train_classification(setup)



if __name__ == "__main__":
    main()