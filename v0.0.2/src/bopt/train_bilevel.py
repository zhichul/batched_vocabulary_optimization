import torch.autograd

from bopt.bilevel.arguments import parse_arguments
from bopt.bilevel.setup import setup_classification
from bopt.bilevel.classification_train_outer import train_classification_outer

def main():
    # torch.set_anomaly_enabled(True)
    args = parse_arguments()
    if args.task == "classification":
        setup = setup_classification(args)
        if args.log_learning_dynamics:
            raise NotImplementedError
        else:
            train_classification_outer(setup)



if __name__ == "__main__":
    main()