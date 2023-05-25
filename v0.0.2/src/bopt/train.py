import torch.autograd

from bopt.training.arguments import parse_arguments
from bopt.training.setup import setup_classification
from bopt.training.classification_train_loop import train_classification
from bopt.learning_dynamics.classification_train_loop import train_classification as dynamics_train_classification


def main():
    # torch.set_anomaly_enabled(True)
    args = parse_arguments()
    if args.task == "classification":
        setup = setup_classification(args)
        if args.log_learning_dynamics:
            dynamics_train_classification(setup)
        else:
            train_classification(setup)



if __name__ == "__main__":
    main()