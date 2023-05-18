import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizers(args, model, tokenizer, model_lr, tokenizer_lr):
    """
    Sets up separate learning rates for the task model and the tokenizer parameters.

    Sets up a learning rate scheduler that decreases by a factor of reduce_factor when
    loss haven't improved for patience epochs.

    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    model_params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'name': 'model_decay', 'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0}, # TODO: if we want weight decay
        {'name': 'model_nodecay','params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    if args.input_tokenizer_learning_rate:
        optimizer_grouped_parameters.append({'name': 'tokenizer', 'params': tokenizer.parameters(), 'weight_decay': 0.0, 'lr': tokenizer_lr})
    optimizer = Adam(optimizer_grouped_parameters, lr=model_lr)
    optimizer.named_param_groups = {group["name"]: group for group in optimizer.param_groups}
    plat_scheduler = ReduceLROnPlateau(optimizer,'min', factor=args.reduce_factor, patience=args.patience)
    return optimizer, plat_scheduler