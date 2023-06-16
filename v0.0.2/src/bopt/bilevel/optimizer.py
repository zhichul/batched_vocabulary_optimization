import code

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_optimizers(args, model, tokenizer, model_lr, tokenizer_lr, embedding_lr=None):
    """
    Sets up separate learning rates for the task model and the tokenizer parameters.

    Sets up a learning rate scheduler that decreases by a factor of reduce_factor when
    loss haven't improved for patience epochs.

    """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    model_params = list(model.named_parameters())
    if embedding_lr is None:
        embedding_lr = model_lr
    inner_optimizer_grouped_parameters = [
        {'name': 'model_decay_embed', 'params': [p for n, p in model_params if not any(nd in n for nd in no_decay) and ".embeddings." in n], 'weight_decay': 0.0, "lr": embedding_lr}, # TODO: if we want weight decay
        {'name': 'model_decay', 'params': [p for n, p in model_params if not any(nd in n for nd in no_decay) and ".embeddings." not in n], 'weight_decay': 0.0}, # TODO: if we want weight decay
        {'name': 'model_nodecay_embed','params': [p for n, p in model_params if any(nd in n for nd in no_decay) and ".embeddings." in n], 'weight_decay': 0.0, "lr": embedding_lr},
        {'name': 'model_nodecay', 'params': [p for n, p in model_params if any(nd in n for nd in no_decay) and  ".embeddings." not in n], 'weight_decay': 0.0},
        {'name': 'tokenizer', 'params': tokenizer.parameters(), 'weight_decay': 0.0, 'lr': 0.0}
    ]
    outer_optimizer_grouped_parameters = [{'name': 'tokenizer', 'params': tokenizer.parameters(), 'weight_decay': 0.0, 'lr': tokenizer_lr}]
    if args.inner_optimizer == "SGD":
        inner_optimizer = SGD(inner_optimizer_grouped_parameters, lr=model_lr)
    elif args.inner_optimizer == "Adam":
        inner_optimizer = Adam(inner_optimizer_grouped_parameters, lr=model_lr)
    outer_optimizer = Adam(outer_optimizer_grouped_parameters, lr=tokenizer_lr)
    inner_optimizer.named_param_groups = {group["name"]: group for group in inner_optimizer.param_groups}
    outer_optimizer.named_param_groups = {group["name"]: group for group in outer_optimizer.param_groups}
    inner_plat_scheduler = ReduceLROnPlateau(inner_optimizer,'min', factor=args.reduce_factor, patience=args.patience)
    outer_plat_scheduler = ReduceLROnPlateau(outer_optimizer,'min', factor=args.reduce_factor, patience=args.patience)
    return inner_optimizer, outer_optimizer, inner_plat_scheduler, outer_plat_scheduler