import torch
from bopt.forward_step import language_modeling_step
INF = 1e9

def language_modeling_loop(args, dataloader, tokenizer, model, device):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in dataloader:
            batch_size =  lengths.size(0)
            loss, ent, lengths, ntokens = language_modeling_step(args, batch, tokenizer, model, device)
            ntokens_total += ntokens.sum().item() - batch_size
            nchars_total += lengths.sum().item() - batch_size
            loss_total += loss.item() * (lengths.sum().item() - batch_size)
    return loss_total / nchars_total, loss_total / ntokens_total, loss_total, nchars_total, ntokens_total


