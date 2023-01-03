import code

import torch
from tqdm import tqdm

from bopt.forward_step import language_modeling_lattice_step, language_modeling_unigram_step

INF = 1e9

def language_modeling_lattice_loop(args, dataloader, tokenizer, model, device):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            loss, ent, lengths, ntokens, _, _ = language_modeling_lattice_step(args, batch, tokenizer, model, device, eval=True)
            batch_size =  lengths.size(0)
            ntokens_total += ntokens.sum().item() - batch_size
            nchars_total += lengths.sum().item() - batch_size
            loss_total += loss.item() * (lengths.sum().item() - batch_size)
    return loss_total / nchars_total, loss_total / ntokens_total, loss_total, nchars_total, ntokens_total


def language_modeling_unigram_loop(args, dataloader, tokenizer, model, device):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, pos_ids, input_mask, labels, lengths, ntokens, text = batch
            loss, ent, lengths, ntokens, _, _ = language_modeling_unigram_step(args, batch, tokenizer, model, device)
            batch_size =  lengths.size(0)
            ntokens_total += ntokens.sum().item() - batch_size
            nchars_total += lengths.sum().item() - batch_size
            loss_total += loss.item() * (labels!=-100).sum().item()
    return loss_total / nchars_total, loss_total / ntokens_total, loss_total, nchars_total, ntokens_total



def language_modeling_lattice_decode_loop(args, dataloader, tokenizer, model, device, remove_csp=True, remove_padding=True):
    decodings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_decodings = language_modeling_lattice_step(args, batch, tokenizer, model, device, decode=True,
                                                             decode_remove_csp=remove_csp,
                                                             decode_remove_padding=remove_padding)
            decodings.extend(batch_decodings)
    return decodings

def language_modeling_unigram_decode_loop(args, dataloader, tokenizer, model, device, remove_csp=True, remove_padding=True):
    decodings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, pos_ids, input_mask, labels, lengths, ntokens, text = batch
            decodings.extend([ [tokenizer.id2str(id, remove_csp=remove_csp) for id in input_id.tolist() if not remove_padding or not tokenizer.is_padding(id)] for input_id in input_ids])
    return decodings

