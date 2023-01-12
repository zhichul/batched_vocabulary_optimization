import code

import torch
from tqdm import tqdm

from bopt.forward_step import language_modeling_lattice_step, language_modeling_unigram_step, \
    morpheme_prediction_lattice_step, morpheme_prediction_unigram_step

INF = 1e9

def language_modeling_lattice_loop(args, dataloader, tokenizer, model, device, unigram_expert=None, skip_gram=False):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits, loss, ent, lengths, ntokens, _, _ = language_modeling_lattice_step(args, batch, tokenizer, model, device, eval=True, unigram_expert=unigram_expert, skip_gram=skip_gram)
            batch_size =  lengths.size(0)
            ntokens_total += ntokens.sum().item() - batch_size
            nchars_total += lengths.sum().item() - batch_size
            loss_total += loss.item() * (lengths.sum().item() - batch_size)
    return loss_total / nchars_total, loss_total / ntokens_total, loss_total, nchars_total, ntokens_total


def language_modeling_unigram_loop(args, dataloader, tokenizer, model, device, skip_gram=False):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, pos_ids, input_mask, labels, lengths, ntokens, text = batch
            logits, loss, ent, lengths, ntokens, _, _ = language_modeling_unigram_step(args, batch, tokenizer, model, device, skip_gram=skip_gram)
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

def zero_one_loss(logits, labels):
    # they are batch x seq_len x out_vocab_size
    predictions = logits.topk(1, dim=-1)[1].squeeze(-1)
    label_mask = labels != -100
    label_count = label_mask.sum().item()
    label_linear = labels[label_mask]
    prediction_linear = predictions[label_mask]
    correct_count = (label_linear == prediction_linear).sum()
    return correct_count.item(), label_count

def expected_zero_one_loss(logits, labels):
    # they are batch x seq_len x out_vocab_size
    labels_c = labels.clone().detach()
    labels_c[labels_c == -100] = 0
    predictions = torch.gather(logits.softmax(-1), -1, labels_c[:,:,None]).squeeze(-1)
    label_mask = labels != -100
    label_count = label_mask.sum().item()
    prediction_linear = predictions[label_mask]
    correct_prob = (prediction_linear).sum()
    return correct_prob.item(), label_count

def morpheme_prediction_lattice_loop(args, dataloader, tokenizer, model, device):
    loss_total = 0
    zero_one_loss_total = 0
    expected_zero_one_loss_total = 0
    num_predictions = 0
    example_total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, pos_ids, input_mask, label_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, tmask = [t.to(device) for t in batch]
            logits, loss, ent, lengths, _, _, _ = morpheme_prediction_lattice_step(args, batch, tokenizer, model, device)
            correct_count, label_count1 = zero_one_loss(logits, label_ids)
            correct_prob, label_count2 = expected_zero_one_loss(logits, label_ids)
            assert label_count1 == label_count2 == 3
            zero_one_loss_total += correct_count
            expected_zero_one_loss_total += correct_prob
            num_predictions += label_count1
            batch_size =  lengths.size(0)
            loss_total += loss.item() * batch_size
            example_total += batch_size
    return loss_total / example_total, zero_one_loss_total / num_predictions, expected_zero_one_loss_total / num_predictions, example_total, num_predictions

def morpheme_prediction_unigram_loop(args, dataloader, tokenizer, model, device):
    loss_total = 0
    zero_one_loss_total = 0
    expected_zero_one_loss_total = 0
    num_predictions = 0
    example_total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, pos_ids, input_mask, label_ids = [t.to(device) for t in batch]
            logits, loss, _, _, _, _, _ = morpheme_prediction_unigram_step(args, batch, tokenizer, model, device)
            correct_count, label_count1 = zero_one_loss(logits, label_ids)
            correct_prob, label_count2 = expected_zero_one_loss(logits, label_ids)
            assert label_count1 == label_count2 == 3
            zero_one_loss_total += correct_count
            expected_zero_one_loss_total += correct_prob
            num_predictions += label_count1
            batch_size = logits.size(0)
            loss_total += loss.item() * batch_size
            example_total += batch_size
    return loss_total / example_total, zero_one_loss_total / num_predictions, expected_zero_one_loss_total / num_predictions, example_total, num_predictions

