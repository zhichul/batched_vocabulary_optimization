import code
import math

import torch
from tqdm import tqdm

from bopt.analysis.morpheme_prediction import spans, load_gold_segmentations_morpheme_prediction, morpheme_prediction_segmentation_stats
from bopt.data.language_modeling.utils import viterbi_tokenize, pack_viterbi_chunks
from bopt.forward_step import language_modeling_lattice_step, language_modeling_unigram_step, \
    morpheme_prediction_lattice_step, morpheme_prediction_unigram_step

INF = 1e9
DEBUG1 = False
DEBUG2 = False

def language_modeling_lattice_loop(args, dataloader, tokenizer, model, device, unigram_expert=None, skip_gram=False):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits, loss, ent, lengths, ntokens, _, _, _, _ = language_modeling_lattice_step(args, batch, tokenizer, model, device, eval=True, unigram_expert=unigram_expert, skip_gram=skip_gram)
            batch_size =  lengths.size(0)
            ntokens_total += ntokens.sum().item() - (batch_size if not skip_gram else 0)
            nchars_total += lengths.sum().item() - (batch_size if not skip_gram else 0)
            loss_total += loss.item() * (lengths.sum().item() - batch_size)
    return loss_total / nchars_total, loss_total / ntokens_total, loss_total, nchars_total, ntokens_total


def language_modeling_unigram_loop(args, dataloader, tokenizer, model, device, skip_gram=False):
    loss_total =  0
    ntokens_total =  0
    nchars_total =  0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, pos_ids, input_mask, labels, lengths, ntokens, text = batch
            logits, loss, ent, lengths, ntokens, _, _, _, _ = language_modeling_unigram_step(args, batch, tokenizer, model, device, skip_gram=skip_gram)
            batch_size =  lengths.size(0)
            ntokens_total += ntokens.sum().item() - (batch_size if not skip_gram else 0)
            nchars_total += lengths.sum().item() - (batch_size if not skip_gram else 0)
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
    # for segmentation eval
    tokenizations = []
    tok_precision = -42
    tok_recall = -42
    tok_f1 = -42
    tok_marginal = path_marginal = gold_tok_total = 0
    entropies = []
    masks = []
    leakage = over_attention_total = over_attention_count = entropy_count = over_attention_mass = total_attention_count = 0
    if args.eval_segmentation:
        gseg = load_gold_segmentations_morpheme_prediction(*args.segmentation_dictionary)
        gseg_tokens = load_gold_segmentations_morpheme_prediction(*args.segmentation_dictionary, use_set=False, use_span=False)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, pos_ids, input_mask, label_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, tmask, text = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            logits, loss, ent, lengths, _, _, _, _, astat = morpheme_prediction_lattice_step(args, batch, tokenizer, model, device, eval=True)
            correct_count, label_count1 = zero_one_loss(logits, label_ids)
            correct_prob, label_count2 = expected_zero_one_loss(logits, label_ids)
            assert label_count1 == label_count2 == 3
            zero_one_loss_total += correct_count
            expected_zero_one_loss_total += correct_prob
            num_predictions += label_count1
            batch_size =  lengths.size(0)
            loss_total += loss.item() * batch_size
            example_total += batch_size
            if args.eval_segmentation:
                viterbi_segs = viterbi_tokenize(tokenizer, text, device=input_ids.device, remove_csp=True)
                tokenizations.extend([(t, spans(seg)) for t, seg in zip(text, viterbi_segs)])

                # get the probability of the gold paths
                gold_tokens = [[tok if i == 0 or tokenizer.csp is None else tokenizer.csp + tok for i, tok in enumerate(gseg_tokens[t])] for t in text]
                gold_tok_total += sum(len(toks) for toks in gold_tokens)
                gold_tokens_in_vocab = []
                packed_chunks = []
                for t, gold_tokens_of_word in zip(text, gold_tokens):
                    if all(token in tokenizer.vocab for token in gold_tokens_of_word):
                        gold_tokens_in_vocab.append(gold_tokens_of_word)
                        packed_chunks.append([t])
                gold_chunks, _ = pack_viterbi_chunks(packed_chunks, gold_tokens_in_vocab)
                # encode the chunks into lattice / serial versions
                gold_fwd_ids, gold_fwd_ms, gold_lengths, gold_bwd_ids, gold_bwd_ms, gold_bwd_lengths, gold_mmask, gold_emask = tokenizer.encode_packed_batch(gold_chunks, args.max_unit_length, args.max_block_length, compact=False, verbatim=True, device=input_ids.device)
                all_fwd_ids, all_fwd_ms, all_lengths, all_bwd_ids, all_bwd_ms, all_bwd_lengths, all_mmask, all_emask = tokenizer.encode_packed_batch(packed_chunks, args.max_unit_length, args.max_block_length, compact=False, verbatim=False, device=input_ids.device)

                # path marginal
                gold_log_alphas, _,_ = tokenizer.viterbi_algorithm(tokenizer.get_weights(gold_fwd_ids), gold_fwd_ms, gold_lengths)
                all_log_alphas, all_edge_alphas = tokenizer.forward_algorithm(tokenizer.get_weights(all_fwd_ids), all_fwd_ms, all_lengths)
                path_marginal += (gold_log_alphas - all_log_alphas).exp().sum().item()

                # tok marginal
                B = len(gold_tokens_in_vocab)
                all_log_betas, all_edge_log_betas = tokenizer.forward_algorithm(tokenizer.get_weights(all_bwd_ids), all_bwd_ms, all_bwd_lengths)
                all_edge_log_betas = all_edge_log_betas.reshape(B, -1, *all_edge_log_betas.size()[1:])[:,-1,...].flip(-1)
                triu_ones = torch.triu(torch.ones(args.max_unit_length, args.max_block_length, dtype=torch.bool).to(device), diagonal=0)
                ea = all_edge_alphas[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, -1)  # [B, E] where E  = L (L+1) / 2
                eb = all_edge_log_betas[triu_ones.flip(-1)[None, ...].expand(B, -1, -1)].reshape(B, -1)  # [B, L, E]
                td = tokenizer.get_weights(all_fwd_ids)[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, -1)  # [B, E]
                ms = gold_fwd_ms[triu_ones[None, ...].expand(B, -1, -1)].reshape(B, -1)  # [B, E]
                tok_marginal += (ea + eb - td - all_log_alphas[:,None]).exp()[ms.to(torch.bool)].sum().item()
                if DEBUG1:
                    code.interact(local={k:v for k,v in list(locals().items()) + list(globals().items())})
            if args.log_attention_statistics:
                leakage += astat["leakage"]
                over_attention_mass += astat["over_attention_mass"]
                total_attention_count += astat["total_attention_count"]
                over_attention_total += astat["over_attention_mean"] * astat["over_attention_count"]
                over_attention_count += astat["over_attention_count"]
                entropy_count += astat["entropy_count"]
                entropies.append(astat["entropy"])
                masks.append(astat["mask"])
    if args.eval_segmentation and len(tokenizations) != example_total:
        raise AssertionError
    if args.eval_segmentation:
        tp, pred, true = morpheme_prediction_segmentation_stats(tokenizations, gseg)
        tok_precision = tp / pred
        tok_recall = tp / true
        tok_f1 = 2 * tok_recall * tok_precision / (tok_recall + tok_precision)
    astats = {}
    if args.log_attention_statistics:
        astats["leakage"] = leakage
        astats["over_attention_mean"] = over_attention_total / over_attention_count if over_attention_count > 0 else 0.0
        astats["over_attention_count"] = over_attention_count
        astats["over_attention_mass"] = over_attention_mass
        astats["total_attention_count"] = total_attention_count # is the number of atten values
        astats["entropy_mean"] = sum(entropy.sum().item() for entropy in entropies) / entropy_count
        astats["total_attention_dist_count"] = entropy_count # is the number of atten distribuitnos
        astats["entropy_std"] = math.sqrt(sum((((entropy - astats["entropy_mean"]) * mask[...,0]) ** 2).sum().item() for mask, entropy in zip(masks, entropies)) / entropy_count)
    if DEBUG2:
        code.interact(local={k: v for k, v in list(locals().items()) + list(globals().items())})
    return loss_total / example_total, zero_one_loss_total / num_predictions, expected_zero_one_loss_total / num_predictions, example_total, num_predictions, tok_precision, tok_recall, tok_f1, path_marginal/example_total, tok_marginal/gold_tok_total, astats

def morpheme_prediction_unigram_loop(args, dataloader, tokenizer, model, device):
    loss_total = 0
    zero_one_loss_total = 0
    expected_zero_one_loss_total = 0
    num_predictions = 0
    example_total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, pos_ids, input_mask, label_ids = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
            logits, loss, _, _, _, _, _, _, _ = morpheme_prediction_unigram_step(args, batch, tokenizer, model, device)
            correct_count, label_count1 = zero_one_loss(logits, label_ids)
            correct_prob, label_count2 = expected_zero_one_loss(logits, label_ids)
            assert label_count1 == label_count2 == 3
            zero_one_loss_total += correct_count
            expected_zero_one_loss_total += correct_prob
            num_predictions += label_count1
            batch_size = logits.size(0)
            loss_total += loss.item() * batch_size
            example_total += batch_size
    return loss_total / example_total, zero_one_loss_total / num_predictions, expected_zero_one_loss_total / num_predictions, example_total, num_predictions, None, None, None,None, None, None

