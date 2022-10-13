import code

import torch

from bopt.core.utils import increasing_roll_left, increasing_roll_right

INF = 1e9

def morpheme_prediction_step(args, batch, tokenizer, model, device):
    batch = [t.to(device) for t in batch]
    input_ids, pos_ids, input_mask, label_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask, tmask = batch
    # dp lattice if necessasry
    ent, a, m, c = None, None, None, None
    if args.vopt:
        ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                                 bwd_ids, bwd_ms, bwd_lengths,
                                 mmask, emask, tmask)

    # run model
    losses = model(input_ids=input_ids, position_ids=pos_ids, labels=label_ids, attn_bias=a if args.vopt else None)

    # get loss
    loss = losses[0] * args.main_loss_multiplier
    return loss, ent, lengths, None

def language_modeling_step(args, batch, tokenizer, model, device):
    batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch ]
    input_ids, pos_ids, input_mask, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask, binary_mask, txt, ntokens = batch
    batch_size, N, M, L = fwd_ids.size()
    B = args.max_blocks

    # do some prep
    bwd_ids = (bwd_ids.unsqueeze(2) * emask.to(torch.long) + tokenizer.pad_index * (1-emask.to(torch.long))).reshape(batch_size, B * L, M, L)
    bwd_ms = (bwd_ms.unsqueeze(2) * mmask).reshape(batch_size, B * L, M, L)
    global_mask = binary_mask.unsqueeze(1) * binary_mask.unsqueeze(2) * tokenizer.causal_mask(N, L, M, device=device).unsqueeze(0)
    # dp lattice if necessasry
    ent, a, m, c = None, None, None, None
    if args.vopt:
        ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                                 bwd_ids, bwd_ms, bwd_lengths,
                                 mmask, emask, None, lm=True, lm_mask=global_mask)

    # run model
    losses = model(input_ids=input_ids, position_ids=pos_ids, attn_bias=a if args.vopt else None, return_dict=True)

    # get indices
    indices = increasing_roll_left(fwd_ids, tokenizer.pad_index).transpose(-1, -2).reshape(batch_size, N * L, M) # batch x NL x M

    # get output probabilities
    log_edge_weights = torch.log_softmax(losses["logits"][:, -N * L:, :], -1) # batch x NL x V

    # get log probs and convert back to transition matrix
    output_fwd_ts = torch.gather(log_edge_weights, -1, indices).reshape(batch_size, N, L, M).transpose(-1,-2) # batch x N x M x L
    bos_mask = torch.ones_like(output_fwd_ts)
    bos_mask[:,0,:,0] = 0 # first column of first block is bos
    output_fwd_ts = bos_mask * output_fwd_ts
    output_fwd_ts = increasing_roll_right(output_fwd_ts, 0)
    output_fwd_ts = output_fwd_ts * fwd_ms + (1-fwd_ms) * -INF

    # compute prob
    log_alphas, _= tokenizer.forward_algorithm(output_fwd_ts.reshape(batch_size * N, M, L), fwd_ms.reshape(batch_size * N, M, L), lengths.reshape(batch_size * N))

    # get loss
    nchars = lengths.sum() - batch_size # adjust for BOS
    log_probs = log_alphas.sum()
    loss = -log_probs / nchars * args.main_loss_multiplier
    return loss, ent, lengths, ntokens

