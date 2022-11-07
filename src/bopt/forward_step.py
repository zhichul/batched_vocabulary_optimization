import code
import json
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

from bopt.core.utils import increasing_roll_left, increasing_roll_right, forever_generator
from bopt.data.logging.lattice_loggers import LOGGERS

INF = 1e9
DEBUG = False

def morpheme_prediction_lattice_step(args, batch, tokenizer, model, device):
    batch = [t.to(device) for t in batch]
    input_ids, pos_ids, input_mask, label_ids, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms, bwd_lengths, mmask, emask, tmask = batch
    # dp lattice if necessasry
    ent, a, m, c = None, None, None, None
    if args.vopt:
        ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                                 bwd_ids, bwd_ms, bwd_lengths,
                                 mmask, emask, tmask, marginal_temperature=args.marginal_temperature)

    # run model
    losses = model(input_ids=input_ids, position_ids=pos_ids, labels=label_ids, attn_bias=a if args.vopt else None)

    # get loss
    loss = losses[0] * args.main_loss_multiplier
    return loss, ent, lengths, None

def language_modeling_lattice_step(args, batch, tokenizer, model, device, eval=False):
    batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch ]
    if args.output_viterbi:
        input_ids, pos_ids, input_mask, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms_c, bwd_lengths, vfwd_ids, vfwd_ms, vbwd_ids, vbwd_ms, mmask, emask, binary_mask, txt, ntokens = batch
    else:
        input_ids, pos_ids, input_mask, fwd_ids, fwd_ms, lengths, bwd_ids, bwd_ms_c, bwd_lengths, mmask, emask, binary_mask, txt, ntokens = batch
    # code.interact(local=locals())
    batch_size, N, M, L = fwd_ids.size()
    B = args.max_blocks

    # do some prep
    bwd_ids = (bwd_ids.unsqueeze(2) * emask.to(torch.long) + tokenizer.pad_index * (1-emask.to(torch.long))).reshape(batch_size, B * L, M, L)
    bwd_ms = (bwd_ms_c.unsqueeze(2) * emask + mmask).reshape(batch_size, B * L, M, L)
    bwd_connector = (bwd_ids == tokenizer.pad_index).to(torch.float)
    bwd_connector[:, :, 1:, :] = 0
    global_mask = binary_mask.unsqueeze(1) * binary_mask.unsqueeze(2) * tokenizer.causal_mask(N, L, M, device=device).unsqueeze(0)

    if args.output_viterbi:
        output_fwd_ids = vfwd_ids
        output_fwd_ms = vfwd_ms
    else:
        output_fwd_ids = fwd_ids
        output_fwd_ms = fwd_ms
        output_bwd_ms_c = bwd_ms_c
        output_bwd_ms = bwd_ms

    output_fwd_ts = tokenizer.get_weights(fwd_ids)
    output_bwd_ts = None
    if args.debug_fixed_point:
        assert not args.output_viterbi
        prev = output_fwd_ts
        counter =0
        errors = []
        with torch.no_grad():
            # tqdm_bar = tqdm(forever_generator())
            for _ in forever_generator():
                ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                                         bwd_ids, bwd_ms, bwd_lengths,
                                         mmask, emask, None, lm=True, lm_mask=global_mask, fwd_ts=output_fwd_ts, bwd_ts=output_bwd_ts, marginal_temperature=args.marginal_temperature)
                # run model
                losses = model(input_ids=input_ids, position_ids=pos_ids, attn_bias=a if args.vopt else None, return_dict=True)

                if args.output_viterbi:
                    output_fwd_ids = vfwd_ids
                    output_fwd_ms = vfwd_ms
                    output_bwd_ms = vbwd_ms
                else:
                    output_fwd_ids = fwd_ids
                    output_fwd_ms = fwd_ms
                    output_bwd_ms = bwd_ms
                # get indices
                indices = increasing_roll_left(output_fwd_ids, tokenizer.pad_index).transpose(-1, -2).reshape(batch_size, N * L,M)  # batch x NL x M

                # get output probabilities
                log_edge_weights = torch.log_softmax(losses["logits"][:, -N * L:, :], -1)  # batch x NL x V

                # get log probs and convert back to transition matrix
                output_fwd_ts = torch.gather(log_edge_weights, -1, indices).reshape(batch_size, N, L, M).transpose(-1,
                                                                                                                   -2)  # batch x N x M x L
                bos_mask = torch.ones_like(output_fwd_ts)
                bos_mask[:, 0, :, 0] = 0  # first column of first block is bos
                ofts = output_fwd_ts.reshape(batch_size, -1)
                conditioning = ofts.max(-1)[0].detach()
                output_fwd_ts = bos_mask * output_fwd_ts + (1 - bos_mask) * conditioning[:, None, None, None]
                output_bwd_ts = torch.flip(output_fwd_ts, dims=[-1])
                output_bwd_ts = output_bwd_ts * output_bwd_ms_c + -INF * (1 - output_bwd_ms_c)
                output_bwd_ts = (output_bwd_ts.unsqueeze(2) * emask + -INF * (1 - emask)).reshape(batch_size, B * L, M, L)
                output_bwd_ts = output_bwd_ts * (1 - bwd_connector)
                output_fwd_ts = increasing_roll_right(output_fwd_ts, 0)
                output_fwd_ts = output_fwd_ts * output_fwd_ms + (1 - output_fwd_ms) * -INF
                diff = ((output_fwd_ts - prev)**2).sum().item()
                # tqdm_bar.desc = f"Diff = {diff}"
                errors.append(diff)
                prev = output_fwd_ts
                # print(counter)
                # code.interact(local=locals())
                if diff < 1e-3:
                    # print()
                    break
                counter += 1
                if counter > 100:
                    print(f"Error: did not converge in 100 iterations {errors}")
                    code.interact(local=locals())



    iters = 2 if args.debug_fixed_point else 1
    for iter in range(iters):
        # dp lattice if necessasry
        ent, a, m, c = None, None, None, None
        if args.vopt:
            ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                                     bwd_ids, bwd_ms, bwd_lengths,
                                     mmask, emask, None, lm=True, lm_mask=global_mask, fwd_ts=output_fwd_ts, bwd_ts=output_bwd_ts, marginal_temperature=args.marginal_temperature)
        # run model
        losses = model(input_ids=input_ids, position_ids=pos_ids, attn_bias=a if args.vopt else None, return_dict=True)

        # get indices
        indices = increasing_roll_left(output_fwd_ids, tokenizer.pad_index).transpose(-1, -2).reshape(batch_size, N * L, M) # batch x NL x M

        # get output probabilities
        log_edge_weights = torch.log_softmax(losses["logits"][:, -N * L:, :], -1) # batch x NL x V

        # get log probs and convert back to transition matrix
        output_fwd_ts = torch.gather(log_edge_weights, -1, indices).reshape(batch_size, N, L, M).transpose(-1,-2) # batch x N x M x L
        bos_mask = torch.ones_like(output_fwd_ts)
        bos_mask[:,0,:,0] = 0 # first column of first block is bos
        ofts = output_fwd_ts.reshape(batch_size, -1)
        conditioning = ofts.max(-1)[0].detach()
        output_fwd_ts = bos_mask * output_fwd_ts + (1-bos_mask) * conditioning[:,None, None, None]
        output_bwd_ts = torch.flip(output_fwd_ts, dims=[-1])
        output_bwd_ts = output_bwd_ts * output_bwd_ms_c + -INF * (1 - output_bwd_ms_c)
        output_bwd_ts = (output_bwd_ts.unsqueeze(2) * emask + -INF * (1-emask)).reshape(batch_size, B * L, M, L)
        output_bwd_ts = output_bwd_ts * (1-bwd_connector)
        output_fwd_ts = increasing_roll_right(output_fwd_ts, 0)
        output_fwd_ts = output_fwd_ts * output_fwd_ms + (1-output_fwd_ms) * -INF

    # local = locals()
    # def hook(grad):
    #     print("hook")
    #     nonlocal local
    #     # code.interact(local=dict(list(local.items()) + [("grad", grad)]))
    # losses["logits"].register_hook(hook)

    if args.log_lattice and eval:
        with open(os.path.join(args.output_dir, args.log_lattice_file), "at") as f:
            d = OrderedDict()
            d["key"] = args.log_lattice_key
            for field in args.log_lattice:
                value = LOGGERS[field](**locals())
                d[field] = value
            f.write(json.dumps(d) + "\n")
    # compute prob
    log_alphas, _= tokenizer.forward_algorithm(output_fwd_ts.reshape(batch_size * N, M, L), output_fwd_ms.reshape(batch_size * N, M, L), lengths.reshape(batch_size * N))


    # get loss
    nchars = lengths.sum() - batch_size # adjust for BOS
    log_probs = log_alphas.sum()
    if eval:
        loss = -log_probs / nchars  * args.main_loss_multiplier
    elif args.normalize_by_tokens:
        loss = -log_probs / (output_fwd_ms.sum() - batch_size)  * args.main_loss_multiplier
    elif args.normalize_by_expected_length:
        length_transition = (torch.ones(fwd_ids.size(-2)))[None,None, :, None].expand(*fwd_ids.size()).to(output_fwd_ms.device) * output_fwd_ms
        expected_lengths, _ = tokenizer.expectation(output_fwd_ts.reshape(batch_size * N, M, L), length_transition.reshape(batch_size * N, M, L), output_fwd_ms.reshape(batch_size * N, M, L), lengths.reshape(batch_size * N))
        loss = -log_probs / (expected_lengths.sum().item() - batch_size) * args.main_loss_multiplier
    elif args.no_normalization:
        loss = -log_probs * args.main_loss_multiplier / batch_size
    elif args.constant_normalization:
        loss = -log_probs * args.main_loss_multiplier / batch_size / args.constant_normalization
    else:
        loss = -log_probs / nchars  * args.main_loss_multiplier
    # if loss.isnan().any():
    #     print("nan loss")
    #     code.interact(local=locals())
    # if txt[0].startswith("some changes to the plan"):
    #     print("some changes to the plan")
    #     code.interact(local=locals())
    code.interact(local=locals())
    if loss.isnan().any():
        code.interact(local=locals())

    if DEBUG:
        code.interact(local=locals())
    return loss, ent, lengths, ntokens

def language_modeling_unigram_step(args, batch, tokenizer, model, device):
    batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
    input_ids, pos_ids, input_mask, labels, lengths, ntokens, text = batch

    # run model
    if args.debug_node_unigram:
        tril = torch.tril(torch.ones((input_ids.size(-1)//2, input_ids.size(-1)//2), dtype=torch.float, device=device))
        eye = torch.eye(input_ids.size(-1)//2, dtype=torch.float, device=device)
        left = torch.cat([tril, tril], dim=0)
        right = torch.cat([eye * 0, eye], dim=0)
        causal_mask = torch.cat([left, right], dim=1)
        causal_mask = causal_mask[None, ...].expand(input_ids.size(0), input_ids.size(1), input_ids.size(1))
    else:
        causal_mask = torch.tril(torch.ones((input_ids.size(-1), input_ids.size(-1)), dtype=torch.float, device=device))[None, ...].expand(input_ids.size(0), input_ids.size(1), input_ids.size(1))
    attn_bias = causal_mask * 0 + (1-causal_mask) * -INF
    losses = model(input_ids=input_ids, position_ids=pos_ids, attention_mask=input_mask, labels=labels, attn_bias=attn_bias, return_dict=True)
    # get loss
    loss = losses[0] * args.main_loss_multiplier
    return loss, None, lengths, ntokens # sum of mask is the number of tokens
