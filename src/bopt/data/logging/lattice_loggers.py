

def log_prob(output_fwd_ts=None, **kwargs):
    return output_fwd_ts.tolist()

def unit(fwd_ids=None, tokenizer=None, **kwargs):
    units = fwd_ids.tolist()
    for i in range(fwd_ids.size(0)):
        for j in range(fwd_ids.size(1)):
            for k in range(fwd_ids.size(2)):
                for l in range(fwd_ids.size(3)):
                    units[i][j][k][l] = tokenizer.id2str(fwd_ids[i,j,k,l], remove_csp=True)
    return units

def marginal(m=None, **kwargs):
    return m.tolist()

def ent(ent=None, **kwargs):
    return ent.tolist()

def lm_marginal(tokenizer=None, fwd_ids=None, fwd_ms=None, lengths=None, bwd_ids=None, bwd_ms=None, bwd_lengths=None, mmask=None, emask=None, global_mask=None, output_fwd_ts=None, output_bwd_ts=None, **kwargs):
    ent, a, m, c = tokenizer(fwd_ids, fwd_ms, lengths,
                             bwd_ids, bwd_ms, bwd_lengths,
                             mmask, emask, None, lm=True, lm_mask=global_mask, fwd_ts=output_fwd_ts,
                             bwd_ts=output_bwd_ts)
    return m.tolist()

def marginal_count(om_list=None, **kwargs):
    return om_list.tolist() if om_list is not None else []

def token(unit_list=None, **kwargs):
    return unit_list.tolist() if unit_list is not None else []

LOGGERS = {
    "log_prob": log_prob,
    "unit": unit,
    "marginal": marginal,
    "ent": ent,
    "lm_marginal": lm_marginal,
    "marginal_count": marginal_count,
    "token": token
}

