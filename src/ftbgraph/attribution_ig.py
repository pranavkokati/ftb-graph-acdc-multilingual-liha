import gc
from collections import defaultdict

import numpy as np
import torch

from . import config
from .data import metric_logprob_diff


def _interp_patch_map(clean_pre, corr_pre, t):
    out = {}
    for head in clean_pre:
        if head not in corr_pre:
            continue
        c = clean_pre[head].to(torch.float32)
        k = corr_pre[head].to(torch.float32)
        out[head] = (k + t * (c - k))
    return out


def eap_ig_scores_single_pair(model, clean_prompt, corr_prompt, target_ids, foil_ids, m=config.IG_STEPS):
    clean_head_store, corr_head_store = {}, {}
    with torch.no_grad():
        clean_logits0 = model.first_token_logits(clean_prompt, store=clean_head_store, requires_grad=False, retain_head_grad=False)
        _ = model.first_token_logits(corr_prompt, store=corr_head_store, requires_grad=False)
    L0 = metric_logprob_diff(clean_logits0, target_ids, foil_ids)
    if torch.isnan(L0):
        return {}

    clean_pre = model.capture_preproj_heads(clean_prompt)
    corr_pre = model.capture_preproj_heads(corr_prompt)

    edge_grad_sum = defaultdict(float)
    for step in range(1, m + 1):
        t = step / m
        patch_map = _interp_patch_map(clean_pre, corr_pre, t)
        layer_grad_store = {}
        with model.capture_block_inputs(layer_grad_store), model.patch_heads_ctx(patch_map):
            inp = model._encode(clean_prompt)
            out = model.model(**inp)
            logits = out.logits[:, -1, :]
            L = metric_logprob_diff(logits, target_ids, foil_ids).to(torch.float32)
            if torch.isnan(L):
                continue
            model.model.zero_grad(set_to_none=True)
            L.backward()
        for dst_layer, x in layer_grad_store.items():
            if x.grad is None or torch.isnan(x.grad).any():
                continue
            grad_at_dst = x.grad[:, -1, :].to(torch.float32)
            for src_head in patch_map:
                if src_head.layer >= dst_layer:
                    continue
                key = (src_head, dst_layer)
                edge_grad_sum[key] = edge_grad_sum.get(key, 0.0) + grad_at_dst.detach().cpu()

    edge_scores = {}
    for src_head, clean_contrib in clean_head_store.items():
        corr_contrib = corr_head_store.get(src_head)
        if corr_contrib is None:
            continue
        clean_val = clean_contrib.to(torch.float32)
        corr_val = corr_contrib.to(torch.float32)
        delta = corr_val - clean_val
        for dst_layer in range(src_head.layer + 1, model.n_layers):
            key = (src_head, dst_layer)
            grad_sum = edge_grad_sum.get(key)
            if grad_sum is None:
                continue
            avg_grad = grad_sum / m
            score = float((delta * avg_grad).sum().item())
            if not np.isnan(score):
                edge_scores[key] = score
    return edge_scores


def run_eap_ig(model, samples, lang_token_cache, m=config.IG_STEPS):
    agg = defaultdict(list)
    for i, s in enumerate(samples):
        target_ids = lang_token_cache[s["target_lang"]]
        foil_ids = lang_token_cache[s["foil_lang"]]
        edge_scores = eap_ig_scores_single_pair(model, s["prompt"], s["corrupted"], target_ids, foil_ids, m=m)
        for k, v in edge_scores.items():
            if not np.isnan(v):
                agg[k].append(v)
        if (i + 1) % 5 == 0:
            gc.collect()
    mean_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    return sorted(mean_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
