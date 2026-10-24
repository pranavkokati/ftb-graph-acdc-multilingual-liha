import gc
from collections import defaultdict

import numpy as np
import torch

from . import config
from .data import metric_logprob_diff


def node_attribution_scores(model, clean_prompt, corr_prompt, target_ids, foil_ids):
    clean_store, corr_store = {}, {}
    clean_logits = model.first_token_logits(clean_prompt, store=clean_store, requires_grad=True)
    L = metric_logprob_diff(clean_logits, target_ids, foil_ids)
    if torch.isnan(L):
        return {}
    with torch.no_grad():
        _ = model.first_token_logits(corr_prompt, store=corr_store, requires_grad=False)
    model.model.zero_grad(set_to_none=True)
    L.backward()
    scores = {}
    for head, clean_contrib in clean_store.items():
        grad = clean_contrib.grad
        if grad is None or torch.isnan(grad).any():
            scores[head] = 0.0
            continue
        corr_contrib = corr_store.get(head)
        if corr_contrib is None:
            scores[head] = 0.0
            continue
        corr_val = corr_contrib.to(clean_contrib.device).to(torch.float32)
        clean_val = clean_contrib[:, -1, :].detach().to(torch.float32)
        delta = corr_val - clean_val
        g = grad[:, -1, :].to(torch.float32)
        product = delta * g
        scores[head] = 0.0 if torch.isnan(product).any() else float(product.sum().item())
    return scores


def run_node_attribution(model, samples, lang_token_cache):
    agg = defaultdict(list)
    for i, s in enumerate(samples):
        target_ids = lang_token_cache[s["target_lang"]]
        foil_ids = lang_token_cache[s["foil_lang"]]
        scores = node_attribution_scores(model, s["prompt"], s["corrupted"], target_ids, foil_ids)
        for head, v in scores.items():
            if not np.isnan(v):
                agg[head].append(v)
        if (i + 1) % 10 == 0:
            gc.collect()
    mean_scores = {h: float(np.mean(v)) if v else 0.0 for h, v in agg.items()}
    return sorted(mean_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)


def capture_layer_inputs(model, store):
    return model.capture_dst_inputs(store)


def eap_scores_single_pair(model, clean_prompt, corr_prompt, target_ids, foil_ids):
    clean_head_store, corr_head_store, clean_layer_grad_store = {}, {}, {}
    with capture_layer_inputs(model, clean_layer_grad_store):
        clean_logits = model.first_token_logits(clean_prompt, store=clean_head_store, requires_grad=True, retain_head_grad=False)
        L = metric_logprob_diff(clean_logits, target_ids, foil_ids).to(torch.float32)
        if torch.isnan(L):
            return {}
        model.model.zero_grad(set_to_none=True)
        L.backward()
    with torch.no_grad():
        _ = model.first_token_logits(corr_prompt, store=corr_head_store, requires_grad=False)
    edge_scores = {}
    for dst_layer, x in clean_layer_grad_store.items():
        if x.grad is None or torch.isnan(x.grad).any():
            continue
        grad_at_dst = x.grad[:, -1, :].to(torch.float32)
        for src_head, clean_contrib in clean_head_store.items():
            if src_head.layer >= dst_layer:
                continue
            corr_contrib = corr_head_store.get(src_head)
            if corr_contrib is None:
                continue
            corr_val = corr_contrib.to(clean_contrib.device).to(torch.float32)
            clean_val = clean_contrib[:, -1, :].to(torch.float32) if clean_contrib.dim() == 3 else clean_contrib.to(torch.float32)
            delta = (corr_val - clean_val).detach().to(grad_at_dst.device)
            product = delta * grad_at_dst
            if torch.isnan(product).any():
                continue
            score = float(product.sum().item())
            if not np.isnan(score):
                edge_scores[(src_head, dst_layer)] = edge_scores.get((src_head, dst_layer), 0.0) + score
    return edge_scores


def run_eap(model, samples, lang_token_cache):
    agg = defaultdict(list)
    for i, s in enumerate(samples):
        target_ids = lang_token_cache[s["target_lang"]]
        foil_ids = lang_token_cache[s["foil_lang"]]
        edge_scores = eap_scores_single_pair(model, s["prompt"], s["corrupted"], target_ids, foil_ids)
        for k, v in edge_scores.items():
            if not np.isnan(v):
                agg[k].append(v)
        if (i + 1) % 10 == 0:
            gc.collect()
    mean_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    return sorted(mean_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
