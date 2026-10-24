from collections import defaultdict

import numpy as np
import torch

from . import config
from .arch import Head
from .data import metric_logprob_diff
from .stats import bootstrap_ci, benjamini_hochberg


def edges_to_verify(top_edges, model, k=config.VERIFY_TOPK, dst_cap=config.VERIFY_DST_CAP, seed=config.SEED):
    import random
    out = []
    rng = random.Random(seed + 10)
    for (src, dst_layer), score in top_edges[:k]:
        all_dst_heads = list(range(model.n_heads))
        sampled = rng.sample(all_dst_heads, dst_cap) if len(all_dst_heads) > dst_cap else all_dst_heads
        for dst_head in sorted(sampled):
            out.append((src, Head(dst_layer, dst_head), score))
    return out


def patch_and_score(model, prompt, src_head, src_value, target_ids, foil_ids):
    def hook_fn(module, inp):
        xh_all = model.head_slice_of(src_head.layer, inp[0])
        xh_patched = xh_all.clone()
        safe_src = torch.clamp(src_value, min=-10000.0, max=10000.0)
        xh_patched[:, -1, src_head.head, :] = safe_src
        return (xh_patched.reshape(inp[0].shape),) + tuple(inp[1:])
    handle = model._proj(src_head.layer).register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            logits = model.first_token_logits(prompt)
    finally:
        handle.remove()
    return metric_logprob_diff(logits, target_ids, foil_ids).item()


def verify_edges(model, candidate_edges, samples, lang_token_cache):
    per_sample_deltas = defaultdict(list)
    corr_metrics = {}
    for i, s in enumerate(samples):
        target_ids, foil_ids = lang_token_cache[s["target_lang"]], lang_token_cache[s["foil_lang"]]
        with torch.no_grad():
            corr_logits = model.first_token_logits(s["corrupted"])
        corr_metrics[i] = float(metric_logprob_diff(corr_logits, target_ids, foil_ids).to(torch.float32).item())

    edges_by_src = defaultdict(list)
    for src, dst, eap_score in candidate_edges:
        edges_by_src[src].append((dst, eap_score))

    for src, dst_list in edges_by_src.items():
        clean_src_values = {}
        for i, s in enumerate(samples):
            store = {}
            def cap_hook(module, inp, _store=store, _layer=src.layer):
                _store["x"] = model.head_slice_of(_layer, inp[0])[:, -1, src.head, :].detach().clone()
                return inp
            handle = model._proj(src.layer).register_forward_pre_hook(cap_hook)
            with torch.no_grad():
                _ = model.first_token_logits(s["prompt"])
            handle.remove()
            clean_src_values[i] = store["x"]

        for dst, eap_score in dst_list:
            deltas = []
            for i, s in enumerate(samples):
                target_ids, foil_ids = lang_token_cache[s["target_lang"]], lang_token_cache[s["foil_lang"]]
                patched_metric = patch_and_score(model, s["corrupted"], src, clean_src_values[i], target_ids, foil_ids)
                if np.isnan(patched_metric) or np.isnan(corr_metrics[i]):
                    deltas.append(0.0)
                else:
                    deltas.append(float(patched_metric) - corr_metrics[i])
            per_sample_deltas[(src, dst)] = {"eap_score": float(eap_score), "deltas": deltas}
    return per_sample_deltas


def summarize_verification(per_sample_deltas, fdr_q=config.FDR_Q):
    keys = list(per_sample_deltas)
    results = {}
    pvalues = []
    for key in keys:
        deltas = per_sample_deltas[key]["deltas"]
        ci = bootstrap_ci(deltas)
        n = len(deltas)
        mean, std = ci["mean"], float(np.std(deltas)) if n else 0.0
        se = std / np.sqrt(n) if n > 1 else 1e-9
        z = mean / se if se > 0 else 0.0
        p = 2 * (1 - _normal_cdf(abs(z)))
        pvalues.append(p)
        results[key] = {
            "eap_score": per_sample_deltas[key]["eap_score"],
            "mean_exact_delta": mean, "std_exact_delta": std,
            "ci_lo": ci["lo"], "ci_hi": ci["hi"], "p_value": p,
        }
    reject = benjamini_hochberg(np.array(pvalues), q=fdr_q) if pvalues else np.array([], dtype=bool)
    for key, r in zip(keys, reject):
        results[key]["fdr_significant"] = bool(r)
    return results


def _normal_cdf(x):
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
