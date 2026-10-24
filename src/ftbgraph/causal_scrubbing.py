import random
from collections import defaultdict

import numpy as np
import torch

from . import config
from .data import metric_logprob_diff
from .stats import bootstrap_ci, permutation_test


def _donor_pool(samples):
    by_lang = defaultdict(list)
    for i, s in enumerate(samples):
        by_lang[s["target_lang"]].append(i)
    return by_lang


def resample_ablate_heads(model, prompt, heads, donor_prompts, k=config.RESAMPLE_K, seed=config.SEED):
    heads = list(heads)
    if not heads or not donor_prompts:
        with torch.no_grad():
            return model.first_token_logits(prompt)
    rng = random.Random(seed)
    donor_batch = [rng.choice(donor_prompts) for _ in range(k)]

    donor_values = defaultdict(list)
    for donor in donor_batch:
        pre = model.capture_preproj_heads(donor)
        for h in heads:
            if h in pre:
                donor_values[h].append(pre[h])

    patch_map = {}
    for h in heads:
        if donor_values[h]:
            patch_map[h] = torch.stack(donor_values[h]).mean(dim=0)

    with model.patch_heads_ctx(patch_map):
        with torch.no_grad():
            return model.first_token_logits(prompt)


def resample_validate_circuit(model, in_graph_heads, samples, lang_token_cache, seed=config.SEED):
    all_heads = set(model.all_heads())
    in_graph = set(in_graph_heads)
    out_graph = all_heads - in_graph
    by_lang = _donor_pool(samples)

    out_sample = random.Random(seed).sample(sorted(out_graph), min(len(in_graph), len(out_graph))) if in_graph else []

    baseline, in_drop, out_drop = [], [], []
    for s in samples:
        target_ids, foil_ids = lang_token_cache[s["target_lang"]], lang_token_cache[s["foil_lang"]]
        donor_pool = [samples[j]["prompt"] for lang, idxs in by_lang.items() if lang != s["target_lang"] for j in idxs]
        if not donor_pool:
            continue

        with torch.no_grad():
            clean_logits = model.first_token_logits(s["prompt"])
        clean_metric = metric_logprob_diff(clean_logits, target_ids, foil_ids).to(torch.float32).item()
        if np.isnan(clean_metric):
            continue
        baseline.append(clean_metric)

        in_logits = resample_ablate_heads(model, s["prompt"], in_graph, donor_pool, seed=seed)
        in_metric = metric_logprob_diff(in_logits, target_ids, foil_ids).to(torch.float32).item()
        in_drop.append(0.0 if np.isnan(in_metric) else clean_metric - in_metric)

        out_logits = resample_ablate_heads(model, s["prompt"], out_sample, donor_pool, seed=seed + 1)
        out_metric = metric_logprob_diff(out_logits, target_ids, foil_ids).to(torch.float32).item()
        out_drop.append(0.0 if np.isnan(out_metric) else clean_metric - out_metric)

    if not baseline:
        return {"error": "no_valid_samples"}

    in_ci = bootstrap_ci(in_drop)
    out_ci = bootstrap_ci(out_drop)
    perm = permutation_test(in_drop, out_drop)

    return {
        "model": model.name,
        "n_in_graph_heads": len(in_graph),
        "n_out_graph_heads_sampled": len(out_sample) if in_graph else 0,
        "baseline_metric_mean": float(np.mean(baseline)),
        "in_graph_resample_drop": in_ci,
        "out_of_graph_resample_drop": out_ci,
        "permutation_test": perm,
        "necessity_supported": bool(in_ci["mean"] > out_ci["mean"] and perm["p_value"] < 0.05),
    }
