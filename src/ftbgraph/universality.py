from itertools import combinations

import numpy as np
import torch

from .data import metric_logprob_diff


def head_language_signature(model, head, samples, lang_token_cache, langs):
    per_lang_drop = {}
    for lang in langs:
        lang_samples = [s for s in samples if s["target_lang"] == lang]
        if not lang_samples:
            continue
        drops = []
        for s in lang_samples:
            target_ids, foil_ids = lang_token_cache.get(lang), lang_token_cache.get(s["foil_lang"])
            if not target_ids or not foil_ids:
                continue
            with torch.no_grad():
                clean_logits = model.first_token_logits(s["prompt"])
            clean_metric = metric_logprob_diff(clean_logits, target_ids, foil_ids).to(torch.float32).item()
            with model.patch_heads_ctx({head: torch.zeros(model.head_dim)}):
                with torch.no_grad():
                    ablated_logits = model.first_token_logits(s["prompt"])
            ablated_metric = metric_logprob_diff(ablated_logits, target_ids, foil_ids).to(torch.float32).item()
            if not (np.isnan(clean_metric) or np.isnan(ablated_metric)):
                drops.append(clean_metric - ablated_metric)
        if drops:
            per_lang_drop[lang] = float(np.mean(drops))
    return per_lang_drop


def build_signature_matrix(model, heads, samples, lang_token_cache, langs):
    rows = {}
    for head in heads:
        sig = head_language_signature(model, head, samples, lang_token_cache, langs)
        if sig:
            rows[head] = sig
    return rows


def _vectorize(sig, langs):
    vec = np.array([sig.get(l, 0.0) for l in langs], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_best_matches(sig_a, sig_b, langs):
    heads_a, heads_b = list(sig_a), list(sig_b)
    if not heads_a or not heads_b:
        return []
    mat_a = np.stack([_vectorize(sig_a[h], langs) for h in heads_a])
    mat_b = np.stack([_vectorize(sig_b[h], langs) for h in heads_b])
    sims = mat_a @ mat_b.T
    matches = []
    for i, head_a in enumerate(heads_a):
        j = int(np.argmax(sims[i]))
        matches.append({"head_a": repr(head_a), "head_b": repr(heads_b[j]), "cosine_sim": float(sims[i, j])})
    return matches


def universality_null(sig_a, sig_b, langs, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    observed = cosine_best_matches(sig_a, sig_b, langs)
    if not observed:
        return {"observed_mean_sim": None, "p_value": None}
    observed_mean = float(np.mean([m["cosine_sim"] for m in observed]))

    heads_b = list(sig_b)
    vecs_b = {h: sig_b[h] for h in heads_b}
    count = 0
    for _ in range(n_perm):
        shuffled_keys = heads_b.copy()
        rng.shuffle(shuffled_keys)
        shuffled_sig_b = {orig: vecs_b[shuf] for orig, shuf in zip(heads_b, shuffled_keys)}
        null_matches = cosine_best_matches(sig_a, shuffled_sig_b, langs)
        null_mean = float(np.mean([m["cosine_sim"] for m in null_matches])) if null_matches else 0.0
        if null_mean >= observed_mean:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {"observed_mean_sim": observed_mean, "p_value": p_value, "n_perm": n_perm, "best_matches": observed}


def cross_model_universality_report(model_signatures, langs, n_perm=500):
    model_names = list(model_signatures)
    report = []
    for name_a, name_b in combinations(model_names, 2):
        result = universality_null(model_signatures[name_a], model_signatures[name_b], langs, n_perm=n_perm)
        report.append({"model_a": name_a, "model_b": name_b, **result})
    return report
