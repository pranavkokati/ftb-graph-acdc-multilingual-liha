from collections import defaultdict

import numpy as np
import torch

from .data import metric_logprob_diff, detect_lang


def collect_layer_means(model, samples, layer):
    by_lang = defaultdict(list)
    for s in samples:
        hs = model.hidden_states(s["prompt"])
        vec = hs[layer + 1][0, -1, :].to(torch.float32)
        by_lang[s["target_lang"]].append(vec)
    return {lang: torch.stack(vecs).mean(dim=0) for lang, vecs in by_lang.items()}


def diff_in_means_vector(layer_means, lang_a, lang_b):
    return layer_means[lang_a] - layer_means[lang_b]


def steer_and_score(model, prompt, layer, vector, alpha, target_ids, foil_ids):
    with model.add_residual_ctx(layer, alpha * vector):
        with torch.no_grad():
            logits = model.first_token_logits(prompt)
    metric = metric_logprob_diff(logits, target_ids, foil_ids).to(torch.float32).item()
    top_id = int(torch.argmax(logits[0]).item())
    top_token = model.tok.decode([top_id])
    return metric, top_token


def steering_sufficiency_test(model, layer_means, holdout_samples, lang_token_cache, layer, alphas, fallback_langs):
    results = []
    lang_pairs = [(a, b) for a in layer_means for b in layer_means if a != b]
    for lang_a, lang_b in lang_pairs:
        vector = diff_in_means_vector(layer_means, lang_a, lang_b)
        targets = [s for s in holdout_samples if s["target_lang"] == lang_b]
        if not targets:
            continue
        for alpha in alphas:
            flips, deltas = 0, []
            for s in targets:
                target_ids = lang_token_cache.get(lang_a)
                foil_ids = lang_token_cache.get(lang_b)
                if not target_ids or not foil_ids:
                    continue
                with torch.no_grad():
                    base_logits = model.first_token_logits(s["prompt"])
                base_metric = metric_logprob_diff(base_logits, target_ids, foil_ids).to(torch.float32).item()
                steered_metric, top_token = steer_and_score(model, s["prompt"], layer, vector, alpha, target_ids, foil_ids)
                deltas.append(steered_metric - base_metric)
                decoded_lang = detect_lang(top_token, fallback_langs)
                if decoded_lang == lang_a:
                    flips += 1
            if deltas:
                results.append({
                    "from_lang": lang_b, "to_lang": lang_a, "alpha": alpha,
                    "mean_metric_shift": float(np.mean(deltas)),
                    "n_samples": len(deltas),
                    "flip_rate": flips / len(deltas),
                })
    return results
