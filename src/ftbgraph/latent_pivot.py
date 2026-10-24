from collections import Counter, defaultdict

import numpy as np
import torch

from .data import detect_lang


def logit_lens_trace(model, prompt, langs):
    hs = model.hidden_states(prompt)
    trace = []
    for layer_idx, h in enumerate(hs):
        vec = h[0, -1, :]
        logits = model.unembed(vec)
        top_id = int(torch.argmax(logits).item())
        token_str = model.tok.decode([top_id])
        lang = detect_lang(token_str, langs)
        trace.append({"layer": layer_idx, "top_token": token_str, "detected_lang": lang})
    return trace


def pivot_profile(model, samples, langs, pivot_lang="en"):
    per_sample_traces = []
    exit_layers = []
    pivot_dominant_layers = defaultdict(int)
    n_layers_plus_embed = model.n_layers + 1

    for s in samples:
        if s["target_lang"] == pivot_lang:
            continue
        trace = logit_lens_trace(model, s["prompt"], langs)
        per_sample_traces.append({"prompt": s["prompt"], "target_lang": s["target_lang"], "trace": trace})

        exit_layer = None
        for step in trace:
            if step["detected_lang"] == s["target_lang"]:
                exit_layer = step["layer"]
                break
        exit_layers.append(exit_layer if exit_layer is not None else n_layers_plus_embed)

        for step in trace:
            if step["detected_lang"] == pivot_lang:
                pivot_dominant_layers[step["layer"]] += 1

    lang_counts_per_layer = []
    for layer_idx in range(n_layers_plus_embed):
        counts = Counter()
        for rec in per_sample_traces:
            lang = rec["trace"][layer_idx]["detected_lang"]
            counts[lang] += 1
        lang_counts_per_layer.append(dict(counts))

    return {
        "n_samples": len(per_sample_traces),
        "mean_target_exit_layer": float(np.mean(exit_layers)) if exit_layers else None,
        "median_target_exit_layer": float(np.median(exit_layers)) if exit_layers else None,
        "pivot_dominant_layer_counts": dict(pivot_dominant_layers),
        "lang_counts_per_layer": lang_counts_per_layer,
        "n_layers_plus_embed": n_layers_plus_embed,
    }


def correlate_pivot_with_circuit(pivot_result, in_graph_heads):
    if not in_graph_heads or pivot_result["mean_target_exit_layer"] is None:
        return {"n_in_graph_heads": len(in_graph_heads) if in_graph_heads else 0, "correlation": None}
    circuit_layers = [h.layer for h in in_graph_heads]
    mean_circuit_layer = float(np.mean(circuit_layers))
    exit_layer = pivot_result["mean_target_exit_layer"]
    frac_after_exit = sum(1 for l in circuit_layers if l >= exit_layer) / len(circuit_layers)
    return {
        "mean_circuit_layer": mean_circuit_layer,
        "mean_target_exit_layer": exit_layer,
        "fraction_circuit_heads_at_or_after_pivot_exit": frac_after_exit,
        "circuit_sits_downstream_of_pivot": bool(frac_after_exit > 0.5 and mean_circuit_layer >= exit_layer),
    }
