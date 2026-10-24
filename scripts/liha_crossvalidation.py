import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ftbgraph import config, dataset_gen, data
from ftbgraph.arch import GradModel, Head
from ftbgraph.attribution import run_node_attribution
from ftbgraph.verify import edges_to_verify, verify_edges, summarize_verification
from ftbgraph.causal_scrubbing import resample_ablate_heads
from ftbgraph.data import metric_logprob_diff
from ftbgraph.stats import bootstrap_ci

PUBLISHED_LIHA_TOP5 = {
    "L6H1": {"switch_rate": 0.32, "sigma_above_mean": 3.23},
    "L0H4": {"switch_rate": 0.28, "sigma_above_mean": None},
    "L3H1": {"switch_rate": 0.28, "sigma_above_mean": None},
    "L9H9": {"switch_rate": 0.28, "sigma_above_mean": None},
    "L7H3": {"switch_rate": 0.24, "sigma_above_mean": None},
}
PREDICTED_COMPENSATION_LAYERS = [7, 8, 9]


def parse_head(label):
    layer_str, head_str = label[1:].split("H")
    return Head(int(layer_str), int(head_str))


def layer_mean_abs_scores(ranked):
    by_layer = {}
    for head, score in ranked:
        by_layer.setdefault(head.layer, []).append(abs(score))
    return {l: float(np.mean(v)) for l, v in by_layer.items()}


def main():
    model = GradModel("gpt2", model_type="gpt2")
    suite = dataset_gen.build_suite(seed=config.SEED)
    cache = data.audit_all_lang_tokens(model.tok, config.LANGS)
    usable = list(cache)

    eap_comparison_path = config.NOVEL_DIR / "gpt2_eap_comparison.json"
    eap_comparison = json.load(open(eap_comparison_path))
    ig_ranked = [
        ((parse_head(e["src"]), e["dst_layer"]), e["score"])
        for e in eap_comparison["eap_ig_top"]
    ]

    verify_samples = data.select_balanced_samples(suite, "neutral", 20, langs=usable, seed=config.SEED + 2)
    candidates = edges_to_verify(ig_ranked, model, k=50, dst_cap=4)
    per_sample = verify_edges(model, candidates, verify_samples, cache)
    summary = summarize_verification(per_sample)

    eap_scores = np.array([v["eap_score"] for v in summary.values()])
    exact_deltas = np.array([v["mean_exact_delta"] for v in summary.values()])
    eap_exact_correlation = float(np.corrcoef(eap_scores, exact_deltas)[0, 1]) if len(eap_scores) > 1 else None

    valid_samples = data.select_balanced_samples(suite, "neutral", 24, langs=usable, seed=config.SEED + 3)

    published_head_results = {}
    for label, published in PUBLISHED_LIHA_TOP5.items():
        head = parse_head(label)
        drops = []
        for s in valid_samples:
            target_ids, foil_ids = cache.get(s["target_lang"]), cache.get(s["foil_lang"])
            if not target_ids or not foil_ids:
                continue
            with torch.no_grad():
                clean_logits = model.first_token_logits(s["prompt"])
            clean_metric = metric_logprob_diff(clean_logits, target_ids, foil_ids).to(torch.float32).item()
            donor_pool = [s2["prompt"] for s2 in valid_samples if s2["target_lang"] != s["target_lang"]]
            if not donor_pool or np.isnan(clean_metric):
                continue
            ablated_logits = resample_ablate_heads(model, s["prompt"], [head], donor_pool)
            ablated_metric = metric_logprob_diff(ablated_logits, target_ids, foil_ids).to(torch.float32).item()
            if not np.isnan(ablated_metric):
                drops.append(clean_metric - ablated_metric)
        published_head_results[label] = {
            "published_liha": published,
            "resample_ablation_metric_drop": bootstrap_ci(drops),
            "in_eap_ig_verified_circuit": label in json.load(open(config.NOVEL_DIR / "gpt2_novel_results.json"))["verified_graph"]["heads"],
        }

    baseline_ranked = run_node_attribution(model, valid_samples, cache)
    baseline_by_layer = layer_mean_abs_scores(baseline_ranked)

    with model.patch_heads_ctx({Head(6, 1): torch.zeros(model.head_dim, dtype=config.DTYPE)}):
        ablated_ranked = run_node_attribution(model, valid_samples, cache)
    ablated_by_layer = layer_mean_abs_scores(ablated_ranked)

    delta_by_layer = {
        l: ablated_by_layer.get(l, 0.0) - baseline_by_layer.get(l, 0.0)
        for l in set(baseline_by_layer) | set(ablated_by_layer)
    }
    predicted_layers_mean_delta = float(np.mean([delta_by_layer.get(l, 0.0) for l in PREDICTED_COMPENSATION_LAYERS]))
    other_layers = [l for l in delta_by_layer if l not in PREDICTED_COMPENSATION_LAYERS]
    other_layers_mean_delta = float(np.mean([delta_by_layer.get(l, 0.0) for l in other_layers])) if other_layers else 0.0

    result = {
        "eap_vs_exact_patching_correlation": {
            "ours_eap_ig_plus_exact": eap_exact_correlation,
            "n_edges": int(len(eap_scores)),
            "published_ftb_graph_plain_eap": -0.226,
        },
        "published_liha_top5_head_comparison": published_head_results,
        "l6h1_ablation_compensatory_redistribution_test": {
            "baseline_mean_abs_attribution_by_layer": baseline_by_layer,
            "ablated_mean_abs_attribution_by_layer": ablated_by_layer,
            "delta_by_layer": delta_by_layer,
            "predicted_layers": PREDICTED_COMPENSATION_LAYERS,
            "predicted_layers_mean_delta": predicted_layers_mean_delta,
            "other_layers_mean_delta": other_layers_mean_delta,
            "redistribution_concentrated_in_predicted_layers": bool(predicted_layers_mean_delta > other_layers_mean_delta),
        },
    }

    out_path = config.NOVEL_DIR / "gpt2_liha_crossvalidation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    model.free()


if __name__ == "__main__":
    main()
