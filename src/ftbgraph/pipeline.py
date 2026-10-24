import json
from pathlib import Path

from . import config
from .arch import GradModel
from .data import audit_all_lang_tokens, select_balanced_samples
from .dataset_gen import build_suite, confound_report
from .attribution import run_eap
from .attribution_ig import run_eap_ig
from .graph import build_eap_graph, build_verified_graph, graph_topology_report, in_graph_heads, plot_dag
from .verify import edges_to_verify, verify_edges, summarize_verification
from .causal_scrubbing import resample_validate_circuit
from .steering import collect_layer_means, steering_sufficiency_test
from .latent_pivot import pivot_profile, correlate_pivot_with_circuit


def _save(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: list(o) if isinstance(o, set) else str(o))


def run_novel_pipeline(model_name, model_type, load_path=None, load_in_4bit=False,
                        eap_n=24, verify_topk=40, verify_dst_cap=4, verify_n=20,
                        ig_steps=6, valid_n=24, pivot_n=20, steer_alphas=(2.0, 4.0, 8.0),
                        require_fdr=True, out_dir=None):
    out_dir = Path(out_dir) if out_dir else config.NOVEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.replace("/", "_")

    model = GradModel(model_name, load_path=load_path, model_type=model_type, load_in_4bit=load_in_4bit)
    lang_token_cache = audit_all_lang_tokens(model.tok, config.LANGS)
    usable_langs = list(lang_token_cache)

    suite = build_suite(seed=config.SEED)
    _save({"usable_langs": usable_langs, "confound": confound_report(model.tok, lang_token_cache)},
          out_dir / f"{tag}_dataset_audit.json")

    eap_samples = select_balanced_samples(suite, "neutral", eap_n, langs=usable_langs, seed=config.SEED)
    plain_ranked = run_eap(model, eap_samples, lang_token_cache)
    ig_ranked = run_eap_ig(model, eap_samples, lang_token_cache, m=ig_steps)

    def _edges_to_dicts(ranked):
        return [{"src": repr(src), "dst_layer": dst_layer, "score": score} for (src, dst_layer), score in ranked]

    _save({"plain_eap_top": _edges_to_dicts(plain_ranked[:50]), "eap_ig_top": _edges_to_dicts(ig_ranked[:50])},
          out_dir / f"{tag}_eap_comparison.json")

    _, top_edges = build_eap_graph(ig_ranked, topk=verify_topk)
    candidates = edges_to_verify(top_edges, model, k=verify_topk, dst_cap=verify_dst_cap)
    verify_samples = select_balanced_samples(suite, "neutral", verify_n, langs=usable_langs, seed=config.SEED + 2)
    per_sample = verify_edges(model, candidates, verify_samples, lang_token_cache)
    summary = summarize_verification(per_sample)

    G = build_verified_graph(summary, require_fdr=require_fdr)
    if G.number_of_edges() == 0 and require_fdr:
        G = build_verified_graph(summary, require_fdr=False)
    topo = graph_topology_report(G)
    plot_dag(G, f"{model_name}: EAP-IG verified circuit (FDR-controlled)", out_dir / f"{tag}_dag_novel.png")
    circuit_heads = in_graph_heads(G)
    _save({"n_nodes": topo["n_nodes"], "n_edges": topo["n_edges"], "heads": [repr(h) for h in circuit_heads]},
          out_dir / f"{tag}_graph_checkpoint.json")

    valid_samples = select_balanced_samples(suite, "neutral", valid_n, langs=usable_langs, seed=config.SEED + 3)
    scrub = resample_validate_circuit(model, circuit_heads, valid_samples, lang_token_cache)

    pivot_samples = select_balanced_samples(suite, "neutral", pivot_n, langs=usable_langs, seed=config.SEED + 4)
    pivot = pivot_profile(model, pivot_samples, usable_langs)
    pivot_corr = correlate_pivot_with_circuit(pivot, circuit_heads)

    steer_result = None
    if circuit_heads:
        steer_layer = max(h.layer for h in circuit_heads)
        layer_means = collect_layer_means(model, eap_samples, steer_layer)
        steer_result = steering_sufficiency_test(model, layer_means, valid_samples, lang_token_cache,
                                                   steer_layer, list(steer_alphas), usable_langs)

    result = {
        "model": model_name,
        "usable_langs": usable_langs,
        "verified_graph": {"n_nodes": topo["n_nodes"], "n_edges": topo["n_edges"],
                            "sources": topo["sources"], "sinks": topo["sinks"],
                            "heads": [repr(h) for h in circuit_heads]},
        "causal_scrubbing_validation": scrub,
        "latent_pivot_profile": {k: v for k, v in pivot.items() if k != "lang_counts_per_layer"},
        "pivot_circuit_correlation": pivot_corr,
        "steering_sufficiency": steer_result,
    }
    _save(result, out_dir / f"{tag}_novel_results.json")
    _save(pivot, out_dir / f"{tag}_pivot_full.json")
    model.free()
    return result
