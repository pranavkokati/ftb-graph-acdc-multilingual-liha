import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .arch import Head


def build_eap_graph(ranked_edges, topk):
    G = nx.DiGraph()
    top = ranked_edges[:topk]
    for (src, dst_layer), score in top:
        G.add_edge(repr(src), f"L{dst_layer}*", weight=score)
    return G, top


def build_verified_graph(verify_summary, keep_threshold=1e-5, require_fdr=True):
    G = nx.DiGraph()
    for (src, dst), v in verify_summary.items():
        exact_delta, eap_score = v["mean_exact_delta"], v["eap_score"]
        if np.isnan(exact_delta) or np.isnan(eap_score):
            continue
        if abs(exact_delta) < keep_threshold:
            continue
        if (exact_delta * eap_score) <= 0:
            continue
        if require_fdr and not v.get("fdr_significant", False):
            continue
        G.add_edge(repr(src), repr(dst), weight=exact_delta, eap_score=eap_score, p_value=v.get("p_value", 1.0))
    return G


def graph_topology_report(G):
    if G.number_of_edges() == 0:
        return {"n_nodes": 0, "n_edges": 0, "betweenness": {}, "sources": [], "sinks": []}
    betweenness = nx.betweenness_centrality(G)
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "betweenness": betweenness, "sources": sources, "sinks": sinks,
    }


def in_graph_heads(G):
    heads = set()
    for node in G.nodes:
        if "H" not in node:
            continue
        try:
            layer, head = node.replace("L", "").split("H")
            heads.add(Head(int(layer), int(head)))
        except Exception:
            continue
    return heads


def plot_dag(G, title, save_path):
    if G.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    weights = [abs(G[u][v].get("weight", 1.0)) for u, v in G.edges]
    max_w = max(weights) if weights and max(weights) > 0 else 1.0
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[max(1.0, 3 * w / max_w) for w in weights], alpha=0.6, ax=ax,
                            arrowstyle="-|>", arrowsize=15)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
