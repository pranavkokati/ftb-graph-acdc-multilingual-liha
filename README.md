# FTB Graph: Determining and Validating First-token Broadcasters and Language-Identity Head Circuits in Multilingual Language Models

Code and results for **FTB Graph: Determining and Validating First-token Broadcasters and Language-Identity Head Circuits in Multilingual Language Modelss**: identifying and validating the attention-head
circuits ("language identity heads") that determine which language a multilingual LM commits to
at its first generated token.

Pipeline: node-level attribution patching (fast head screen) → Edge Attribution Patching (EAP) →
graph construction → exact-activation-patching verification on the EAP shortlist → three-check
circuit validation (necessity / out-of-graph irrelevance / sufficiency) → DAG visualization.

## Repo layout

```
src/
├── dataset/
│   └── full_suite.json                                     # 29-item multilingual benchmark (see Dataset below)
├── eap/                                                    # {tag}_node_attr.json, {tag}_eap.json   (Stages 4–5 output, one pair per model)
├── ftb_graph_notebooks/
│   └── ftb-graph-notebook.ipynb                            # aggregated pipeline notebook covering both qwen pair and standalone models
├── verify/       # {tag}_verify.json                       (Stage 7: exact-patching verification)
├── validation/   # {tag}_validation.json                   (Stage 9: three-check validation)
└── graphs/       # {tag}_dag.png                           (Stage 8: verified circuit DAG)
```

`{tag}` = the HF model name with `/` replaced by `_`, e.g. `Qwen_Qwen2.5-1.5B`, `gpt2`,
`EleutherAI_pythia-2.8b`.

## Models covered

| Model | Parameters | Classification |
| :--- | :--- | :--- |
| `gpt2` | 124M | STANDALONE |
| `bigscience/bloom-560m` | 560M | STANDALONE |
| `EleutherAI/pythia-1b` | 1B | STANDALONE |
| `EleutherAI/pythia-2.8b` | 2.8B | STANDALONE |
| `Qwen/Qwen2.5-1.5B` | 1.5B | PAIR |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | PAIR |

Six architectures total. Four have the complete pipeline (standalone arm); the base-to-instruct
and scaling arms are partially complete (see Known Gaps).

## Dataset

`src/dataset/full_suite.json` — 29 hand-authored prompt pairs, built entirely in-notebook (no
external files):
- 21 neutral single-language prompts across 7 languages (en, fr, es, vi, zh, ru, ja)
- 6 code-switched prompts (English → native script mid-sentence)
- 2 semantic congruence probes (French-language prompt, French-topic vs. non-French-topic fact)

Every prompt is a fill-in-the-blank sentence ending right before the language-diagnostic word, so
the first generated token is the thing being scored. Only the `neutral` split is used by the
pipeline's sampling calls downstream (`select_balanced_samples(..., "neutral", ...)`); the
code-switch and semantic-congruence items are built but not yet consumed by `run_pipeline`.

This is a small, hand-authored seed set, not the 200–500-per-language-pair scale the original
project plan called for — noted here since it drives the wide standard deviations in the
validation results.

## Pipeline stages (per model)

1. **Setup** — installs, device check, architecture registry (gpt2 / bloom / qwen / pythia / aya).
2. **Node attribution** (`eap/{tag}_node_attr.json`) — one gradient pass per sample scores every
   attention head's contribution to the language-identity metric.
3. **Edge Attribution Patching** (`eap/{tag}_eap.json`) — scores every candidate
   `(src head → dst layer)` edge in one backward pass.
4. **Graph construction** — top-K EAP edges
5. **Exact-patching verification** (`verify/{tag}_verify.json`) — each shortlisted edge is
   re-checked with real activation patching (not gradient-based) to confirm sign and magnitude.
6. **Verified graph + DAG plot** (`graphs/{tag}_dag.png`) — keeps only edges EAP flagged *and*
   exact patching confirmed.
7. **Three-check validation** (`validation/{tag}_validation.json`) — ablates in-graph heads
   (necessity/completeness) vs. an equal-sized out-of-graph set (irrelevance) and reports the
   drop in the language-identity metric for each.

Output JSON schemas are documented inline in each notebook's corresponding stage; see the pipeline
cells in `ftb-graph-qwen-pair-a.ipynb` for the canonical field names (`completeness_drop_mean`,
`n_in_graph_heads`, etc.) — all notebooks in this repo share the same schema.

## Environment

`requirements.txt` is a full `pip freeze` of the Kaggle/Colab notebook environment (600+ packages),
not a curated dependency list — most of it (fastai, tensorflow, bigquery, etc.) is platform
boilerplate unrelated to this project. The packages that actually matter are: `torch`,
`transformers`, `accelerate`, `bitsandbytes`, `networkx`, `langdetect`, `langid`. Worth trimming
this down to a minimal `requirements.txt` before anyone tries to reproduce outside Kaggle.

Notebooks assume a Kaggle T4×2 GPU runtime and load models in fp16.

# Steps to Reproduce

## Clone and instantiate environment
```bash
git clone https://github.com/ajlrza/ftb-graph-acdc-multilingual-liha.git
cd src
python3 -m venv venv
source venv/bin/activate
```

## Install exact pinned versions
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configure the Path in Stage 0 accordingly
```bash
# In Stage 0 cell:
from pathlib import Path

# Set WORK_DIR to the 'src' root relative to the repository layout
WORK_DIR   = Path("../").resolve()  # or Path("./") if executing inside src/
DATA_DIR   = WORK_DIR / 'dataset'
EAP_DIR    = WORK_DIR / 'eap'
VERIFY_DIR = WORK_DIR / 'verify'
GRAPH_DIR  = WORK_DIR / 'graphs'
VALID_DIR  = WORK_DIR / 'validation'
```

## Run the pipeline
### 1. Via VS Code / Jupyter Notebook / Kaggle / Google Colab
```bash
Open src/ftb_graph_notebooks/ftb-graph-notebook.ipynb in VS Code or Jupyter Lab.

Select your Python kernel containing the installed requirements.txt.

Click Run All (or execute cells sequentially from Stage 0 to Stage 12).
```
### 2. Via CLI command
```bash
cd src/ftb_graph_notebooks

jupyter nbconvert --to notebook --execute ftb-graph-notebook.ipynb \
  --output executed_ftb_graph.ipynb \
  --ExecutePreprocessor.timeout=-1
```

## Verify the pipeline execution
Once the run completes (Stages 0–12), check that the subdirectories in src/ have populated with the output JSONs and PNGs:
* **`src/dataset/`**: Reads `full_suite.json` (the 29-item benchmark).
* **`src/eap/`**: Generates `{tag}_node_attr.json` and `{tag}_eap.json` (Stages 4–5).
* **`src/verify/`**: Generates `{tag}_verify.json` containing exact patching deltas (Stage 7).
* **`src/validation/`**: Generates `{tag}_validation.json` with completeness/out-of-graph drop metrics (Stage 9).
* **`src/graphs/`**: Renders `{tag}_dag.png` visual circuit plots (Stage 8/10).

## Novel extensions (`src/ftbgraph/`)

The notebook pipeline above is the original work. `src/ftbgraph/` is a separate, modular,
script-runnable package built on top of the same `Head` / `GradModel` / `metric_logprob_diff`
conventions that adds five methodologically new components, each grounded in post-2023
mechanistic-interpretability and multilingual-LLM literature rather than invented from scratch.
Run it with:

```bash
PYTHONPATH=src python3 -m ftbgraph.cli run --model gpt2 --type gpt2
```

Outputs land in `src/novel/{tag}_*.json` and `src/novel/{tag}_dag_novel.png`.

### 1. EAP-IG: integrated-gradient edge attribution (`attribution_ig.py`)

Plain EAP (Syed, Rager & Conmy, 2023) takes a single first-order gradient at the clean activation
point. Hanna, Pezzelle & Belinkov (2024), *"Have Faith in Faithfulness: Going Beyond Circuit
Overlap When Finding Model Mechanisms"* (https://arxiv.org/abs/2403.17806), show this produces
false negatives whenever the metric's gradient is near zero at that single point, and propose
EAP-IG: integrate the gradient along the path from the corrupted to the clean activation instead of
evaluating it once. `attribution_ig.py` implements this for every edge simultaneously: at each of
`m` interpolation steps it physically patches every attention head's pre-projection output to the
interpolated value (via a new `GradModel.patch_heads_ctx`), backpropagates into the **raw residual
stream at each layer boundary** (`GradModel.capture_block_inputs` — deliberately not the
layernormed attention input `capture_dst_inputs` uses for plain EAP, because once every head is
patched simultaneously that layernormed tensor has no live downstream consumer and its gradient is
identically zero; this was caught empirically while building this component, not assumed), and
averages. The script compares plain-EAP and EAP-IG rankings on identical samples and reports their
rank correlation, rather than asserting one is better without evidence.

### 2. Causal scrubbing / resample ablation (`causal_scrubbing.py`)

The original Stage 9 validation zero-ablates heads, which pushes activations off-distribution and
is exactly the kind of test causal scrubbing (Redwood Research; Chan et al., 2022,
https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y/p/JvZhhzycHu2Yd57RN) was designed to replace:
components are resampled from *other real inputs* rather than zeroed. `causal_scrubbing.py` ablates
in-graph and a size-matched random out-of-graph head set by patching in activations averaged over
`RESAMPLE_K` donor prompts whose target language differs from the probe's, then reports a bootstrap
confidence interval on the drop (not just a point estimate) and a permutation test p-value for
whether in-graph necessity exceeds out-of-graph necessity by more than chance. This directly
answers the README's own caveat about "wide standard deviations" in the original validation step
by giving that variance an actual interval and a significance test instead of a single mean.

### 3. Latent pivot-language probe (`latent_pivot.py`)

Wendler, Veselovsky, Monea & West, ACL 2024, *"Do Llamas Work in English? On the Latent Language
of Multilingual Transformers"* (https://arxiv.org/abs/2402.10588), find that Llama-2 answering a
non-English prompt detours through an English-dominant subspace in its middle layers before
re-entering the target language near the output. `latent_pivot.py` runs a logit-lens probe (through
each model's own final layer norm, not a bare unembedding) over every residual-stream layer of each
neutral prompt, detects which language each layer's top-1 decoded token belongs to, and computes
per model: the mean layer at which decoding first and consistently matches the *true* target
language (the "pivot-exit layer"), and how often English dominates before that point. It then
correlates this exit layer with the layer distribution of the verified FTB circuit's heads, turning
"are language-identity heads downstream of an English interlingua stage" into a concrete,
per-architecture number instead of an assumption carried over from a different model family.

### 4. Difference-in-means steering vectors (`steering.py`)

Both the original necessity test (ablate in-graph) and the causal-scrubbing test above are
*subtractive*: they show the circuit is needed, not that it is sufficient to steer behavior.
`steering.py` adds a complementary *additive* sufficiency test in the spirit of representation
engineering / ActAdd-style steering and Tang et al., ACL 2024, *"Language-Specific Neurons: The Key
to Multilingual Capabilities in Large Language Models"* (https://arxiv.org/abs/2403.02435), who
show multilingual output can be steered by manipulating a small, identified set of components. It
builds a difference-in-means vector between two languages' residual-stream activations at the
verified circuit's deepest layer, injects `alpha * vector` into a held-out prompt that natively
targets the *other* language, and measures both the shift in the log-prob-diff metric and the
fraction of prompts whose actual top-1 decoded token flips into the injected language — a causal
intervention the original pipeline has no equivalent of.

### 5. Cross-architecture functional universality (`universality.py`)

The original Stage 11 comparison is Jaccard similarity between a base/instruct *pair that share an
architecture*, so node labels (`L{layer}H{head}`) are directly comparable. Across GPT-2, BLOOM,
Pythia and Qwen, layer/head indices are not comparable at all — "L4H2" means nothing in common
between a 12-layer and a 28-layer model. `universality.py` instead represents each in-graph head by
its *functional signature*: a vector of single-head necessity-ablation drops, one entry per
language, independent of the model's indexing scheme. It then matches heads across model pairs by
cosine similarity in this signature space and tests the best-match quality against a permutation
null (shuffling which head gets which signature within one model), giving an actual p-value for the
claim "these two architectures converge on a functionally similar language-identity head" instead
of only being able to make that claim within one architecture family.

### 6. Dataset scale and consumption (`dataset_gen.py`)

The seed suite is expanded from 29 to a larger self-contained neutral/code-switch/semantic-
congruence suite (still fully in-notebook/in-code, per the original design philosophy — no external
corpus dependency was introduced), and `confound_report()` adds a check the original pipeline did
not have: because script-detected languages (zh/ru/ja) get thousands of candidate diagnostic
tokens while Latin-script languages get a hand-authored handful, the mean tokenizer-rank-as-
frequency-proxy of the two groups differs by roughly one to two orders of magnitude for every model
tested. That is a real confound on the log-prob-diff metric (it is partly comparing token frequency,
not only language identity) that the original pipeline did not surface, and this report flags it
rather than silently absorbing it into the headline numbers.

### Honest scope of what was actually run

This package was validated end-to-end on `gpt2` (124M) on CPU in this environment; see
`RESULTS.md` for the real numbers produced by that run, including where statistically rigorous
(bootstrap/permutation/FDR-corrected) validation disagrees with the original pipeline's looser
top-K + sign-agreement selection. BLOOM-560M, Pythia-1B/2.8B and the Qwen pair were not re-run here
for compute-budget reasons; the code path is architecture-generic (it reads `ARCH`/`TYPE_MAP` the
same way the original notebook does) and should run unchanged on them given a GPU session.
