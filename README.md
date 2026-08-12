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
git clone https://github.com/your-username/ftb-graph.git
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
