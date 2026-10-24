from pathlib import Path
import random
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = SRC_DIR / "dataset"
EAP_DIR = SRC_DIR / "eap"
VERIFY_DIR = SRC_DIR / "verify"
GRAPH_DIR = SRC_DIR / "graphs"
VALID_DIR = SRC_DIR / "validation"
NOVEL_DIR = SRC_DIR / "novel"

for d in (DATA_DIR, EAP_DIR, VERIFY_DIR, GRAPH_DIR, VALID_DIR, NOVEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LANGS = ["en", "fr", "es", "vi", "zh", "ru", "ja"]

EAP_N = 128
VERIFY_TOPK = 250
VERIFY_DST_CAP = 8
VALID_N = 100
IG_STEPS = 8
STEER_LAYER_OFFSET = 0
BOOTSTRAP_N = 2000
PERM_N = 1000
FDR_Q = 0.05
RESAMPLE_K = 6
