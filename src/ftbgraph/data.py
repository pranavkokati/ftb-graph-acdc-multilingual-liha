import json
import random
from collections import Counter

import torch.nn.functional as F

from . import config

SCRIPT_RANGES = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
    "ru": [(0x0400, 0x04FF)],
    "ko": [(0xAC00, 0xD7AF)],
}
LANG_META = {
    "en": "English", "fr": "French", "es": "Spanish", "vi": "Vietnamese",
    "zh": "Chinese", "ru": "Russian", "ja": "Japanese",
}
LATIN_LANG_WORDS = {
    "en": ["is", "the", "and", "of", "in"],
    "fr": ["est", "le", "la", "et", "dans"],
    "es": ["es", "el", "la", "y", "en"],
    "vi": ["là", "và", "của", "trong", "ở"],
}


def script_of_char(ch):
    cp = ord(ch)
    for lang, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return lang
    return None


def detect_lang(text, fallback_langs):
    for ch in text:
        s = script_of_char(ch)
        if s is not None:
            return s
    low = text.strip().lower()
    for lang in fallback_langs:
        for word in LATIN_LANG_WORDS.get(lang, []):
            if low == word.lower():
                return lang
    return None


def build_lang_token_ids(tok, lang):
    if lang in SCRIPT_RANGES:
        ids = []
        for tid in range(tok.vocab_size):
            s = tok.decode([tid])
            if s.strip() and all((script_of_char(c) == lang or not c.isalpha()) for c in s):
                ids.append(tid)
        return ids
    if lang not in LATIN_LANG_WORDS:
        raise ValueError(f"no candidate word list for {lang}")
    kept = []
    for word in LATIN_LANG_WORDS[lang]:
        ids_for_word = tok.encode(" " + word, add_special_tokens=False)
        if len(ids_for_word) == 1:
            kept.append(ids_for_word[0])
    return kept


def audit_all_lang_tokens(tok, langs):
    per_lang = {lang: set(build_lang_token_ids(tok, lang)) for lang in langs}
    return {l: list(ids) for l, ids in per_lang.items() if ids}


def metric_logprob_diff(logits, target_ids, foil_ids):
    import torch
    logp = F.log_softmax(logits.float(), dim=-1)[0]
    target_lp = logp[target_ids].mean() if target_ids else torch.tensor(0.0, device=logits.device)
    foil_lp = logp[foil_ids].mean() if foil_ids else torch.tensor(0.0, device=logits.device)
    return target_lp - foil_lp


def select_balanced_samples(suite, suite_name, n, langs=None, seed=config.SEED):
    pool = [s for s in suite if s["suite"] == suite_name]
    if langs:
        pool = [s for s in pool if s["target_lang"] in langs and s["foil_lang"] in langs]
    rng = random.Random(seed)
    rng.shuffle(pool)
    if n is not None and n < len(pool):
        pool = pool[:n]
    return pool


def summarize_samples(samples):
    return {"n_samples": len(samples), "by_lang": dict(Counter(s["target_lang"] for s in samples))}


def load_suite(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_suite(suite, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
