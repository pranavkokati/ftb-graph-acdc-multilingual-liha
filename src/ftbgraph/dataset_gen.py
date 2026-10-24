import random
from collections import Counter

import numpy as np

NEUTRAL_TEMPLATES = {
    "en": [
        "The capital of France", "The sun rises in the", "Paris is the capital of",
        "A week has seven", "The opposite of hot is", "Ice melts into",
        "The largest ocean on Earth is the", "Dogs are a type of",
        "The Great Wall is located in", "Fire is extremely",
    ],
    "fr": [
        "La capitale de la France", "Le soleil se lève à l'", "Paris est la capitale de la",
        "Une semaine a sept", "Le contraire du chaud est le", "La glace fond en",
        "Le plus grand océan du monde est le", "Les chiens sont un type d'",
        "La Grande Muraille se trouve en", "Le feu est extrêmement",
    ],
    "es": [
        "La capital de Francia", "El sol sale por el", "París es la capital de",
        "Una semana tiene siete", "Lo contrario de caliente es", "El hielo se derrite en",
        "El océano más grande del mundo es el", "Los perros son un tipo de",
        "La Gran Muralla se encuentra en", "El fuego es extremadamente",
    ],
    "vi": [
        "Thủ đô của Pháp", "Nước sôi ở một trăm độ", "Mặt trời mọc ở",
        "Một tuần có bảy", "Paris là thủ đô của", "Nước đá tan thành",
    ],
    "zh": [
        "法国的首都", "水在一百度", "太阳从",
        "一个星期有七", "巴黎是法国的", "长城位于",
    ],
    "ru": [
        "Столица Франции", "Вода кипит при ста градусах", "Солнце всходит на",
        "В неделе семь", "Париж является столицей", "Лёд превращается в",
    ],
    "ja": [
        "フランスの首都", "水は百度で", "太陽は",
        "一週間は七", "パリはフランスの", "氷は溶けて",
    ],
}

CODE_SWITCH_TEMPLATES = {
    "fr": ["I think the capital of France, c'est-à-dire la capitale, est"],
    "es": ["I believe the capital of Spain, es decir la capital, es"],
    "vi": ["I think the capital city, tức là thủ đô, là"],
    "zh": ["I think the capital, namely 首都, 是"],
    "ru": ["I think the capital, that is столица, это"],
    "ja": ["I think the capital, that is 首都, は"],
}

SEMANTIC_CONGRUENCE_TEMPLATES = {
    "fr_eiffel": ("La Tour Eiffel se trouve à", "congruent"),
    "fr_china": ("La Grande Muraille se trouve en", "incongruent"),
}


def build_suite(seed):
    suite = []
    langs = list(NEUTRAL_TEMPLATES)

    for lang in langs:
        for prefix in NEUTRAL_TEMPLATES[lang]:
            foil = random.Random(hash(prefix) & 0xFFFF).choice([l for l in langs if l != lang])
            foil_prefix = random.Random(hash((prefix, foil)) & 0xFFFF).choice(NEUTRAL_TEMPLATES[foil])
            suite.append({
                "prompt": prefix, "corrupted": foil_prefix,
                "target_lang": lang, "foil_lang": foil, "suite": "neutral",
            })

    for lang, prefixes in CODE_SWITCH_TEMPLATES.items():
        for prefix in prefixes:
            foil = random.Random(hash(prefix) & 0xFFFF).choice([l for l in langs if l != lang and l != "en"])
            foil_prefix = random.Random(hash((prefix, foil)) & 0xFFFF).choice(NEUTRAL_TEMPLATES[foil])
            suite.append({
                "prompt": prefix, "corrupted": foil_prefix,
                "target_lang": lang, "foil_lang": foil, "suite": "code_switch",
            })

    for key, (prefix, congruence) in SEMANTIC_CONGRUENCE_TEMPLATES.items():
        foil = "es"
        foil_prefix = random.Random(hash(prefix) & 0xFFFF).choice(NEUTRAL_TEMPLATES[foil])
        suite.append({
            "prompt": prefix, "corrupted": foil_prefix,
            "target_lang": "fr", "foil_lang": foil, "suite": f"semantic_{congruence}",
        })

    return suite


def suite_report(suite):
    return {
        "n_total": len(suite),
        "by_suite": dict(Counter(s["suite"] for s in suite)),
        "by_target_lang": dict(Counter(s["target_lang"] for s in suite)),
    }


def frequency_rank_proxy(tok, token_id):
    return token_id


def confound_report(tok, lang_token_cache):
    report = {}
    for lang, ids in lang_token_cache.items():
        ranks = [frequency_rank_proxy(tok, tid) for tid in ids]
        report[lang] = {"n_tokens": len(ids), "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks))}
    langs = list(report)
    flags = []
    for i, l1 in enumerate(langs):
        for l2 in langs[i + 1:]:
            r1, r2 = report[l1]["mean_rank"], report[l2]["mean_rank"]
            if min(r1, r2) == 0:
                continue
            ratio = max(r1, r2) / min(r1, r2)
            if ratio > 3.0:
                flags.append({"lang_a": l1, "lang_b": l2, "rank_ratio": ratio})
    return {"per_lang": report, "imbalance_flags": flags}
