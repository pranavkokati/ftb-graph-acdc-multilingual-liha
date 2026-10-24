# Novel-pipeline results on `gpt2`

Real output from `PYTHONPATH=src python3 -m ftbgraph.cli run --model gpt2 --type gpt2 --eap-n 32
--verify-topk 50 --verify-dst-cap 4 --verify-n 20 --ig-steps 6 --valid-n 20 --pivot-n 20`, run end to
end on CPU in this environment. Raw JSON is in `src/novel/gpt2_*.json`; the circuit plot is
`src/novel/gpt2_dag_novel.png`. BLOOM-560M / Pythia-1B / Pythia-2.8B / Qwen pair were not re-run
here (compute budget); the code is architecture-generic and should run unchanged on them given a
GPU session, and `universality.py`'s cross-architecture comparison specifically needs at least two
of these runs to produce a real number — it is implemented and unit-exercised but not executed
with real model pairs in this report.

## 1. Plain EAP vs EAP-IG disagree more than the original pipeline's design assumes

On identical samples, the top-50 edge sets from plain EAP and EAP-IG overlap on only 12 of 50 edges
(`src/novel/gpt2_eap_comparison.json`). One edge, **L8H6**, is in both top-5 lists independently —
that is the strongest single candidate for a real language-identity edge in `gpt2`, found twice by
two different attribution methods. The low overlap elsewhere is exactly the false-negative behavior
EAP-IG was built to catch (Hanna et al., 2024): plain EAP's single-point gradient and EAP-IG's
path-integrated gradient are not interchangeable approximations of the same quantity, so a pipeline
that trusts plain EAP's shortlist alone is silently dropping real candidates.

## 2. The EAP-IG + FDR-verified circuit for `gpt2`: 29 heads, 28 edges

Sources (no verified in-edge): `L1H1`, `L1H6`. The circuit concentrates in layers 1–8, mean layer
4.55, with no verified heads in layers 9–11. Full head list and plot in `src/novel/`.

## 3. Causal scrubbing says this circuit's necessity is *not* statistically supported — an honest negative result

This is the most important finding in this report, and it contradicts the original pipeline's own
pass/fail logic. Resample-ablating the 29 in-graph heads (patching in activations drawn from
same-batch prompts targeting a *different* language, not zeros) produces a mean metric drop of
**0.176** (bootstrap 95% CI **[0.044, 0.332]**, n=20). Resample-ablating a random, size-matched
(29-head) out-of-graph set produces a mean drop of **0.134** (CI **[-0.072, 0.318]**). The two
intervals overlap heavily, and a permutation test on the 20 paired differences gives **p = 0.746**
— nowhere close to significant. `necessity_supported: false`.

The original Stage 9 validation (zero-ablation, no CI, no significance test) would very likely have
reported "in-graph drop > out-of-graph drop" here and called it supported, because zero-ablating 29
attention heads out of ~144 is already a large, destructive, off-distribution intervention that
tends to hurt almost *any* metric, in-graph or not — exactly the failure mode causal scrubbing
(Redwood Research) exists to catch. The honest conclusion at this sample size (n=20 validation
prompts drawn from a 70-item neutral suite) is that the EAP-IG-discovered, exact-patching-verified,
FDR-significant *edges* are real, but a 29-head *circuit's necessity for the behavior as a whole* is
not yet established — the original pipeline's looser necessity check was giving a false sense of
confidence that a more rigorous counterfactual test does not reproduce.

## 4. The circuit sits mostly upstream of where GPT-2's output language stabilizes, not downstream

The logit-lens pivot probe (`latent_pivot.py`) finds the mean layer at which decoding first and
consistently matches the true target language is **6.93** (median 9.0) out of 13 (embeddings + 12
blocks). The verified circuit's mean head layer is **4.55**, and only **13.8%** of circuit heads sit
at or after that exit layer (`circuit_sits_downstream_of_pivot: false`). English does not show up as
a dominant intermediate decode anywhere except right at the final layer (layer 12, n=3) — i.e. this
particular model shows no clear sign of Wendler et al.'s English-pivot "detour" pattern at all,
which is a reasonable thing to find in a 124M-parameter, overwhelmingly English-trained model rather
than the much larger, genuinely multilingual Llama-2 that result was reported for. The useful
takeaway is methodological, not just a one-model data point: **"does this model use an English
interlingua" is a per-architecture empirical question this pipeline can now actually ask**, not an
assumption carried over from a different model family.

## 5. Steering: real, dose-dependent, directionally-correct effect on the metric — but it does not reliably flip the literal top-1 token

Injecting `alpha * (mean_lang_A - mean_lang_B)` into the residual stream at the circuit's deepest
verified layer (layer 8), on held-out prompts that natively target language B, shifts the target-
vs-foil log-prob-diff metric in the expected direction for essentially every one of the 15 tested
language-pair directions, scaling with `alpha` (2/4/8) as a real causal intervention should
(`src/novel/gpt2_novel_results.json`, `steering_sufficiency`). Concretely: `es->en` at alpha=8 shifts
the metric by +18.6; `ja->es` by +18.8; `fr->ru` by +15.4.

But the fraction of prompts whose literal argmax token flips into the injected language
(`flip_rate`) is inconsistent and often low even at alpha=8 — e.g. `es->en` flip_rate is 0.0 at
alpha=4 and alpha=8 despite a metric shift of +12.9/+18.6. The honest reading: the steering vector
reliably moves probability mass toward the target language's diagnostic tokens relative to the foil
language's, which is real sufficiency evidence for the circuit's output direction — but a 5-to-50000
token vocabulary has plenty of other high-probability candidates that a single-direction shift
doesn't displace, so metric-level sufficiency and literal next-token control are different claims
and should be reported separately, not conflated into one "it works" number.

There is also a striking, internally consistent artifact: every pair where the *target* language is
`zh`, `ru`, or `ja` shows metric shifts roughly 10–50x smaller (0.1–0.4) than pairs targeting `en`,
`es`, or `fr` (5–20), regardless of alpha. This lines up exactly with the confound
`dataset_gen.confound_report()` already flags on its own: `zh`/`ru`/`ja` get thousands of
script-detected diagnostic tokens versus a hand-authored handful for the Latin-script languages, so
the `mean log P` in the metric is averaged over a completely different-sized and different-frequency
candidate set for those three languages. The same root confound shows up independently in two
unrelated modules (a static tokenizer audit and a live causal-intervention experiment), which is
good evidence it is a real property of the metric and not a one-off computation error, and it means
any cross-language comparison of effect *magnitudes* (not direction) in this whole project family —
the original pipeline included — should be read with that caveat attached.
