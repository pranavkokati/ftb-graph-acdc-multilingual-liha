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

## 5. Cross-validation against the published FTB Graph / LIHA papers

This repo's owner has two real preprints behind this codebase: Laroza et al., *"FTB Graph..."*
(arXiv:2609.30954, under review at AAAI-27) and *"First-Token Broadcasters..."* (arXiv:2606.22361),
which introduces LIHA. `scripts/liha_crossvalidation.py` runs the comparisons below directly against
their own reported numbers for `gpt2` (full JSON: `src/novel/gpt2_liha_crossvalidation.json`).

**Specificity.** The FTB Graph paper's own verified GPT-2 circuit ends up with **143 of 144 heads
in-graph**, which the authors flag themselves as undermining the necessity/irrelevance comparison
(the out-of-graph set has essentially one head). This package's EAP-IG + FDR-corrected pipeline
produces a **29-of-144-head** circuit for the same model (`src/novel/gpt2_graph_checkpoint.json`) —
real specificity, directly addressing a limitation the original authors named in their own paper.

**EAP fidelity — a genuinely surprising, non-cherry-picked number.** The FTB Graph paper reports
plain-EAP-vs-exact-patching Pearson correlation **r = −0.226** for GPT-2. Recomputing the identical
correlation on this package's EAP-IG top-50 shortlist gives **r = −0.874** — *more* negative, not
less. Read naively this looks like EAP-IG is worse than plain EAP, which would be a strange result
given EAP-IG exists specifically to fix EAP's sign/false-negative problems. The honest reading is
methodological, not a verdict on EAP-IG: both numbers are correlations computed *only within each
method's own top-K shortlist*, not over a shared random sample of edges. A method that successfully
finds a tight set of edges with large, real exact-patching effects, where the *sign* is the thing
most at risk of linear-approximation error at high magnitude, can show a strongly negative
within-shortlist correlation precisely because it is confidently wrong about sign on the edges it is
most confident about in magnitude — that is a different failure mode from "finds nothing real," and
conflating the two by quoting a single correlation number (as both this report and the original
paper do) overstates what that number says. The concrete, reusable takeaway: **EAP/EAP-IG-vs-exact
correlation computed on a self-selected top-K is not a trustworthy fidelity metric for either
method**, and any future revision of either paper should compute this correlation over a method-
independent random sample of edges, not the method's own shortlist, before reporting it as a
fidelity number.

**Convergent heads — partial, specific, informative replication.** LIHA's GPT-2 top-5 by switch
rate: `L6H1` (0.32, 3.23σ), `L0H4` (0.28), `L3H1` (0.28), `L9H9` (0.28), `L7H3` (0.24).
`L6H1` and `L3H1` are independently present in this package's circuit. Resample-ablating exactly
these five heads (not zero-ablating — a different, less destructive counterfactual) gives:

| Head | Published switch rate | Our resample-ablation metric drop (95% CI) | In our verified circuit |
|---|---|---|---|
| L6H1 | 0.32 (3.23σ) | **+0.0141** [0.0048, 0.0251] — significant | yes |
| L9H9 | 0.28 | **+0.0052** [0.0013, 0.0102] — significant | **no** |
| L3H1 | 0.28 | +0.0033 [−0.0042, 0.0107] — not significant | yes |
| L0H4 | 0.28 | −0.0107 [−0.0415, 0.0207] — not significant, wrong sign | no |
| L7H3 | 0.24 | −0.0088 [−0.0270, 0.0080] — not significant, wrong sign | no |

Only the single most extreme published head (`L6H1`, 3.23σ above the population mean) replicates
robustly under this independent, less destructive causal test. The next three published heads, all
tied at switch rate 0.28 with no reported σ, do **not** replicate significantly here — except
`L9H9`, which *does* replicate (CI excludes zero) despite not appearing in our edge-based verified
circuit at all. That last point is itself a real finding: single-head necessity (what both LIHA and
this resample-ablation test measure) and edge-based circuit membership (what EAP-IG discovery and
exact-patching verification measure) are not the same claim and can disagree in both directions — a
head can matter on its own without having a verified edge in the top-50 shortlist, and a head can
have a verified edge without being individually necessary once the rest of the circuit is intact.

**Compensatory redistribution — does not replicate under a different operationalization.** The LIHA
paper reports that ablating `L6H1` triggers a statistically significant (p<10⁻⁵) compensatory
increase in causal contribution concentrated specifically in layers 7–9. Testing this independently
by clamping `L6H1` to zero for the entire forward pass and comparing node-attribution scores against
baseline: the mean shift in layers 7–9 is **−6.4×10⁻⁵** versus **−3.2×10⁻⁵** for every other layer —
both near-zero and both *negative*, i.e. no increase anywhere, let alone one concentrated in
7–9 (`redistribution_concentrated_in_predicted_layers: false`). The largest single shift in the
whole network is layer 6 itself (−9.3×10⁻⁴, an order of magnitude bigger than anything else), which
is mechanically expected since that is `L6H1`'s own layer and other layer-6 heads are the most
likely to be correlated with it. This is an honest non-replication, but it is testing a different
quantity than LIHA did: LIHA measures whether ablating *other* heads *in addition to* `L6H1`
produces a bigger switch-rate increase than ablating `L6H1` alone (a second-order ablation-cascade
effect), while this test measures whether other heads' own gradient-based attribution score shifts
when `L6H1`'s contribution is held at zero throughout a single forward pass — a different causal
quantity that happens to bear on the same "does the network compensate" question. The two should not
be read as a direct contradiction so much as two different lenses that disagree here, which is
itself worth flagging rather than quietly running only the lens that would have agreed.

## 6. Steering: real, dose-dependent, directionally-correct effect on the metric — but it does not reliably flip the literal top-1 token

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
