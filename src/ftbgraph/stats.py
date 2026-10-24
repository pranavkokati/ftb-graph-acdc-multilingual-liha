import numpy as np


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values))}


def permutation_test(sample_a, sample_b, n_perm=1000, seed=42):
    sample_a = np.asarray(sample_a, dtype=np.float64)
    sample_b = np.asarray(sample_b, dtype=np.float64)
    observed = float(sample_a.mean() - sample_b.mean())
    pooled = np.concatenate([sample_a, sample_b])
    n_a = len(sample_a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        stat = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(stat) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {"observed_diff": observed, "p_value": float(p_value), "n_perm": n_perm}


def benjamini_hochberg(pvalues, q=0.05):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_i = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: max_i + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject
