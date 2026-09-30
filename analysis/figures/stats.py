"""Statistics used by the figures: bootstrap CIs, effect sizes, rank correlations."""
import numpy as np
from scipy import stats

RNG_SEED = 0


def median_ci(x, n_boot=2000, level=0.95):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    boots = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def ratio_of_medians_ci(num, den, n_boot=2000, level=0.95):
    """median(num) / median(den) with a percentile-bootstrap CI (independent resampling)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    a = np.median(rng.choice(num, size=(n_boot, len(num)), replace=True), axis=1)
    b = np.median(rng.choice(den, size=(n_boot, len(den)), replace=True), axis=1)
    r = a / b
    lo, hi = np.quantile(r, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(np.median(num) / np.median(den)), float(lo), float(hi)


def cliffs_delta(a, b):
    """Cliff's delta: P(a > b) - P(a < b). |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # O(n log n) via ranks
    allv = np.concatenate([a, b])
    ranks = stats.rankdata(allv)
    ra = ranks[: len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def mann_whitney(a, b):
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def spearman(x, y):
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def linfit(x, y):
    res = stats.linregress(x, y)
    return {"slope": float(res.slope), "intercept": float(res.intercept), "r2": float(res.rvalue ** 2),
            "p": float(res.pvalue), "slope_stderr": float(res.stderr)}


def icc_oneway(groups):
    """ICC(1): share of variance between trials (groups = list of arrays, one per trial)."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    k = np.mean([len(g) for g in groups])
    grand = np.mean(np.concatenate(groups))
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (sum(len(g) for g in groups) - len(groups))
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else float("nan")
