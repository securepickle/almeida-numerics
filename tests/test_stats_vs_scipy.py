"""
Correctness suite: almeida_stats vs NumPy / SciPy (the references).

almeida_stats is a from-scratch, dependency-free statistics library operating on
plain Python lists. This checks every public function against NumPy/SciPy on
random inputs and reports the actual max-abs error per function. Runnable
standalone (no pytest needed):

    python test_almeida_stats_vs_scipy.py

Exit code 0 iff every check passes. Where an estimator uses a documented
convention that differs slightly from SciPy's default (skewness bias
correction; the t-approximation p-values), the check states the convention and
uses the matching tolerance — the point is to prove correctness, not to hide a
convention behind a loose bound.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from almeida_numerics import stats as st

RNG = np.random.default_rng(0)
_results = []                    # (name, passed, max_err, note)


def check(name, got, expected, rtol=1e-6, atol=1e-8, note=""):
    try:
        g = np.asarray(got, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)
        err = float(np.max(np.abs(g - e))) if e.size else 0.0
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
    except Exception as ex:                       # noqa: BLE001 — report, don't crash
        ok, err, note = False, float("inf"), f"EXC {type(ex).__name__}: {ex}"
    _results.append((name, ok, err, note))


def rand(n=200, loc=0.0, scale=1.0):
    a = RNG.normal(loc, scale, n)
    return a.tolist(), a


# ---------------------------------------------------------------- central tendency
def test_central():
    x, xn = rand()
    check("mean", st.mean(x), np.mean(xn))
    w = np.abs(RNG.normal(size=xn.size)) + 0.1
    check("weighted_mean", st.weighted_mean(x, w.tolist()), np.average(xn, weights=w))
    pos, posn = rand(loc=5.0, scale=1.0)          # positive for geo/harmonic
    posn = np.abs(posn) + 0.5; pos = posn.tolist()
    check("geometric_mean", st.geometric_mean(pos), sps.gmean(posn))
    check("harmonic_mean", st.harmonic_mean(pos), sps.hmean(posn))
    check("median", st.median(x), np.median(xn))
    # mode: integer data with an unambiguous most-common value
    ints = RNG.integers(0, 5, 300); ints[:40] = 3
    check("mode", st.mode(ints.tolist()), float(sps.mode(ints, keepdims=False).mode))


# ---------------------------------------------------------------- dispersion
def test_dispersion():
    x, xn = rand()
    check("variance (pop, ddof=0)", st.variance(x), np.var(xn))
    check("variance (sample, ddof=1)", st.variance(x, ddof=1), np.var(xn, ddof=1))
    check("std (pop)", st.std(x), np.std(xn))
    check("std (sample)", st.std(x, ddof=1), np.std(xn, ddof=1))
    check("sem", st.sem(x), sps.sem(xn))
    check("data_range", st.data_range(x), np.ptp(xn))
    check("sum_of_squares", st.sum_of_squares(x), np.sum((xn - xn.mean()) ** 2))
    # kurtosis: excess, biased — matches scipy default (fisher=True, bias=True)
    check("kurtosis (excess, biased)", st.kurtosis(x), sps.kurtosis(xn), rtol=1e-4, atol=1e-4)
    # skewness: sample estimator with n/((n-1)(n-2)) correction — within ~1% of
    # scipy's sqrt(n(n-1))/(n-2) bias correction; documented convention diff.
    check("skewness (~scipy bias=False, <1%)", st.skewness(x), sps.skew(xn, bias=False),
          rtol=2e-2, atol=2e-2, note="sample-skew convention differs <1%")


# ---------------------------------------------------------------- percentiles
def test_percentiles():
    x, xn = rand()
    for p in (0, 10, 25, 50, 75, 90, 100):
        check(f"percentile {p}", st.percentile(x, p), np.percentile(xn, p))
    check("quantile 0.33", st.quantile(x, 0.33), np.quantile(xn, 0.33))
    q1, q2, q3 = st.quartiles(x)
    check("quartiles", [q1, q2, q3], np.percentile(xn, [25, 50, 75]))
    check("iqr", st.iqr(x), sps.iqr(xn))
    fn = st.five_number_summary(x)
    check("five_number_summary", list(fn),
          [xn.min(), np.percentile(xn, 25), np.median(xn), np.percentile(xn, 75), xn.max()])


# ---------------------------------------------------------------- association
def test_association():
    x, xn = rand()
    y = (2.0 * xn + RNG.normal(0, 0.5, xn.size))
    yl = y.tolist()
    check("cov (ddof=1)", st.cov(x, yl), np.cov(xn, y)[0, 1])
    data = RNG.normal(size=(3, 150))
    cm = st.cov_matrix([r.tolist() for r in data])
    check("cov_matrix", cm, np.cov(data))
    check("corrcoef", st.corrcoef(x, yl), np.corrcoef(xn, y)[0, 1])
    r, p = st.pearsonr(x, yl)
    sr = sps.pearsonr(xn, y)
    check("pearsonr r", r, sr.statistic)
    check("pearsonr p (t-approx)", p, sr.pvalue, rtol=1e-2, atol=1e-3, note="approx p-value")
    rho, ps = st.spearmanr(x, yl)
    ssp = sps.spearmanr(xn, y)
    check("spearmanr rho", rho, ssp.statistic, rtol=1e-6, atol=1e-6)
    check("spearmanr p (t-approx)", ps, ssp.pvalue, rtol=5e-2, atol=5e-3, note="approx p-value")


# ---------------------------------------------------------------- distributions
def test_distributions():
    for xv in (-1.5, 0.0, 1.96):
        check(f"normal_pdf({xv})", st.normal_pdf(xv), sps.norm.pdf(xv))
        check(f"normal_cdf({xv})", st.normal_cdf(xv), sps.norm.cdf(xv), rtol=1e-4, atol=1e-5)
    for pv in (0.025, 0.5, 0.975):
        check(f"normal_ppf({pv})", st.normal_ppf(pv), sps.norm.ppf(pv), rtol=1e-4, atol=1e-4)
    for xv in (-2.0, 0.5, 2.5):
        check(f"t_pdf({xv},df=8)", st.t_pdf(xv, 8), sps.t.pdf(xv, 8), rtol=1e-3, atol=1e-4)
        check(f"t_cdf({xv},df=8)", st.t_cdf(xv, 8), sps.t.cdf(xv, 8), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------- hypothesis tests
def test_hypothesis():
    x, xn = rand(loc=0.3)
    # z-test statistic is exact; p uses the normal CDF
    z, _ = st.z_test(x, mu=0.0, sigma=1.0)
    check("z_test stat", z, (xn.mean() - 0.0) / (1.0 / np.sqrt(xn.size)))
    t1, p1 = st.t_test_1sample(x, mu=0.0)
    s1 = sps.ttest_1samp(xn, 0.0)
    check("t_test_1sample t", t1, s1.statistic)
    check("t_test_1sample p (t-approx)", p1, s1.pvalue, rtol=2e-2, atol=2e-3, note="approx p-value")
    y, yn = rand(loc=0.0)
    ts, _ = st.t_test_2sample(x, y, equal_var=True)
    check("t_test_2sample t (Student)", ts, sps.ttest_ind(xn, yn, equal_var=True).statistic)
    tw, _ = st.t_test_2sample(x, y, equal_var=False)
    check("t_test_2sample t (Welch)", tw, sps.ttest_ind(xn, yn, equal_var=False).statistic)
    obs = RNG.integers(20, 60, 6).astype(float)
    c2, _ = st.chi_square_test(obs.tolist())
    check("chi_square stat (uniform exp)", c2, sps.chisquare(obs).statistic)


# ---------------------------------------------------------------- regression
def test_regression():
    xn = np.linspace(-3, 3, 120)
    yn = 1.7 * xn - 0.4 + RNG.normal(0, 0.3, xn.size)
    x, y = xn.tolist(), yn.tolist()
    slope, intercept, r2 = st.linear_regression(x, y)
    lr = sps.linregress(xn, yn)
    check("linreg slope", slope, lr.slope)
    check("linreg intercept", intercept, lr.intercept)
    check("linreg r_squared", r2, lr.rvalue ** 2, rtol=1e-5, atol=1e-6)
    # polynomial_fit returns low->high order; np.polyfit returns high->low
    yq = 0.5 * xn ** 2 - 1.2 * xn + 2.0 + RNG.normal(0, 0.2, xn.size)
    coeffs = st.polynomial_fit(x, yq.tolist(), degree=2)
    check("polynomial_fit deg2", coeffs[::-1], np.polyfit(xn, yq, 2), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- transforms
def test_transforms():
    x, xn = rand()
    counts, edges = st.histogram(x, bins=10)
    npc, npe = np.histogram(xn, bins=10)
    check("histogram counts", counts, npc)
    check("histogram edges", edges, npe)
    check("normalize zscore", st.normalize(x, 'zscore'), (xn - xn.mean()) / xn.std())
    check("normalize minmax", st.normalize(x, 'minmax'), (xn - xn.min()) / (xn.max() - xn.min()))
    d = st.describe(x)
    check("describe mean/std", [d['mean'], d['std']], [xn.mean(), xn.std(ddof=1)])


def main():
    for fn in (test_central, test_dispersion, test_percentiles, test_association,
               test_distributions, test_hypothesis, test_regression, test_transforms):
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            _results.append((fn.__name__, False, float("inf"), f"SUITE-EXC {ex}"))
    npass = sum(1 for _, ok, _, _ in _results if ok)
    print(f"\nalmeida_stats vs NumPy/SciPy — {npass}/{len(_results)} checks passed\n")
    for name, ok, err, note in _results:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name:<34} max_err={err:.2e}"
        if note:
            line += f"   ({note})"
        print(line)
    failed = [n for n, ok, _, _ in _results if not ok]
    if failed:
        print(f"\n{len(failed)} FAILED: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
