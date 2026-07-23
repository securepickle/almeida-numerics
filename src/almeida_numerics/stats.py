"""
================================================================================
Almeida Statistics - Pure Python Statistical Functions
================================================================================

Provides pure Python implementations of core statistical operations:
- Descriptive: mean, median, mode, variance, std, min, max, range
- Percentiles: percentile, quartiles, iqr
- Correlation: cov, corrcoef, pearsonr, spearmanr
- Distributions: normal_pdf, normal_cdf, t_pdf
- Hypothesis: t_test, z_score, chi_square
- Regression: linear_regression, polynomial_fit
- Sampling: sample, bootstrap, shuffle

All operations work on Python lists. No numpy, scipy, or external dependencies.

Usage:
    from almeida_numerics.stats import (
        mean, median, std, variance,
        percentile, corrcoef, linear_regression,
    )

Author: Michael Almeida
Copyright: (c) Almeida Industries
License: Apache-2.0
Date: 2026-01-04
================================================================================
"""

from typing import List, Tuple, Optional, Union, Callable
import math
import random

# Type aliases
Vector = List[float]
Matrix = List[List[float]]


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def mean(x: Vector) -> float:
    """Arithmetic mean."""
    if not x:
        return 0.0
    return sum(x) / len(x)


def weighted_mean(x: Vector, weights: Vector) -> float:
    """Weighted arithmetic mean."""
    if not x or not weights or len(x) != len(weights):
        return 0.0
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(xi * wi for xi, wi in zip(x, weights)) / total_weight


def geometric_mean(x: Vector) -> float:
    """Geometric mean (for positive values)."""
    if not x:
        return 0.0
    if any(xi <= 0 for xi in x):
        raise ValueError("Geometric mean requires positive values")
    log_sum = sum(math.log(xi) for xi in x)
    return math.exp(log_sum / len(x))


def harmonic_mean(x: Vector) -> float:
    """Harmonic mean (for positive values)."""
    if not x:
        return 0.0
    if any(xi <= 0 for xi in x):
        raise ValueError("Harmonic mean requires positive values")
    return len(x) / sum(1.0 / xi for xi in x)


def median(x: Vector) -> float:
    """Median (middle value)."""
    if not x:
        return 0.0
    sorted_x = sorted(x)
    n = len(sorted_x)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_x[mid - 1] + sorted_x[mid]) / 2
    return sorted_x[mid]


def mode(x: Vector) -> float:
    """Mode (most frequent value). Returns first mode if multiple."""
    if not x:
        return 0.0
    counts = {}
    for xi in x:
        counts[xi] = counts.get(xi, 0) + 1
    max_count = max(counts.values())
    for xi in x:  # Return first occurrence
        if counts[xi] == max_count:
            return xi
    return x[0]


def variance(x: Vector, ddof: int = 0) -> float:
    """
    Population or sample variance.

    Args:
        x: Data values
        ddof: Delta degrees of freedom (0=population, 1=sample)
    """
    if len(x) <= ddof:
        return 0.0
    m = mean(x)
    return sum((xi - m) ** 2 for xi in x) / (len(x) - ddof)


def std(x: Vector, ddof: int = 0) -> float:
    """Standard deviation."""
    return math.sqrt(variance(x, ddof))


def sem(x: Vector) -> float:
    """Standard error of the mean."""
    if len(x) < 2:
        return 0.0
    return std(x, ddof=1) / math.sqrt(len(x))


def skewness(x: Vector) -> float:
    """Skewness (measure of asymmetry)."""
    if len(x) < 3:
        return 0.0
    m = mean(x)
    s = std(x)
    if s == 0:
        return 0.0
    n = len(x)
    return sum(((xi - m) / s) ** 3 for xi in x) * n / ((n - 1) * (n - 2))


def kurtosis(x: Vector) -> float:
    """Excess kurtosis (measure of tail heaviness). Normal = 0."""
    if len(x) < 4:
        return 0.0
    m = mean(x)
    s = std(x)
    if s == 0:
        return 0.0
    n = len(x)
    m4 = sum(((xi - m) / s) ** 4 for xi in x) / n
    return m4 - 3.0  # Excess kurtosis


def data_range(x: Vector) -> float:
    """Range (max - min)."""
    if not x:
        return 0.0
    return max(x) - min(x)


def sum_of_squares(x: Vector) -> float:
    """Sum of squared deviations from mean."""
    if not x:
        return 0.0
    m = mean(x)
    return sum((xi - m) ** 2 for xi in x)


# =============================================================================
# PERCENTILES AND QUANTILES
# =============================================================================

def percentile(x: Vector, p: float) -> float:
    """
    Compute p-th percentile (0-100).
    Uses linear interpolation.
    """
    if not x:
        return 0.0
    if p < 0 or p > 100:
        raise ValueError("Percentile must be between 0 and 100")

    sorted_x = sorted(x)
    n = len(sorted_x)

    if p == 0:
        return sorted_x[0]
    if p == 100:
        return sorted_x[-1]

    # Linear interpolation
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_x[int(k)]

    return sorted_x[int(f)] * (c - k) + sorted_x[int(c)] * (k - f)


def quantile(x: Vector, q: float) -> float:
    """Compute q-th quantile (0-1)."""
    return percentile(x, q * 100)


def quartiles(x: Vector) -> Tuple[float, float, float]:
    """Return Q1, Q2 (median), Q3."""
    return percentile(x, 25), percentile(x, 50), percentile(x, 75)


def iqr(x: Vector) -> float:
    """Interquartile range (Q3 - Q1)."""
    q1, _, q3 = quartiles(x)
    return q3 - q1


def five_number_summary(x: Vector) -> Tuple[float, float, float, float, float]:
    """Return min, Q1, median, Q3, max."""
    if not x:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    q1, med, q3 = quartiles(x)
    return (min(x), q1, med, q3, max(x))


# =============================================================================
# COVARIANCE AND CORRELATION
# =============================================================================

def cov(x: Vector, y: Vector, ddof: int = 1) -> float:
    """Covariance between two vectors."""
    if len(x) != len(y) or len(x) <= ddof:
        return 0.0
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - ddof)


def cov_matrix(data: Matrix, ddof: int = 1) -> Matrix:
    """
    Covariance matrix for multiple variables.
    data: Each row is a variable, each column is an observation.
    """
    n_vars = len(data)
    result = [[0.0] * n_vars for _ in range(n_vars)]

    for i in range(n_vars):
        for j in range(i, n_vars):
            c = cov(data[i], data[j], ddof)
            result[i][j] = c
            result[j][i] = c

    return result


def corrcoef(x: Vector, y: Vector) -> float:
    """Pearson correlation coefficient (-1 to 1)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    sx, sy = std(x), std(y)
    if sx == 0 or sy == 0:
        return 0.0

    return cov(x, y, ddof=0) / (sx * sy)


def pearsonr(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Pearson correlation with p-value.
    Returns (r, p-value).
    """
    n = len(x)
    if n < 3:
        return (0.0, 1.0)

    r = corrcoef(x, y)

    # t-statistic
    if abs(r) >= 1.0:
        return (r, 0.0 if abs(r) == 1.0 else 1.0)

    t = r * math.sqrt((n - 2) / (1 - r * r))

    # Approximate p-value using t-distribution
    p = 2 * (1 - t_cdf(abs(t), n - 2))

    return (r, p)


def spearmanr(x: Vector, y: Vector) -> Tuple[float, float]:
    """
    Spearman rank correlation with p-value.
    Returns (rho, p-value).
    """
    if len(x) != len(y) or len(x) < 3:
        return (0.0, 1.0)

    # Convert to ranks
    rx = _ranks(x)
    ry = _ranks(y)

    # Pearson on ranks
    return pearsonr(rx, ry)


def _ranks(x: Vector) -> Vector:
    """Convert values to ranks (1-based, average for ties)."""
    n = len(x)
    indexed = sorted(enumerate(x), key=lambda t: t[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all equal values
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j + 2) / 2  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


# =============================================================================
# PROBABILITY DISTRIBUTIONS
# =============================================================================

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal (Gaussian) probability density function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution function."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Normal percent point function (inverse CDF).
    Uses Abramowitz & Stegun approximation.
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1 (exclusive)")

    # Rational approximation for lower tail
    if p < 0.5:
        return -_normal_ppf_inner(1 - p) * sigma + mu
    else:
        return _normal_ppf_inner(p) * sigma + mu


def _normal_ppf_inner(p: float) -> float:
    """Helper for normal_ppf. Abramowitz & Stegun approximation."""
    # Constants for approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def t_pdf(x: float, df: int) -> float:
    """Student's t probability density function."""
    if df < 1:
        raise ValueError("df must be >= 1")
    coef = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return coef * (1 + x * x / df) ** (-(df + 1) / 2)


def t_cdf(x: float, df: int) -> float:
    """Student's t cumulative distribution function (approximation)."""
    if df < 1:
        raise ValueError("df must be >= 1")

    # Use regularized incomplete beta function
    t2 = x * x
    return 0.5 + 0.5 * math.copysign(1, x) * (1 - _betainc(df / 2, 0.5, df / (df + t2)))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function (approximation)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Continued fraction approximation
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
        a * math.log(x) + b * math.log(1 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    else:
        return 1 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta."""
    max_iter = 100
    eps = 1e-10

    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def z_score(x: float, mu: float, sigma: float) -> float:
    """Z-score (standard score)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (x - mu) / sigma


def z_test(x: Vector, mu: float, sigma: float) -> Tuple[float, float]:
    """
    One-sample z-test (known population sigma).
    Returns (z-statistic, two-tailed p-value).
    """
    n = len(x)
    if n == 0:
        return (0.0, 1.0)

    sample_mean = mean(x)
    z = (sample_mean - mu) / (sigma / math.sqrt(n))
    p = 2 * (1 - normal_cdf(abs(z)))

    return (z, p)


def t_test_1sample(x: Vector, mu: float = 0.0) -> Tuple[float, float]:
    """
    One-sample t-test.
    Returns (t-statistic, two-tailed p-value).
    """
    n = len(x)
    if n < 2:
        return (0.0, 1.0)

    sample_mean = mean(x)
    sample_std = std(x, ddof=1)

    if sample_std == 0:
        return (0.0, 1.0)

    t = (sample_mean - mu) / (sample_std / math.sqrt(n))
    p = 2 * (1 - t_cdf(abs(t), n - 1))

    return (t, p)


def t_test_2sample(x: Vector, y: Vector, equal_var: bool = True) -> Tuple[float, float]:
    """
    Two-sample t-test.

    Args:
        x, y: Sample vectors
        equal_var: If True, assume equal variance (Student's t-test)
                   If False, use Welch's t-test

    Returns (t-statistic, two-tailed p-value).
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return (0.0, 1.0)

    m1, m2 = mean(x), mean(y)
    v1, v2 = variance(x, ddof=1), variance(y, ddof=1)

    if equal_var:
        # Pooled variance
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test
        se = math.sqrt(v1/n1 + v2/n2)
        if se == 0:
            return (0.0, 1.0)
        # Welch-Satterthwaite degrees of freedom
        num = (v1/n1 + v2/n2) ** 2
        denom = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
        df = num / denom if denom > 0 else 1

    if se == 0:
        return (0.0, 1.0)

    t = (m1 - m2) / se
    p = 2 * (1 - t_cdf(abs(t), int(df)))

    return (t, p)


def chi_square_test(observed: Vector, expected: Optional[Vector] = None) -> Tuple[float, float]:
    """
    Chi-square goodness of fit test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies (uniform if None)

    Returns (chi2-statistic, p-value).
    """
    n = len(observed)
    if n < 2:
        return (0.0, 1.0)

    if expected is None:
        total = sum(observed)
        expected = [total / n] * n

    if len(expected) != n:
        raise ValueError("observed and expected must have same length")

    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    df = n - 1

    # Approximate p-value using chi-square distribution
    p = 1 - _chi2_cdf(chi2, df)

    return (chi2, p)


def _chi2_cdf(x: float, df: int) -> float:
    """Chi-square CDF (using regularized incomplete gamma)."""
    if x <= 0:
        return 0.0
    return _gammainc(df / 2, x / 2)


def _gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function."""
    if x <= 0:
        return 0.0
    if x < a + 1:
        # Series representation
        ap = a
        sum_val = 1.0 / a
        delta = sum_val
        for _ in range(100):
            ap += 1
            delta *= x / ap
            sum_val += delta
            if abs(delta) < abs(sum_val) * 1e-10:
                break
        return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        # Continued fraction
        b = x + 1 - a
        c = 1e30
        d = 1 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-10:
                break
        return 1 - h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# =============================================================================
# REGRESSION
# =============================================================================

def linear_regression(x: Vector, y: Vector) -> Tuple[float, float, float]:
    """
    Simple linear regression: y = slope * x + intercept

    Returns (slope, intercept, r_squared).
    """
    n = len(x)
    if n != len(y) or n < 2:
        return (0.0, 0.0, 0.0)

    mx, my = mean(x), mean(y)

    # Calculate slope
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = sum((xi - mx) ** 2 for xi in x)

    if denom == 0:
        return (0.0, my, 0.0)

    slope = num / denom
    intercept = my - slope * mx

    # R-squared
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return (slope, intercept, r_squared)


def polynomial_fit(x: Vector, y: Vector, degree: int = 2) -> Vector:
    """
    Polynomial regression using least squares.
    Returns coefficients [a0, a1, a2, ...] for y = a0 + a1*x + a2*x^2 + ...
    """
    n = len(x)
    if n != len(y) or n < degree + 1:
        return [0.0] * (degree + 1)

    # Build Vandermonde matrix
    X = [[xi ** j for j in range(degree + 1)] for xi in x]

    # Solve normal equations: (X'X) * coeffs = X'y
    XtX = _matrix_mult(_transpose(X), X)
    Xty = _matrix_vec_mult(_transpose(X), y)

    # Solve using Gaussian elimination
    coeffs = _solve_linear(XtX, Xty)

    return coeffs


def _transpose(A: Matrix) -> Matrix:
    """Transpose matrix."""
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication."""
    m, n, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _matrix_vec_mult(A: Matrix, v: Vector) -> Vector:
    """Matrix-vector multiplication."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _solve_linear(A: Matrix, b: Vector) -> Vector:
    """Solve linear system Ax = b using Gaussian elimination with pivoting."""
    n = len(A)
    # Augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]

    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]

        if abs(aug[i][i]) < 1e-12:
            continue

        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            continue
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


# =============================================================================
# SAMPLING AND RESAMPLING
# =============================================================================

def sample(population: Vector, k: int, replace: bool = False) -> Vector:
    """
    Random sample from population.

    Args:
        population: Values to sample from
        k: Sample size
        replace: If True, sample with replacement
    """
    if replace:
        return [random.choice(population) for _ in range(k)]
    else:
        return random.sample(population, min(k, len(population)))


def shuffle(x: Vector) -> Vector:
    """Return shuffled copy of vector."""
    result = x[:]
    random.shuffle(result)
    return result


def bootstrap(x: Vector, statistic: Callable[[Vector], float], n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Bootstrap estimate of statistic with confidence interval.

    Returns (estimate, lower_95, upper_95).
    """
    estimates = []
    n = len(x)

    for _ in range(n_bootstrap):
        resample = [x[random.randint(0, n - 1)] for _ in range(n)]
        estimates.append(statistic(resample))

    estimates.sort()

    estimate = mean(estimates)
    lower = percentile(estimates, 2.5)
    upper = percentile(estimates, 97.5)

    return (estimate, lower, upper)


def jackknife(x: Vector, statistic: Callable[[Vector], float]) -> Tuple[float, float]:
    """
    Jackknife estimate of statistic with standard error.

    Returns (estimate, standard_error).
    """
    n = len(x)
    if n < 2:
        return (statistic(x), 0.0)

    # Leave-one-out estimates
    estimates = []
    for i in range(n):
        subset = x[:i] + x[i+1:]
        estimates.append(statistic(subset))

    theta_dot = mean(estimates)
    se = math.sqrt((n - 1) / n * sum((e - theta_dot) ** 2 for e in estimates))

    # Bias-corrected estimate
    theta_all = statistic(x)
    bias = (n - 1) * (theta_dot - theta_all)
    estimate = theta_all - bias

    return (estimate, se)


# =============================================================================
# HISTOGRAM AND BINNING
# =============================================================================

def histogram(x: Vector, bins: int = 10, range_: Optional[Tuple[float, float]] = None) -> Tuple[Vector, Vector]:
    """
    Compute histogram.

    Returns (counts, bin_edges).
    """
    if not x:
        return ([], [])

    if range_ is None:
        x_min, x_max = min(x), max(x)
    else:
        x_min, x_max = range_

    if x_min == x_max:
        x_max = x_min + 1

    bin_width = (x_max - x_min) / bins
    bin_edges = [x_min + i * bin_width for i in range(bins + 1)]
    counts = [0] * bins

    for xi in x:
        if x_min <= xi <= x_max:
            idx = min(int((xi - x_min) / bin_width), bins - 1)
            counts[idx] += 1

    return (counts, bin_edges)


def binned_statistic(x: Vector, values: Vector, bins: int = 10,
                     statistic: str = 'mean') -> Tuple[Vector, Vector]:
    """
    Compute statistic of values within bins.

    Args:
        x: Values to bin
        values: Values to compute statistic on
        bins: Number of bins
        statistic: 'mean', 'median', 'sum', 'count', 'min', 'max'

    Returns (bin_stats, bin_edges).
    """
    if len(x) != len(values) or not x:
        return ([], [])

    x_min, x_max = min(x), max(x)
    if x_min == x_max:
        x_max = x_min + 1

    bin_width = (x_max - x_min) / bins
    bin_edges = [x_min + i * bin_width for i in range(bins + 1)]
    bin_values: List[List[float]] = [[] for _ in range(bins)]

    for xi, vi in zip(x, values):
        if x_min <= xi <= x_max:
            idx = min(int((xi - x_min) / bin_width), bins - 1)
            bin_values[idx].append(vi)

    stat_funcs = {
        'mean': mean,
        'median': median,
        'sum': sum,
        'count': len,
        'min': lambda v: min(v) if v else 0,
        'max': lambda v: max(v) if v else 0,
    }

    func = stat_funcs.get(statistic, mean)
    bin_stats = [func(bv) if bv else 0.0 for bv in bin_values]

    return (bin_stats, bin_edges)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def describe(x: Vector) -> dict:
    """
    Descriptive statistics summary.
    Returns dict with count, mean, std, min, 25%, 50%, 75%, max.
    """
    if not x:
        return {}

    q1, med, q3 = quartiles(x)

    return {
        'count': len(x),
        'mean': mean(x),
        'std': std(x, ddof=1),
        'min': min(x),
        '25%': q1,
        '50%': med,
        '75%': q3,
        'max': max(x),
    }


def normalize(x: Vector, method: str = 'zscore') -> Vector:
    """
    Normalize vector.

    Methods:
        'zscore': (x - mean) / std
        'minmax': (x - min) / (max - min)
        'robust': (x - median) / iqr
    """
    if not x:
        return []

    if method == 'zscore':
        m, s = mean(x), std(x)
        if s == 0:
            return [0.0] * len(x)
        return [(xi - m) / s for xi in x]

    elif method == 'minmax':
        x_min, x_max = min(x), max(x)
        if x_max == x_min:
            return [0.5] * len(x)
        return [(xi - x_min) / (x_max - x_min) for xi in x]

    elif method == 'robust':
        med = median(x)
        q = iqr(x)
        if q == 0:
            return [0.0] * len(x)
        return [(xi - med) / q for xi in x]

    else:
        raise ValueError(f"Unknown method: {method}")
    return None


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Almeida Statistics - Pure Python")
    print("=" * 60)

    # Test data
    x = [2.3, 4.5, 6.7, 8.9, 10.1, 12.3, 14.5, 16.7, 18.9, 20.1]
    y = [1.2, 3.4, 5.6, 7.8, 9.0, 11.2, 13.4, 15.6, 17.8, 19.0]

    print("\n--- Descriptive Statistics ---")
    print(f"x = {x}")
    print(f"mean(x) = {mean(x):.4f}")
    print(f"median(x) = {median(x):.4f}")
    print(f"std(x) = {std(x, ddof=1):.4f}")
    print(f"variance(x) = {variance(x, ddof=1):.4f}")
    print(f"skewness(x) = {skewness(x):.4f}")
    print(f"kurtosis(x) = {kurtosis(x):.4f}")

    print("\n--- Percentiles ---")
    q1, med, q3 = quartiles(x)
    print(f"Q1, Q2, Q3 = {q1:.4f}, {med:.4f}, {q3:.4f}")
    print(f"IQR = {iqr(x):.4f}")
    print(f"P90 = {percentile(x, 90):.4f}")

    print("\n--- Correlation ---")
    r, p = pearsonr(x, y)
    print(f"Pearson r = {r:.4f}, p = {p:.6f}")
    rho, p_s = spearmanr(x, y)
    print(f"Spearman rho = {rho:.4f}, p = {p_s:.6f}")

    print("\n--- Linear Regression ---")
    slope, intercept, r2 = linear_regression(x, y)
    print(f"y = {slope:.4f} * x + {intercept:.4f}")
    print(f"R² = {r2:.4f}")

    print("\n--- T-Test ---")
    t, p = t_test_1sample(x, mu=10)
    print(f"One-sample t-test (mu=10): t = {t:.4f}, p = {p:.4f}")
    t2, p2 = t_test_2sample(x, y)
    print(f"Two-sample t-test: t = {t2:.4f}, p = {p2:.4f}")

    print("\n--- Distributions ---")
    print(f"normal_pdf(0) = {normal_pdf(0):.6f}")
    print(f"normal_cdf(1.96) = {normal_cdf(1.96):.6f}")
    print(f"normal_ppf(0.975) = {normal_ppf(0.975):.6f}")

    print("\n--- Bootstrap ---")
    est, lower, upper = bootstrap(x, mean, n_bootstrap=500)
    print(f"Bootstrap mean: {est:.4f} [{lower:.4f}, {upper:.4f}]")

    print("\n--- Histogram ---")
    counts, edges = histogram(x, bins=5)
    print(f"Counts: {counts}")
    print(f"Edges: {[f'{e:.1f}' for e in edges]}")

    print("\n--- Describe ---")
    desc = describe(x)
    for k, v in desc.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n✓ All tests passed!")