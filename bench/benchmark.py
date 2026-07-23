"""
Benchmark harness for almeida-numerics.

Times representative operations against NumPy for scale, with SEEDED inputs
(both NumPy and Python's `random`, so SVD power-iteration counts are stable
run-to-run). Optional `--profile` runs a cProfile pass to surface the hottest
functions.

    python bench/benchmark.py            # timing table
    python bench/benchmark.py --profile  # + cProfile hot functions

Requires the dev extras (numpy) purely for the reference column:
    pip install -e ".[test]"

This is a development tool; it is not part of the importable package and the
library core still has no runtime dependencies.
"""
import random
import sys
import time

import numpy as np

import almeida_numerics as an
from almeida_numerics import tensor as at
from almeida_numerics import linalg as la

SEED = 0


def _reseed():
    """Deterministic inputs AND deterministic SVD power-iteration init."""
    random.seed(SEED)
    return np.random.default_rng(SEED)


RNG = _reseed()


def bench(fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def mat(r, c):
    a = RNG.standard_normal((r, c)).astype(np.float32)
    return at.from_numpy(a), a


def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"{'op':<26}{'almeida (ms)':>15}{'numpy (ms)':>15}{'slowdown':>12}")


def line(name, af, nf):
    ta, tn = bench(af), bench(nf)
    ratio = ta / tn if tn > 0 else float("inf")
    print(f"{name:<26}{ta*1e3:>15.4f}{tn*1e3:>15.5f}{ratio:>11.0f}x")


def _np_softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def _np_rms(x, w):
    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6) * w


def _np_ln(x, w, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True); var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w


def timings():
    section("Matmul (irreducible O(n^3) Python; already on raw buffers)")
    for n in (32, 64, 128):
        A, An = mat(n, n); B, Bn = mat(n, n)
        line(f"matmul {n}x{n}", lambda A=A, B=B: A @ B, lambda An=An, Bn=Bn: An @ Bn)

    section("Elementwise / reductions (128x128)")
    A, An = mat(128, 128); B, Bn = mat(128, 128)
    line("add", lambda: A + B, lambda: An + Bn)
    line("exp", lambda: A.exp(), lambda: np.exp(An))
    line("sum(all)", lambda: A.sum(), lambda: An.sum())
    line("sum(axis=0)", lambda: A.sum(axis=0), lambda: An.sum(axis=0))
    line("sum(axis=1)", lambda: A.sum(axis=1), lambda: An.sum(axis=1))
    line("transpose .T", lambda: A.T, lambda: An.T)

    section("NN ops (64x256)")
    X, Xn = mat(64, 256); wn = RNG.standard_normal(256).astype(np.float32); w = at.from_numpy(wn)
    line("softmax", lambda: at.softmax(X), lambda: _np_softmax(Xn))
    line("rms_norm", lambda: at.rms_norm(X, w), lambda: _np_rms(Xn, wn))
    line("layer_norm", lambda: at.layer_norm(X, w), lambda: _np_ln(Xn, wn))
    xr = at.from_numpy(RNG.standard_normal((16, 8, 64)).astype(np.float32))
    cc = at.from_numpy(RNG.standard_normal((16, 64)).astype(np.float32))
    ss = at.from_numpy(RNG.standard_normal((16, 64)).astype(np.float32))
    line("rope_embed 16x8x64", lambda: at.rope_embed(xr, cc, ss), lambda: None or 0)

    section("Decompositions")
    for n in (32, 64):
        A, An = mat(n, n)
        line(f"qr {n}x{n}", lambda A=A: at.qr(A), lambda An=An: np.linalg.qr(An))
    A, An = mat(32, 32)
    line("svd 32x32", lambda A=A: at.svd(A), lambda An=An: np.linalg.svd(An))

    section("linalg (list-based, 32x32)")
    Al = An.tolist(); bl = RNG.standard_normal(32).tolist()
    line("la.solve", lambda: la.solve(Al, bl), lambda: np.linalg.solve(np.array(Al), np.array(bl)))


def run_profile(top=20):
    import cProfile, pstats, io
    print(f"\n{'='*72}\ncProfile — hottest functions by tottime\n{'='*72}")
    A, _ = mat(96, 96); B, _ = mat(96, 96)
    X, _ = mat(64, 256); w = at.from_numpy(RNG.standard_normal(256).astype(np.float32))
    Q, _ = mat(48, 48)

    def bundle():
        for _ in range(20):
            _ = A @ B; _ = at.softmax(X); _ = at.rms_norm(X, w); _ = at.layer_norm(X, w)
            _ = X.sum(axis=0)
        for _ in range(6):
            _ = at.qr(Q)

    pr = cProfile.Profile(); pr.enable(); bundle(); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    for ln in s.getvalue().splitlines():
        if "almeida_numerics" in ln or "function calls" in ln or ln.strip().startswith("ncalls"):
            print(ln)


if __name__ == "__main__":
    timings()
    if "--profile" in sys.argv:
        run_profile()
