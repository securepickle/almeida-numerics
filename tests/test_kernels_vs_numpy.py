"""
Correctness suite: internal raw-buffer kernels vs NumPy.

The kernels (almeida_numerics._kernels) are the unchecked internal algebra QR /
SVD / reductions are built on. They are exercised transitively by the tensor
suite, but validating them directly - including the strided path and both
`array` and `list` storage - pins the contract down. Runnable standalone:

    python tests/test_kernels_vs_numpy.py

Exit code 0 iff every check passes.
"""
from __future__ import annotations

from array import array

import numpy as np

from almeida_numerics import _kernels as k

RNG = np.random.default_rng(0)
_results = []


def check(name, got, expected, rtol=1e-6, atol=1e-8, note=""):
    try:
        g = np.asarray(got, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)
        err = float(np.max(np.abs(g - e))) if e.size else 0.0
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
    except Exception as ex:                       # noqa: BLE001
        ok, err, note = False, float("inf"), f"EXC {type(ex).__name__}: {ex}"
    _results.append((name, ok, err, note))


def test_dot():
    an = RNG.standard_normal(12); bn = RNG.standard_normal(12)
    a, b = array("d", an), array("d", bn)
    check("dot contiguous (array)", k.dot(a, 0, 1, b, 0, 1, 12), an @ bn)
    check("dot contiguous (list)", k.dot(list(an), 0, 1, list(bn), 0, 1, 12), an @ bn)
    # strided: every other element, length 6
    check("dot strided t=2", k.dot(a, 0, 2, b, 1, 2, 6), an[0:12:2] @ bn[1:12:2])
    # mixed array x list
    check("dot array x list", k.dot(a, 2, 1, list(bn), 3, 1, 5), an[2:7] @ bn[3:8])


def test_norm2():
    vn = RNG.standard_normal(20); v = array("d", vn)
    check("norm2 contiguous", k.norm2(v, 0, 1, 20), np.linalg.norm(vn))
    check("norm2 strided t=4", k.norm2(v, 0, 4, 5), np.linalg.norm(vn[0:20:4]))
    check("norm2 list", k.norm2(list(vn), 3, 1, 10), np.linalg.norm(vn[3:13]))


def test_scale():
    vn = RNG.standard_normal(10); v = array("d", vn)
    k.scale(v, 0, 1, 10, 2.5)
    check("scale contiguous", list(v), vn * 2.5)
    vn2 = RNG.standard_normal(10); v2 = array("d", vn2)
    k.scale(v2, 0, 2, 5, -3.0)                     # scale even indices only
    exp = vn2.copy(); exp[0:10:2] *= -3.0
    check("scale strided t=2", list(v2), exp)


def test_axpy():
    xn = RNG.standard_normal(15); yn = RNG.standard_normal(15)
    x, y = array("d", xn), array("d", yn)
    k.axpy(y, 0, 1, x, 0, 1, 1.7, 15)
    check("axpy contiguous", list(y), yn + 1.7 * xn)
    # strided into a list target
    xn2 = RNG.standard_normal(10); yn2 = RNG.standard_normal(10)
    yl = list(yn2)
    k.axpy(yl, 1, 2, list(xn2), 0, 2, -0.5, 5)     # y[1,3,5,7,9] += -0.5 * x[0,2,4,6,8]
    exp = yn2.copy(); exp[1:10:2] += -0.5 * xn2[0:10:2]
    check("axpy strided (list)", yl, exp)


def test_rot():
    import math
    ci = RNG.standard_normal(8); cj = RNG.standard_normal(8)
    # pack two columns contiguously into one buffer
    a = array("d", list(ci) + list(cj))
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    k.rot(a, 0, 8, c, s, 8)
    check("rot col_i", a[0:8], c * ci - s * cj)
    check("rot col_j", a[8:16], s * ci + c * cj)


def test_blas2():
    m, n = 6, 4
    An = RNG.standard_normal((m, n))
    a = array("d", An.ravel())                    # row-major
    xn = RNG.standard_normal(n)
    out = [0.0] * m
    k.gemv(a, m, n, list(xn), out)
    check("gemv out=A@x", out, An @ xn)
    yn = RNG.standard_normal(m)
    outt = [0.0] * n
    k.gemv_t(a, m, n, list(yn), outt)
    check("gemv_t out=A^T@x", outt, An.T @ yn)
    # rank-1 update: A -= alpha * u (x) v^T
    un = RNG.standard_normal(m); vn = RNG.standard_normal(n); alpha = 1.3
    a2 = array("d", An.ravel())
    k.ger_sub(a2, m, n, alpha, list(un), list(vn))
    check("ger_sub A-=a*u(x)v", list(a2), (An - alpha * np.outer(un, vn)).ravel())


def main():
    for fn in (test_dot, test_norm2, test_scale, test_axpy, test_rot, test_blas2):
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            _results.append((fn.__name__, False, float("inf"), f"SUITE-EXC {ex}"))
    npass = sum(1 for _, ok, _, _ in _results if ok)
    print(f"\n_kernels vs NumPy - {npass}/{len(_results)} checks passed\n")
    for name, ok, err, note in _results:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name:<28} max_err={err:.2e}"
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
