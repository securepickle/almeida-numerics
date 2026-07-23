"""
Correctness suite: almeida_linalg vs NumPy (the reference).

almeida_linalg is a from-scratch, dependency-free linear-algebra library
operating on plain Python lists (Vector = list[float], Matrix = list[list]).
This checks every core operation against NumPy on random inputs and reports the
actual max-abs error per op. Decompositions (QR, LU, Cholesky, eig, SVD) are
validated by reconstruction and by their defining invariants (orthonormality,
triangularity, PA=LU), which is basis-independent and the honest way to check an
iterative solver. Runnable standalone (no pytest needed):

    python test_almeida_linalg_vs_numpy.py

Exit code 0 iff every check passes. Iterative routines (power-method eig/SVD)
use looser, clearly-labelled tolerances than the closed-form ops.
"""
from __future__ import annotations

import numpy as np

from almeida_numerics import linalg as la

RNG = np.random.default_rng(0)
_results = []                    # (name, passed, max_err, note)


def check(name, got, expected, rtol=1e-5, atol=1e-6, note=""):
    try:
        g = np.asarray(got, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)
        g = g.reshape(e.shape) if g.shape != e.shape and g.size == e.size else g
        err = float(np.max(np.abs(g - e))) if e.size else 0.0
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
        if not ok and note == "" and g.shape != e.shape:
            note = f"shapes got{g.shape} exp{e.shape}"
    except Exception as ex:                       # noqa: BLE001 — report, don't crash
        ok, err, note = False, float("inf"), f"EXC {type(ex).__name__}: {ex}"
    _results.append((name, ok, err, note))


def randm(r, c):
    a = RNG.standard_normal((r, c))
    return a.tolist(), a


def randv(n):
    a = RNG.standard_normal(n)
    return a.tolist(), a


def spd(n):
    """A random symmetric positive-definite matrix."""
    a = RNG.standard_normal((n, n))
    m = a @ a.T + n * np.eye(n)
    return m.tolist(), m


# ---------------------------------------------------------------- basic ops
def test_basic():
    check("zeros", la.zeros(3, 4), np.zeros((3, 4)))
    check("eye", la.eye(5), np.eye(5))
    A, An = randm(4, 6)
    check("copy_matrix", la.copy_matrix(A), An)
    check("transpose", la.transpose(A), An.T)
    B, Bn = randm(6, 3)
    check("matmul", la.matmul(A, B), An @ Bn)
    x, xn = randv(6)
    check("matvec", la.matvec(A, x), An @ xn)
    y, yn = randv(6)
    check("dot", la.dot(x, y), xn @ yn)
    check("outer", la.outer(x, y), np.outer(xn, yn))
    check("scale_vector", la.scale_vector(2.5, x), 2.5 * xn)
    check("add_vectors", la.add_vectors(x, y), xn + yn)
    check("sub_vectors", la.sub_vectors(x, y), xn - yn)
    S, Sn = spd(5)
    check("trace", la.trace(S), np.trace(Sn))


# ---------------------------------------------------------------- norms
def test_norms():
    x, xn = randv(8)
    check("vector_norm p=2", la.vector_norm(x, 2.0), np.linalg.norm(xn, 2))
    check("vector_norm p=1", la.vector_norm(x, 1.0), np.linalg.norm(xn, 1))
    A, An = randm(4, 5)
    check("frobenius_norm", la.frobenius_norm(A), np.linalg.norm(An, "fro"))
    check("matrix_norm fro", la.matrix_norm(A, "fro"), np.linalg.norm(An, "fro"))
    check("norm(vector)", la.norm(x), np.linalg.norm(xn))


# ---------------------------------------------------------------- QR
def test_qr():
    A, An = randm(5, 5)
    for method in ("householder", "gram_schmidt"):
        Q, R = la.qr(A, method=method)
        Qn, Rn = np.asarray(Q), np.asarray(R)
        check(f"qr[{method}] reconstruct A=QR", Qn @ Rn, An)
        check(f"qr[{method}] Q orthonormal", Qn.T @ Qn, np.eye(5), rtol=1e-4, atol=1e-4)
        check(f"qr[{method}] R upper-tri", np.tril(Rn, -1), np.zeros((5, 5)), atol=1e-6)


# ---------------------------------------------------------------- LU / Cholesky
def test_lu_cholesky():
    A, An = randm(5, 5)
    L, U, perm = la.lu(A)
    Ln, Un = np.asarray(L), np.asarray(U)
    check("lu PA=LU", Ln @ Un, An[perm, :])
    check("lu L lower-tri", np.triu(Ln, 1), np.zeros((5, 5)), atol=1e-9)
    check("lu U upper-tri", np.tril(Un, -1), np.zeros((5, 5)), atol=1e-9)
    Am, Amn = spd(5)
    Lc = np.asarray(la.cholesky(Am))
    check("cholesky A=LL^T", Lc @ Lc.T, Amn)
    check("cholesky L lower-tri", np.triu(Lc, 1), np.zeros((5, 5)), atol=1e-9)


# ---------------------------------------------------------------- solve / lstsq / inv / det
def test_solve():
    A, An = spd(5)                                 # well-conditioned
    b, bn = randv(5)
    check("solve Ax=b", la.solve(A, b), np.linalg.solve(An, bn))
    check("inv", la.inv(A), np.linalg.inv(An))
    check("det", la.det(A), np.linalg.det(An), rtol=1e-4, atol=1e-4)
    check("matrix_power A^3", la.matrix_power(A, 3), np.linalg.matrix_power(An, 3), rtol=1e-4, atol=1e-3)
    # overdetermined least squares
    M, Mn = randm(8, 3)
    c, cn = randv(8)
    check("lstsq", la.lstsq(M, c), np.linalg.lstsq(Mn, cn, rcond=None)[0])
    check("pinv", la.pinv(M), np.linalg.pinv(Mn), rtol=1e-3, atol=1e-4)
    check("rank (full)", la.rank(An), np.linalg.matrix_rank(An))


# ---------------------------------------------------------------- eig / svd (iterative)
def test_eig_svd():
    A, An = spd(5)                                 # symmetric => real eigenvalues
    evals, _ = la.eig(A, method="qr")
    check("eig eigenvalues (sorted)", sorted(evals), sorted(np.linalg.eigvalsh(An).tolist()),
          rtol=1e-3, atol=1e-3, note="power/QR iteration")
    B, Bn = randm(5, 5)
    U, S, Vt = la.svd(B)
    Un, Sn, Vtn = np.asarray(U), np.asarray(S).ravel(), np.asarray(Vt)
    recon = Un @ np.diag(Sn) @ Vtn
    check("svd reconstruct A=USV^T", recon, Bn, rtol=1e-2, atol=1e-2, note="power iteration")
    check("svd singular values match", sorted(Sn.tolist(), reverse=True),
          sorted(np.linalg.svd(Bn, compute_uv=False).tolist(), reverse=True),
          rtol=1e-2, atol=1e-2, note="power iteration")


def main():
    for fn in (test_basic, test_norms, test_qr, test_lu_cholesky, test_solve, test_eig_svd):
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            _results.append((fn.__name__, False, float("inf"), f"SUITE-EXC {ex}"))
    npass = sum(1 for _, ok, _, _ in _results if ok)
    print(f"\nalmeida_linalg vs NumPy — {npass}/{len(_results)} checks passed\n")
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
