"""
Correctness suite: almeida_tensor vs NumPy (the reference).

A from-scratch numerical library is only trustworthy if its outputs match a
reference on random inputs. This checks every core operation — construction,
elementwise math, matmul, shape ops, reductions, the NN ops, and the linalg
decompositions (by reconstruction) — against NumPy, and reports the actual
max-abs error per op. Runnable standalone (no pytest needed):

    python test_almeida_vs_numpy.py

Exit code 0 iff every check passes. Any FAIL prints the op and the error so it
can be fixed — this suite is both the validation and the bug-finder.
"""
from __future__ import annotations

import numpy as np

from almeida_numerics import tensor as at

RNG = np.random.default_rng(0)
RTOL, ATOL = 1e-4, 1e-5          # fp32 tolerances
_results = []                    # (name, passed, max_err, note)


def _np(t):
    """almeida tensor -> numpy, tolerant of tensor-or-scalar returns."""
    if isinstance(t, at.AlmeidaTensor):
        return at.to_numpy(t)
    return np.asarray(t)


def check(name, got, expected, rtol=RTOL, atol=ATOL):
    g, e = _np(got), np.asarray(expected, dtype=np.float32)
    try:
        g = g.reshape(e.shape) if g.shape != e.shape and g.size == e.size else g
        err = float(np.max(np.abs(g - e))) if e.size else 0.0
        ok = np.allclose(g, e, rtol=rtol, atol=atol)
        note = "" if ok else f"shapes got{g.shape} exp{e.shape}" if g.shape != e.shape else ""
    except Exception as ex:                       # noqa: BLE001 — report, don't crash the run
        ok, err, note = False, float("inf"), f"EXC {type(ex).__name__}: {ex}"
    _results.append((name, ok, err, note))


def rand(*shape):
    a = RNG.standard_normal(shape).astype(np.float32)
    return at.from_numpy(a), a


# ---------------------------------------------------------------- construction
def test_construction():
    check("zeros", at.zeros((3, 4)), np.zeros((3, 4)))
    check("ones", at.ones((2, 5)), np.ones((2, 5)))
    check("eye", at.eye(4), np.eye(4))
    check("arange", at.arange(0, 10, 2), np.arange(0, 10, 2))
    a = RNG.standard_normal((3, 3)).astype(np.float32)
    check("from_numpy/to_numpy roundtrip", at.from_numpy(a), a)


# ---------------------------------------------------------------- elementwise
def test_elementwise():
    x, xn = rand(4, 5); y, yn = rand(4, 5)
    check("add", x + y, xn + yn)
    check("sub", x - y, xn - yn)
    check("mul", x * y, xn * yn)
    check("div", x / (y.abs() + 1.0), xn / (np.abs(yn) + 1.0))
    check("neg", -x, -xn)
    check("scalar radd", 2.0 + x, 2.0 + xn)
    check("scalar rmul", 3.0 * x, 3.0 * xn)
    check("scalar div", x / 2.0, xn / 2.0)
    check("pow", x ** 2, xn ** 2)
    check("abs", x.abs(), np.abs(xn))
    check("sqrt", (x.abs() + 0.1).sqrt(), np.sqrt(np.abs(xn) + 0.1))
    check("exp", x.clamp(-3, 3).exp(), np.exp(np.clip(xn, -3, 3)))
    check("log", (x.abs() + 0.1).log(), np.log(np.abs(xn) + 0.1))
    check("sin", x.sin(), np.sin(xn))
    check("cos", x.cos(), np.cos(xn))
    check("clamp", x.clamp(-0.5, 0.5), np.clip(xn, -0.5, 0.5))


# ---------------------------------------------------------------- matmul / shape
def test_matmul_shape():
    a, an = rand(4, 6); b, bn = rand(6, 3)
    check("matmul 2D", a @ b, an @ bn)
    x, xn = rand(3, 5)
    check("transpose", x.transpose(), xn.T)
    check("reshape", x.reshape((5, 3)), xn.reshape(5, 3))


# ---------------------------------------------------------------- reductions
def test_reductions():
    x, xn = rand(4, 6)
    check("sum all", x.sum(), xn.sum())
    check("sum axis0", x.sum(axis=0), xn.sum(axis=0))
    check("sum axis1 keepdims", x.sum(axis=1, keepdims=True), xn.sum(axis=1, keepdims=True))
    check("mean all", x.mean(), xn.mean())
    check("mean axis0", x.mean(axis=0), xn.mean(axis=0))
    check("max axis1", x.max(axis=1), xn.max(axis=1))
    check("min axis0", x.min(axis=0), xn.min(axis=0))
    check("norm L2", x.norm(), np.linalg.norm(xn))


# ---------------------------------------------------------------- NN ops
def test_nn_ops():
    x, xn = rand(3, 8)
    ex = np.exp(xn - xn.max(axis=-1, keepdims=True))
    check("softmax", at.softmax(x), ex / ex.sum(axis=-1, keepdims=True))
    check("silu", at.silu(x), xn / (1 + np.exp(-xn)))
    check("gelu", at.gelu(x),
          0.5 * xn * (1 + np.tanh(np.sqrt(2/np.pi) * (xn + 0.044715 * xn**3))),
          rtol=2e-3, atol=2e-3)                    # tanh-approx gelu
    w, wn = rand(8)
    rms = xn / np.sqrt((xn**2).mean(axis=-1, keepdims=True) + 1e-6)
    check("rms_norm", at.rms_norm(x, w), rms * wn)


# ---------------------------------------------------------------- linalg (by reconstruction)
def test_linalg():
    a = RNG.standard_normal((5, 5)).astype(np.float32)
    A = at.from_numpy(a)
    Q, R = at.qr(A)
    Qn, Rn = _np(Q), _np(R)
    check("qr reconstruct A=QR", Qn @ Rn, a)
    check("qr Q orthonormal", Qn.T @ Qn, np.eye(5), rtol=1e-3, atol=1e-3)
    # svd() = one-sided Jacobi (deterministic, fp32-accurate) -> tight tolerances
    U, S, Vt = at.svd(A)
    Un, Sn, Vtn = _np(U), _np(S), _np(Vt)
    check("svd reconstruct A=USV^T", Un @ np.diag(Sn.ravel()) @ Vtn, a, rtol=1e-4, atol=1e-4)
    check("svd singular values vs numpy", np.sort(Sn.ravel())[::-1],
          np.linalg.svd(a, compute_uv=False), rtol=1e-4, atol=1e-4)
    check("svd singular values descending", np.diff(Sn.ravel()) <= 1e-5, np.ones(Sn.size - 1))
    # wide matrix exercises the transpose path
    aw = RNG.standard_normal((4, 7)).astype(np.float32)
    Uw, Sw, Vtw = at.svd(at.from_numpy(aw))
    check("svd wide (4x7) reconstruct", _np(Uw) @ np.diag(_np(Sw).ravel()) @ _np(Vtw), aw,
          rtol=1e-4, atol=1e-4)
    # svd_power_iteration remains a valid top-k alternative (looser, iterative)
    Up, Sp, Vtp = at.svd_power_iteration(A)
    check("svd_power_iteration reconstruct", _np(Up) @ np.diag(_np(Sp).ravel()) @ _np(Vtp), a,
          rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------- extras (utility / NN)
def _np_layernorm(xn, wn, bn=None, eps=1e-5):
    mu = xn.mean(axis=-1, keepdims=True)
    var = xn.var(axis=-1, keepdims=True)          # population variance
    out = (xn - mu) / np.sqrt(var + eps) * wn
    return out if bn is None else out + bn


def _np_rope(xn, cosn, sinn, position=0):
    seq, nh, hd = xn.shape
    half = hd // 2
    out = xn.copy()
    for s in range(seq):
        pos = position + s
        for h in range(nh):
            for d in range(half):
                x1, x2 = xn[s, h, d], xn[s, h, d + half]
                c, si = cosn[pos, d], sinn[pos, d]
                out[s, h, d] = x1 * c - x2 * si
                out[s, h, d + half] = x1 * si + x2 * c
    return out


def test_extras():
    # diag: vector -> matrix, matrix -> diagonal vector
    v, vn = rand(5)
    check("diag vec->mat", at.diag(v), np.diag(vn))
    m, mn = rand(4, 6)
    check("diag mat->vec", at.diag(m), np.diag(mn))
    # tensordot: last axis of a with first axis of b (2D and 3D)
    a, an = rand(3, 4); b, bn = rand(4, 5)
    check("tensordot 2D", at.tensordot(a, b, ([-1], [0])), np.tensordot(an, bn, axes=([1], [0])))
    a3, a3n = rand(2, 3, 4)
    check("tensordot 3D", at.tensordot(a3, b, ([-1], [0])), np.tensordot(a3n, bn, axes=([2], [0])))
    # argmin / argmax (flattened index), cumsum (flattened), diff (1D)
    x, xn = rand(4, 5)
    check("argmin", at.argmin(x), int(np.argmin(xn)))
    check("argmax", at.argmax(x), int(np.argmax(xn)))
    check("cumsum", at.cumsum(x), np.cumsum(xn))
    d, dn = rand(9)
    check("diff", at.diff(d), np.diff(dn))
    # polyfit deg=1 -> [slope, intercept]
    xsn = np.linspace(-2, 2, 20).astype(np.float32)
    ysn = (1.7 * xsn - 0.5 + RNG.standard_normal(20).astype(np.float32) * 0.05)
    check("polyfit deg1 [slope,intercept]",
          at.polyfit(at.from_numpy(xsn), at.from_numpy(ysn), 1), np.polyfit(xsn, ysn, 1))
    # searchsorted (left)
    base = np.array([1., 3., 5., 7., 9.], dtype=np.float32)
    srt = at.from_numpy(base)
    for val in (0.5, 4.0, 5.0, 10.0):
        check(f"searchsorted({val})", at.searchsorted(srt, val), int(np.searchsorted(base, val)))
    # squeeze
    sq, sqn = rand(1, 4, 1)
    check("squeeze all", at.squeeze(sq), np.squeeze(sqn))
    check("squeeze axis0", at.squeeze(sq, axis=0), np.squeeze(sqn, axis=0))
    # general N-D transpose (odometer path)
    t3, t3n = rand(2, 3, 4)
    check("transpose (2,0,1)", t3.transpose((2, 0, 1)), np.transpose(t3n, (2, 0, 1)))
    check("transpose (1,2,0)", t3.transpose((1, 2, 0)), np.transpose(t3n, (1, 2, 0)))
    # scaled dot-product attention (matmul -> softmax -> matmul)
    q, qn = rand(4, 8); kk, kn = rand(5, 8); vv, vn = rand(5, 6)
    s = (qn @ kn.T) / np.sqrt(8)
    e = np.exp(s - s.max(axis=-1, keepdims=True)); w = e / e.sum(axis=-1, keepdims=True)
    check("attention softmax(QK^T/sqrt d)V", at.scaled_dot_product_attention(q, kk, vv), w @ vn,
          rtol=2e-3, atol=2e-3)
    # conv2d (valid cross-correlation) via im2col + inner product
    ix, ixn = rand(9, 11); kern, kernn = rand(3, 4)
    oh, ow = 9 - 3 + 1, 11 - 4 + 1
    ref = np.array([[(ixn[i:i + 3, j:j + 4] * kernn).sum() for j in range(ow)] for i in range(oh)])
    check("conv2d valid", at.conv2d(ix, kern), ref, rtol=1e-4, atol=1e-4)
    # layer_norm: 1D and 2D, with and without bias
    x1, x1n = rand(8); w, wn = rand(8); bias, biasn = rand(8)
    check("layer_norm 1D no-bias", at.layer_norm(x1, w), _np_layernorm(x1n, wn))
    check("layer_norm 1D +bias", at.layer_norm(x1, w, bias), _np_layernorm(x1n, wn, biasn))
    x2, x2n = rand(4, 8)
    check("layer_norm 2D +bias", at.layer_norm(x2, w, bias), _np_layernorm(x2n, wn, biasn))
    # rope_embed: (seq, n_heads, head_dim)
    xr, xrn = rand(3, 2, 8)
    cosn = RNG.standard_normal((3, 8)).astype(np.float32)
    sinn = RNG.standard_normal((3, 8)).astype(np.float32)
    check("rope_embed", at.rope_embed(xr, at.from_numpy(cosn), at.from_numpy(sinn)),
          _np_rope(xrn, cosn, sinn))


def main():
    for fn in (test_construction, test_elementwise, test_matmul_shape,
               test_reductions, test_nn_ops, test_linalg, test_extras):
        try:
            fn()
        except Exception as ex:                    # noqa: BLE001
            _results.append((fn.__name__, False, float("inf"), f"SUITE-EXC {ex}"))
    npass = sum(1 for _, ok, _, _ in _results if ok)
    print(f"\nalmeida_tensor vs NumPy — {npass}/{len(_results)} checks passed\n")
    for name, ok, err, note in _results:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name:<28} max_err={err:.2e}"
        if note:
            line += f"   {note}"
        print(line)
    failed = [n for n, ok, _, _ in _results if not ok]
    if failed:
        print(f"\n{len(failed)} FAILED: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
