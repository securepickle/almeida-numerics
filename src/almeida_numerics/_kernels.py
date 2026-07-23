"""
Internal raw-buffer vector kernels - the "unchecked algebra".

These operate directly on flat storage (an `array.array`, or any indexable /
sliceable sequence such as a Python list) addressed by (start, stride, length),
and perform NO bounds or shape checking. Callers - the AlmeidaTensor methods and
the linalg routines - establish bounds once at the algorithm level, then spend
them here without paying the per-element __getitem__ / _multi_to_flat tax that
dominated profiling.

Design contract:
  * Each kernel is called ONCE PER VECTOR / column, never once per element, so
    its own Python call overhead is amortized over `length` elements.
  * A stride == 1 fast path uses contiguous slicing, which is meaningfully
    faster than manual index stepping in CPython; the general strided path lets
    a caller walk a matrix row or column in place, with no gather.

Container choice (semantically interchangeable, NOT performance-equivalent):
    array.array  -> typed, compact PERSISTENT tensor storage
    list[float]  -> high-speed TRANSIENT arithmetic workspace
CPython list ops (build/iterate/slice-assign) are faster than the equivalent
array.array ops, so an algorithm should gather the public row-major tensor into
a workspace whose *layout and container* suit the algorithm (e.g. QR uses a
column-major list), then run these kernels over it. Benchmark and choose
deliberately; both containers work through the same kernels.

Aliasing contract (this is an UNCHECKED layer):
  * In-place kernels (`scale`, `axpy`, `rot`) require that a destination region
    and any source region are either identical or NON-overlapping. Partial
    overlap at different offsets is outside the supported domain: the strided
    `axpy`/`scale` paths write while reading, so partial aliasing would corrupt
    results. (QR/SVD callers use disjoint columns, so this holds.)

This module is private (leading underscore) and not part of the public API.
"""
import array as _array
import math


def _store(dst, s, n, values):
    """Bulk-assign `values` into dst[s:s+n], matching dst's container type."""
    if isinstance(dst, _array.array):
        dst[s:s + n] = _array.array(dst.typecode, values)
    else:
        dst[s:s + n] = values


def dot(a, sa, ta, b, sb, tb, n):
    """Sum over i of a[sa + i*ta] * b[sb + i*tb]."""
    if ta == 1 and tb == 1:
        acc = 0.0
        for x, y in zip(a[sa:sa + n], b[sb:sb + n]):
            acc += x * y
        return acc
    acc = 0.0
    ia, ib = sa, sb
    for _ in range(n):
        acc += a[ia] * b[ib]
        ia += ta
        ib += tb
    return acc


def norm2(a, s, t, n):
    """Euclidean (L2) norm of the strided vector a[s : s + n*t : t]."""
    acc = 0.0
    if t == 1:
        for v in a[s:s + n]:
            acc += v * v
    else:
        i = s
        for _ in range(n):
            v = a[i]
            acc += v * v
            i += t
    return math.sqrt(acc)


def scale(a, s, t, n, alpha):
    """In-place: a[s : s + n*t : t] *= alpha."""
    if t == 1:
        # Bulk slice-assignment beats a per-element loop in CPython.
        _store(a, s, n, [v * alpha for v in a[s:s + n]])
    else:
        i = s
        for _ in range(n):
            a[i] *= alpha
            i += t


def axpy(y, sy, ty, x, sx, tx, alpha, n):
    """In-place: y[...] += alpha * x[...]  (the BLAS 'axpy')."""
    if ty == 1 and tx == 1:
        xs = x[sx:sx + n]
        ys = y[sy:sy + n]
        _store(y, sy, n, [b + alpha * a for a, b in zip(xs, ys)])
    else:
        iy, ix = sy, sx
        for _ in range(n):
            y[iy] += alpha * x[ix]
            iy += ty
            ix += tx


def rot(a, si, sj, c, s, n):
    """In-place Givens rotation of two contiguous length-n vectors in `a`:
        col_i, col_j  <-  c*col_i - s*col_j,  s*col_i + c*col_j
    Used by one-sided Jacobi SVD to orthogonalize a pair of columns."""
    ci = a[si:si + n]
    cj = a[sj:sj + n]
    _store(a, si, n, [c * x - s * y for x, y in zip(ci, cj)])
    _store(a, sj, n, [s * x + c * y for x, y in zip(ci, cj)])


# --- BLAS-2: fused matrix-vector kernels (a is row-major m x n) -----------------
# These read the vector operand ONCE and loop the rows internally, avoiding the
# repeated operand-slicing a per-row dot()/axpy() would pay across many rows.

def gemv(a, m, n, x, out):
    """out[i] = sum_j a[i*n + j] * x[j]   (out = A @ x). Overwrites out[0:m]."""
    xs = x if isinstance(x, list) else x[:]        # snapshot the shared operand once
    for i in range(m):
        base = i * n
        acc = 0.0
        for av, xv in zip(a[base:base + n], xs):
            acc += av * xv
        out[i] = acc


def gemv_t(a, m, n, x, out):
    """out[j] = sum_i a[i*n + j] * x[i]   (out = A^T @ x). Overwrites out[0:n]."""
    for j in range(n):
        out[j] = 0.0
    for i in range(m):
        xi = x[i]
        if xi == 0.0:
            continue
        base = i * n
        _store(out, 0, n, [o + xi * av for o, av in zip(out, a[base:base + n])])


def ger_sub(a, m, n, alpha, u, v):
    """Rank-1 update in place: A -= alpha * u (x) v^T   (A row-major m x n)."""
    vs = v if isinstance(v, list) else v[:]
    for i in range(m):
        au = alpha * u[i]
        if au == 0.0:
            continue
        base = i * n
        _store(a, base, n, [rv - au * vv for rv, vv in zip(a[base:base + n], vs)])
