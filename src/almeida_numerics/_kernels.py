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
