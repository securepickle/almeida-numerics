"""
almeida-numerics - a from-scratch numerical stack in pure Python.

Three self-contained, dependency-free modules, each validated against
NumPy/SciPy on random inputs (see the tests/ directory):

    almeida_numerics.tensor   n-dim tensor: dtypes, matmul, NN ops, QR/SVD
    almeida_numerics.linalg   list-based linear algebra: QR/LU/Cholesky/eig/SVD,
                              solve, lstsq, det, inv, pinv, rank, norms
    almeida_numerics.stats    descriptive stats, distributions, hypothesis tests,
                              correlation, regression, resampling

The core of every module uses only the Python standard library. NumPy / PyTorch
are touched only inside the optional interop helpers in `tensor`
(from_numpy / to_numpy / from_torch / to_torch).

    >>> import almeida_numerics as an
    >>> a = an.AlmeidaTensor([[1.0, 2.0], [3.0, 4.0]])
    >>> (a @ a)[0, 0]
    7.0
    >>> an.stats.median([3, 1, 2])
    2.0
    >>> Q, R = an.linalg.qr([[1.0, 2.0], [3.0, 4.0]])

Author: Michael Almeida
Copyright: (c) Almeida Industries
License: Apache-2.0
"""
from __future__ import annotations

from . import tensor, linalg, stats

# Re-export the core tensor API at the top level for convenience.
from .tensor import (
    AlmeidaTensor,
    DType,
    Buffer,
    zeros,
    ones,
    eye,
    randn,
    arange,
    from_list,
    from_numpy,
    to_numpy,
    from_torch,
    to_torch,
    softmax,
    silu,
    gelu,
    rms_norm,
    layer_norm,
    rope_embed,
    scaled_dot_product_attention,
    conv2d,
    qr,
    svd,
    diag,
    tensordot,
)

__version__ = "0.1.0"

__all__ = [
    "tensor", "linalg", "stats",
    "AlmeidaTensor", "DType", "Buffer",
    "zeros", "ones", "eye", "randn", "arange", "from_list",
    "from_numpy", "to_numpy", "from_torch", "to_torch",
    "softmax", "silu", "gelu", "rms_norm", "layer_norm", "rope_embed",
    "scaled_dot_product_attention", "conv2d",
    "qr", "svd", "diag", "tensordot",
    "__version__",
]
