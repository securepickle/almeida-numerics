"""
================================================================================
almeida-numerics: tensor - a from-scratch, pure-Python tensor library
================================================================================

An n-dimensional tensor built directly on Python's `array`/`struct` - no NumPy,
no external dependencies in the core. Custom dtypes (fp32/fp16/bf16), broadcast
elementwise math, matmul, reductions, common NN ops, and QR/SVD by hand.

Validated against NumPy: see tests/test_tensor_vs_numpy.py (QR and SVD
reconstruct to fp32 precision). NumPy/PyTorch are used only inside the optional
`from_numpy`/`to_numpy`/`from_torch`/`to_torch` interop helpers - import them if
you have them; the library itself runs without them.

Author: Michael Almeida
Copyright: (c) Almeida Industries
License: Apache-2.0
================================================================================
"""

from __future__ import annotations
from typing import Union, Tuple, List, Optional, Callable, Iterator, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import array
import math
import struct
import random


# =============================================================================
# DATA TYPES
# =============================================================================

class DType(Enum):
    """Supported data types for Almeida Tensors."""
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"

    @property
    def typecode(self) -> str:
        """Get Python array typecode."""
        codes = {
            DType.FLOAT32: 'f',
            DType.FLOAT64: 'd',
            DType.INT8: 'b',
            DType.INT16: 'h',
            DType.INT32: 'i',
            DType.INT64: 'q',
            DType.UINT8: 'B',
        }
        # FLOAT16 and BFLOAT16 stored as bytes, converted on access
        return codes.get(self, 'f')

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        sizes = {
            DType.FLOAT16: 2,
            DType.BFLOAT16: 2,
            DType.FLOAT32: 4,
            DType.FLOAT64: 8,
            DType.INT8: 1,
            DType.INT16: 2,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.UINT8: 1,
        }
        return sizes[self]


# =============================================================================
# BUFFER CLASS
# =============================================================================

class Buffer:
    """
    Low-level memory buffer for tensor data.

    Uses Python's array module for memory efficiency.
    """

    __slots__ = ('_data', '_dtype', '_size')

    def __init__(self, size: int, dtype: DType = DType.FLOAT32):
        self._dtype = dtype
        self._size = size
        self._data = array.array(dtype.typecode, [0] * size)

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= self._size:
            raise IndexError(f"Buffer index {index} out of range [0, {self._size})")
        return float(self._data[index])

    def __setitem__(self, index: int, value: float):
        if index < 0 or index >= self._size:
            raise IndexError(f"Buffer index {index} out of range [0, {self._size})")
        self._data[index] = value

    def __len__(self) -> int:
        return self._size

    def copy(self) -> 'Buffer':
        """Create a copy of the buffer."""
        new_buf = Buffer(self._size, self._dtype)
        for i in range(self._size):
            new_buf._data[i] = self._data[i]
        return new_buf

    def as_bytes(self) -> bytes:
        """Get raw bytes."""
        return self._data.tobytes()

    def from_bytes(self, data: bytes):
        """Load from raw bytes."""
        self._data = array.array(self._dtype.typecode)
        self._data.frombytes(data)
        self._size = len(self._data)


# =============================================================================
# ALMEIDA TENSOR CLASS
# =============================================================================

class AlmeidaTensor:
    """
    Pure Python n-dimensional tensor.

    Core design principles:
    1. Shape and dtype are immutable after creation
    2. Operations return new tensors (functional style)
    3. No external dependency in the core (array/struct/math only)

    Example:
        >>> a = AlmeidaTensor([[1, 2], [3, 4]])
        >>> b = AlmeidaTensor([[5, 6], [7, 8]])
        >>> c = a @ b
        >>> print(c[0, 0])  # 19.0
    """

    __slots__ = ('_buffer', '_shape', '_dtype', '_strides')

    def __init__(
        self,
        data: Union[list, tuple, 'AlmeidaTensor', None] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: DType = DType.FLOAT32
    ):
        self._dtype = dtype

        if data is not None:
            if isinstance(data, AlmeidaTensor):
                self._shape = data._shape
                self._buffer = data._buffer.copy()
            else:
                self._shape, flat_data = self._parse_input(data)
                self._buffer = Buffer(len(flat_data), dtype)
                for i, v in enumerate(flat_data):
                    self._buffer[i] = float(v)
        elif shape is not None:
            self._shape = shape
            size = 1
            for dim in shape:
                size *= dim
            self._buffer = Buffer(size, dtype)
        else:
            raise ValueError("Must provide either data or shape")

        self._strides = self._compute_strides(self._shape)

    def _parse_input(self, data) -> Tuple[Tuple[int, ...], List[float]]:
        """Parse nested list/tuple into shape and flat data."""
        if not isinstance(data, (list, tuple)):
            return (), [float(data)]

        if len(data) == 0:
            return (0,), []

        # Check if first element is also a list/tuple
        if isinstance(data[0], (list, tuple)):
            # Nested - recurse
            shapes = []
            flat = []
            for item in data:
                item_shape, item_flat = self._parse_input(item)
                shapes.append(item_shape)
                flat.extend(item_flat)

            # Verify all sub-shapes are identical
            if not all(s == shapes[0] for s in shapes):
                raise ValueError("Inconsistent shapes in nested data")

            return (len(data),) + shapes[0], flat
        else:
            # Base case - list of scalars
            return (len(data),), [float(x) for x in data]

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute strides for row-major layout."""
        if len(shape) == 0:
            return ()
        strides = []
        acc = 1
        for dim in reversed(shape):
            strides.append(acc)
            acc *= dim
        return tuple(reversed(strides))

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def T(self) -> 'AlmeidaTensor':
        """Transpose (reverses all axes)."""
        return self.transpose()

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def _flat_to_multi(self, flat_idx: int) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional indices."""
        indices = []
        for stride in self._strides:
            indices.append(flat_idx // stride)
            flat_idx %= stride
        return tuple(indices)

    def _multi_to_flat(self, indices: Tuple[int, ...]) -> int:
        """Convert multi-dimensional indices to flat index."""
        flat = 0
        for idx, stride in zip(indices, self._strides):
            flat += idx * stride
        return flat

    def _get_flat_index(self, indices: List[int]) -> float:
        """Get value at multi-dimensional indices (list version)."""
        flat = 0
        for idx, stride in zip(indices, self._strides):
            flat += idx * stride
        return self._buffer[flat]

    def _set_flat_index(self, indices: List[int], value: float) -> None:
        """Set value at multi-dimensional indices (list version)."""
        flat = 0
        for idx, stride in zip(indices, self._strides):
            flat += idx * stride
        self._buffer[flat] = value

    def __getitem__(self, key):
        """Get element or slice."""
        if isinstance(key, int):
            if self.ndim == 1:
                return self._buffer[key]
            else:
                # Return slice along first dimension
                new_shape = self._shape[1:]
                result = AlmeidaTensor(shape=new_shape, dtype=self._dtype)
                start = key * self._strides[0]
                for i in range(result.size):
                    result._buffer[i] = self._buffer[start + i]
                return result
        elif isinstance(key, tuple):
            if len(key) == self.ndim:
                # Single element
                flat = self._multi_to_flat(key)
                return self._buffer[flat]
            else:
                raise NotImplementedError("Partial indexing not yet supported")
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    def __setitem__(self, key, value):
        """Set element."""
        if isinstance(key, int):
            if self.ndim == 1:
                self._buffer[key] = float(value)
            else:
                raise NotImplementedError("Slice assignment not yet supported")
        elif isinstance(key, tuple):
            if len(key) == self.ndim:
                flat = self._multi_to_flat(key)
                self._buffer[flat] = float(value)
            else:
                raise NotImplementedError("Partial assignment not yet supported")
        else:
            raise TypeError(f"Unsupported index type: {type(key)}")

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self) -> 'AlmeidaTensor':
        """Create a deep copy."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        result._buffer = self._buffer.copy()
        return result

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def _broadcast_shapes(self, other: 'AlmeidaTensor') -> Tuple[Tuple[int, ...], List[int], List[int]]:
        """
        Compute broadcast output shape and index mappings.

        Returns:
            (output_shape, self_strides, other_strides)
            where strides map output index to input buffer index.
        """
        # Pad shapes to same length
        ndim = max(len(self._shape), len(other._shape))
        shape_a = (1,) * (ndim - len(self._shape)) + self._shape
        shape_b = (1,) * (ndim - len(other._shape)) + other._shape

        # Compute output shape
        out_shape = []
        for a, b in zip(shape_a, shape_b):
            if a == b:
                out_shape.append(a)
            elif a == 1:
                out_shape.append(b)
            elif b == 1:
                out_shape.append(a)
            else:
                raise ValueError(f"Cannot broadcast shapes {self._shape} and {other._shape}")

        return tuple(out_shape), shape_a, shape_b

    def _broadcast_index(self, out_idx: int, out_shape: Tuple[int, ...],
                         in_shape: Tuple[int, ...], in_strides: Tuple[int, ...]) -> int:
        """Map output flat index to input flat index with broadcasting."""
        # Convert flat index to multi-index
        coords = []
        remaining = out_idx
        for dim in reversed(out_shape):
            coords.append(remaining % dim)
            remaining //= dim
        coords = coords[::-1]

        # Map to input coords (broadcast dims collapse to 0)
        in_coords = []
        offset = len(out_shape) - len(in_shape)
        for i, (out_d, in_d) in enumerate(zip(out_shape[offset:], in_shape)):
            if in_d == 1:
                in_coords.append(0)  # Broadcast: always index 0
            else:
                in_coords.append(coords[i + offset])

        # Convert to flat index
        flat_idx = 0
        for i, c in enumerate(in_coords):
            flat_idx += c * in_strides[i]
        return flat_idx

    def _apply_binary_op(self, other: 'AlmeidaTensor', op) -> 'AlmeidaTensor':
        """Apply binary operation with broadcasting."""
        if self._shape == other._shape:
            # Fast path: same shape
            result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
            for i in range(self.size):
                result._buffer[i] = op(self._buffer[i], other._buffer[i])
            return result

        # Broadcast path
        out_shape, shape_a, shape_b = self._broadcast_shapes(other)

        # Compute strides for both inputs
        def compute_strides(shape):
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= dim
            return tuple(reversed(strides))

        strides_a = compute_strides(shape_a)
        strides_b = compute_strides(shape_b)

        result = AlmeidaTensor(shape=out_shape, dtype=self._dtype)
        out_strides = compute_strides(out_shape)

        for out_idx in range(result.size):
            # Map output index to input indices
            idx_a = self._broadcast_index(out_idx, out_shape, shape_a, strides_a)
            idx_b = self._broadcast_index(out_idx, out_shape, shape_b, strides_b)
            result._buffer[out_idx] = op(self._buffer[idx_a], other._buffer[idx_b])

        return result

    def __add__(self, other: Union['AlmeidaTensor', float]) -> 'AlmeidaTensor':
        """Element-wise addition with broadcasting."""
        if isinstance(other, (int, float)):
            result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
            for i in range(self.size):
                result._buffer[i] = self._buffer[i] + other
            return result

        return self._apply_binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: float) -> 'AlmeidaTensor':
        return self.__add__(other)

    def __sub__(self, other: Union['AlmeidaTensor', float]) -> 'AlmeidaTensor':
        """Element-wise subtraction."""
        if isinstance(other, (int, float)):
            return self + (-other)
        return self + (-other)

    def __rsub__(self, other: float) -> 'AlmeidaTensor':
        return (-self) + other

    def __mul__(self, other: Union['AlmeidaTensor', float]) -> 'AlmeidaTensor':
        """Element-wise multiplication with broadcasting."""
        if isinstance(other, (int, float)):
            result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
            for i in range(self.size):
                result._buffer[i] = self._buffer[i] * other
            return result

        return self._apply_binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: float) -> 'AlmeidaTensor':
        return self.__mul__(other)

    def __truediv__(self, other: Union['AlmeidaTensor', float]) -> 'AlmeidaTensor':
        """Element-wise division with broadcasting."""
        if isinstance(other, (int, float)):
            return self * (1.0 / other)

        return self._apply_binary_op(other, lambda a, b: a / (b + 1e-10))

    def __neg__(self) -> 'AlmeidaTensor':
        """Element-wise negation."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = -self._buffer[i]
        return result

    def __pow__(self, exponent: float) -> 'AlmeidaTensor':
        """Element-wise power."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = self._buffer[i] ** exponent
        return result

    # -------------------------------------------------------------------------
    # Matrix Operations
    # -------------------------------------------------------------------------

    def __matmul__(self, other: 'AlmeidaTensor') -> 'AlmeidaTensor':
        """
        Matrix multiplication: self @ other

        Supports:
        - 2D @ 2D: Standard matrix multiply
        - 1D @ 1D: Dot product
        """
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("matmul requires at least 1D tensors")

        # 1D @ 1D: dot product
        if self.ndim == 1 and other.ndim == 1:
            if self._shape[0] != other._shape[0]:
                raise ValueError(f"Dot product dimension mismatch: {self._shape} vs {other._shape}")
            acc = 0.0
            for i in range(self._shape[0]):
                acc += self._buffer[i] * other._buffer[i]
            return AlmeidaTensor([acc], dtype=self._dtype)

        # 2D @ 2D: matrix multiply
        if self.ndim == 2 and other.ndim == 2:
            M, K1 = self._shape
            K2, N = other._shape
            if K1 != K2:
                raise ValueError(f"Matmul inner dimension mismatch: {K1} vs {K2}")

            result = AlmeidaTensor(shape=(M, N), dtype=self._dtype)
            # Blocked accumulation on raw buffers: same O(n^3), much lower Python overhead.
            a_data = self._buffer._data
            b_data = other._buffer._data
            c_data = result._buffer._data
            block_k = 32

            for i in range(M):
                a_row = i * K1
                c_row = i * N
                for k0 in range(0, K1, block_k):
                    k1 = min(k0 + block_k, K1)
                    for k in range(k0, k1):
                        a_ik = a_data[a_row + k]
                        if a_ik == 0.0:
                            continue
                        b_row = k * N
                        for j in range(N):
                            c_data[c_row + j] += a_ik * b_data[b_row + j]

            return result

        raise NotImplementedError(f"Matmul not supported for shapes {self._shape} @ {other._shape}")

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'AlmeidaTensor':
        """Transpose tensor axes."""
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))

        new_shape = tuple(self._shape[i] for i in axes)
        result = AlmeidaTensor(shape=new_shape, dtype=self._dtype)

        # Fast path: 2D transpose on flat buffers (the common .T), no per-element
        # _flat_to_multi/_multi_to_flat round-trip.
        if self.ndim == 2 and tuple(axes) == (1, 0):
            m, n = self._shape
            sd, rd = self._buffer._data, result._buffer._data
            for i in range(m):
                base = i * n
                for j in range(n):
                    rd[j * m + i] = sd[base + j]
            return result

        for flat_idx in range(self.size):
            old_indices = self._flat_to_multi(flat_idx)
            new_indices = tuple(old_indices[i] for i in axes)
            new_flat = result._multi_to_flat(new_indices)
            result._buffer[new_flat] = self._buffer[flat_idx]

        return result

    def reshape(self, new_shape: Tuple[int, ...]) -> 'AlmeidaTensor':
        """Reshape tensor to new shape."""
        # Handle -1 dimension
        neg_one_idx = None
        new_size = 1
        new_shape = list(new_shape)

        for i, dim in enumerate(new_shape):
            if dim == -1:
                if neg_one_idx is not None:
                    raise ValueError("Only one dimension can be -1")
                neg_one_idx = i
            else:
                new_size *= dim

        if neg_one_idx is not None:
            inferred = self.size // new_size
            new_shape[neg_one_idx] = inferred
            new_size *= inferred

        new_shape = tuple(new_shape)

        if new_size != self.size:
            raise ValueError(f"Cannot reshape {self._shape} ({self.size}) to {new_shape} ({new_size})")

        result = AlmeidaTensor(shape=new_shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = self._buffer[i]

        return result

    # -------------------------------------------------------------------------
    # Reduction Operations
    # -------------------------------------------------------------------------

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'AlmeidaTensor':
        """Sum along axis or globally."""
        if axis is None:
            acc = sum(self._buffer._data)
            if keepdims:
                return AlmeidaTensor([acc], dtype=self._dtype).reshape((1,) * self.ndim)
            return AlmeidaTensor([acc], dtype=self._dtype)

        # Sum along a specific axis — specialized (no per-element lambda; the
        # contiguous inner==1 case reduces each block at C level via sum()).
        if axis < 0:
            axis += self.ndim
        shape = self._shape
        axis_len = shape[axis]
        inner = 1
        for d in shape[axis + 1:]:
            inner *= d
        outer = 1
        for d in shape[:axis]:
            outer *= d

        new_shape = list(shape)
        if keepdims:
            new_shape[axis] = 1
        else:
            new_shape = new_shape[:axis] + new_shape[axis + 1:]
        result = AlmeidaTensor(shape=tuple(new_shape) if new_shape else (1,), dtype=self._dtype)
        sd, rd = self._buffer._data, result._buffer._data
        span = axis_len * inner

        if inner == 1:
            # Reducing the last axis: each output is a contiguous block sum.
            for o in range(outer):
                base = o * span
                rd[o] = sum(sd[base:base + span])
        else:
            for o in range(outer):
                in_o = o * span
                out_o = o * inner
                for a in range(axis_len):
                    in_oa = in_o + a * inner
                    for b in range(inner):
                        rd[out_o + b] += sd[in_oa + b]

        return result

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'AlmeidaTensor':
        """Mean along axis or globally."""
        s = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            return s / self.size
        return s / self._shape[axis]

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'AlmeidaTensor':
        """Maximum along axis or globally."""
        if axis is None:
            m = self._buffer[0]
            for i in range(1, self.size):
                if self._buffer[i] > m:
                    m = self._buffer[i]
            if keepdims:
                return AlmeidaTensor([m], dtype=self._dtype).reshape((1,) * self.ndim)
            return AlmeidaTensor([m], dtype=self._dtype)

        return self._reduce_axis(axis, keepdims, lambda a, b: a if a > b else b, float('-inf'))

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> 'AlmeidaTensor':
        """Minimum along axis or globally."""
        if axis is None:
            m = self._buffer[0]
            for i in range(1, self.size):
                if self._buffer[i] < m:
                    m = self._buffer[i]
            if keepdims:
                return AlmeidaTensor([m], dtype=self._dtype).reshape((1,) * self.ndim)
            return AlmeidaTensor([m], dtype=self._dtype)

        return self._reduce_axis(axis, keepdims, lambda a, b: a if a < b else b, float('inf'))

    def _reduce_axis(
        self,
        axis: int,
        keepdims: bool,
        op: Callable[[float, float], float],
        init: float
    ) -> 'AlmeidaTensor':
        """Generic reduction along an axis."""
        if axis < 0:
            axis += self.ndim

        new_shape = list(self._shape)
        if keepdims:
            new_shape[axis] = 1
        else:
            new_shape = new_shape[:axis] + new_shape[axis+1:]

        result = AlmeidaTensor(shape=tuple(new_shape) if new_shape else (1,), dtype=self._dtype)

        # Any flat index decomposes as o*(axis_len*inner) + a*inner + b, and the
        # reduced output index is simply o*inner + b. Computing that arithmetically
        # avoids a _flat_to_multi/_multi_to_flat round-trip per element.
        shape = self._shape
        axis_len = shape[axis]
        inner = 1
        for d in shape[axis + 1:]:
            inner *= d
        outer = 1
        for d in shape[:axis]:
            outer *= d

        sd = self._buffer._data
        rd = result._buffer._data
        for i in range(len(rd)):
            rd[i] = init

        span = axis_len * inner
        for o in range(outer):
            in_o = o * span
            out_o = o * inner
            for a in range(axis_len):
                in_oa = in_o + a * inner
                for b in range(inner):
                    oidx = out_o + b
                    rd[oidx] = op(rd[oidx], sd[in_oa + b])

        return result

    def norm(self, ord: int = 2, axis: Optional[int] = None) -> 'AlmeidaTensor':
        """Compute L-p norm."""
        if ord == 2:
            return self.sqrt((self * self).sum(axis=axis))
        elif ord == 1:
            return self.abs().sum(axis=axis)
        else:
            return ((self.abs() ** ord).sum(axis=axis)) ** (1.0 / ord)
        return None

    # -------------------------------------------------------------------------
    # Element-wise Math Functions
    # -------------------------------------------------------------------------

    def abs(self) -> 'AlmeidaTensor':
        """Element-wise absolute value."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = abs(self._buffer[i])
        return result

    def sqrt(self, x: Optional['AlmeidaTensor'] = None) -> 'AlmeidaTensor':
        """Element-wise square root."""
        if x is None:
            x = self
        result = AlmeidaTensor(shape=x._shape, dtype=x._dtype)
        for i in range(x.size):
            result._buffer[i] = math.sqrt(max(0, x._buffer[i]))
        return result

    def exp(self) -> 'AlmeidaTensor':
        """Element-wise exponential."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = math.exp(self._buffer[i])
        return result

    def log(self) -> 'AlmeidaTensor':
        """Element-wise natural logarithm."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = math.log(max(1e-10, self._buffer[i]))
        return result

    def cos(self) -> 'AlmeidaTensor':
        """Element-wise cosine."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = math.cos(self._buffer[i])
        return result

    def sin(self) -> 'AlmeidaTensor':
        """Element-wise sine."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            result._buffer[i] = math.sin(self._buffer[i])
        return result

    def clamp(self, min_val: float, max_val: float) -> 'AlmeidaTensor':
        """Clamp values to range."""
        result = AlmeidaTensor(shape=self._shape, dtype=self._dtype)
        for i in range(self.size):
            val = self._buffer[i]
            result._buffer[i] = max(min_val, min(max_val, val))
        return result

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.size <= 20:
            data_str = self._format_data()
            return f"AlmeidaTensor({data_str}, shape={self._shape}, dtype={self._dtype.value})"
        else:
            return f"AlmeidaTensor(shape={self._shape}, dtype={self._dtype.value})"

    def _format_data(self) -> str:
        """Format data for small tensors."""
        if self.ndim == 0:
            return str(self._buffer[0])
        elif self.ndim == 1:
            return "[" + ", ".join(f"{self._buffer[i]:.4g}" for i in range(self._shape[0])) + "]"
        elif self.ndim == 2:
            rows = []
            for i in range(self._shape[0]):
                row = [f"{self[i, j]:.4g}" for j in range(self._shape[1])]
                rows.append("[" + ", ".join(row) + "]")
            return "[" + ", ".join(rows) + "]"
        else:
            return f"<{self.size} elements>"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def zeros(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create tensor filled with zeros."""
    return AlmeidaTensor(shape=shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create tensor filled with ones."""
    t = AlmeidaTensor(shape=shape, dtype=dtype)
    for i in range(t.size):
        t._buffer[i] = 1.0
    return t


def eye(n: int, dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create identity matrix."""
    t = zeros((n, n), dtype=dtype)
    for i in range(n):
        t[i, i] = 1.0
    return t


def randn(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create tensor with random normal values."""
    t = AlmeidaTensor(shape=shape, dtype=dtype)
    for i in range(t.size):
        t._buffer[i] = random.gauss(0, 1)
    return t


def from_list(data: list, dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create tensor from nested list."""
    return AlmeidaTensor(data, dtype=dtype)


def arange(start: float, stop: float, step: float = 1.0, dtype: DType = DType.FLOAT32) -> AlmeidaTensor:
    """Create tensor with evenly spaced values."""
    n = int((stop - start) / step)
    t = AlmeidaTensor(shape=(n,), dtype=dtype)
    for i in range(n):
        t._buffer[i] = start + i * step
    return t


# =============================================================================
# NUMPY/TORCH BRIDGE FUNCTIONS (Phase 2)
# =============================================================================

def from_numpy(arr: 'numpy.ndarray', dtype: Optional[DType] = None) -> AlmeidaTensor:
    """
    Create AlmeidaTensor from NumPy array.

    Args:
        arr: NumPy ndarray
        dtype: Optional dtype override (infers from numpy if None)

    Returns:
        AlmeidaTensor with same data
    """
    import numpy as np

    # Infer dtype from numpy
    if dtype is None:
        np_dtype = arr.dtype
        if np_dtype == np.float32:
            dtype = DType.FLOAT32
        elif np_dtype == np.float64:
            dtype = DType.FLOAT64
        elif np_dtype == np.float16:
            dtype = DType.FLOAT16
        elif np_dtype == np.int8:
            dtype = DType.INT8
        elif np_dtype == np.int16:
            dtype = DType.INT16
        elif np_dtype == np.int32:
            dtype = DType.INT32
        elif np_dtype == np.int64:
            dtype = DType.INT64
        elif np_dtype == np.uint8:
            dtype = DType.UINT8
        else:
            dtype = DType.FLOAT32  # Default

    shape = tuple(arr.shape)
    t = AlmeidaTensor(shape=shape, dtype=dtype)

    # Flatten and copy
    flat = arr.flatten()
    for i in range(len(flat)):
        t._buffer[i] = float(flat[i])

    return t


def to_numpy(t: AlmeidaTensor) -> 'numpy.ndarray':
    """
    Convert AlmeidaTensor to NumPy array.

    Args:
        t: AlmeidaTensor

    Returns:
        NumPy ndarray with same data
    """
    import numpy as np

    # Map dtype
    dtype_map = {
        DType.FLOAT16: np.float16,
        DType.BFLOAT16: np.float32,  # BF16 → F32 for numpy
        DType.FLOAT32: np.float32,
        DType.FLOAT64: np.float64,
        DType.INT8: np.int8,
        DType.INT16: np.int16,
        DType.INT32: np.int32,
        DType.INT64: np.int64,
        DType.UINT8: np.uint8,
    }
    np_dtype = dtype_map.get(t.dtype, np.float32)

    # Copy data
    arr = np.empty(t.size, dtype=np_dtype)
    for i in range(t.size):
        arr[i] = t._buffer[i]

    return arr.reshape(t.shape)


def from_torch(tensor: 'torch.Tensor', dtype: Optional[DType] = None) -> AlmeidaTensor:
    """
    Create AlmeidaTensor from PyTorch tensor.

    Args:
        tensor: PyTorch tensor (will be moved to CPU if on GPU)
        dtype: Optional dtype override

    Returns:
        AlmeidaTensor with same data
    """
    import torch

    # Move to CPU if needed
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert to numpy, then to Almeida
    return from_numpy(tensor.numpy(), dtype)


def to_torch(t: AlmeidaTensor, device: str = 'cpu') -> 'torch.Tensor':
    """
    Convert AlmeidaTensor to PyTorch tensor.

    Args:
        t: AlmeidaTensor
        device: Target device ('cpu', 'cuda', 'mps', etc.)

    Returns:
        PyTorch tensor on specified device
    """
    import torch

    # Map dtype
    dtype_map = {
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.FLOAT32: torch.float32,
        DType.FLOAT64: torch.float64,
        DType.INT8: torch.int8,
        DType.INT16: torch.int16,
        DType.INT32: torch.int32,
        DType.INT64: torch.int64,
        DType.UINT8: torch.uint8,
    }
    torch_dtype = dtype_map.get(t.dtype, torch.float32)

    # Create tensor from buffer data
    data = [t._buffer[i] for i in range(t.size)]
    tensor = torch.tensor(data, dtype=torch_dtype, device=device)

    return tensor.reshape(t.shape)


# =============================================================================
# INFERENCE FUNCTIONS (LLM Operations)
# =============================================================================

def softmax(x: AlmeidaTensor, axis: int = -1) -> AlmeidaTensor:
    """
    Softmax activation function.

    Args:
        x: Input tensor
        axis: Axis to compute softmax over (default: last axis)

    Returns:
        Softmax output (same shape as input)
    """
    # Handle negative axis
    if axis < 0:
        axis = x.ndim + axis

    result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)

    if x.ndim == 1:
        # 1D case: simple softmax
        max_val = max(x._buffer)
        exp_vals = [math.exp(v - max_val) for v in x._buffer]
        sum_exp = sum(exp_vals)
        for i in range(x.size):
            result._buffer[i] = exp_vals[i] / sum_exp

    elif x.ndim == 2:
        rows, cols = x.shape
        if axis == 1:
            # Softmax over columns (each row independently)
            for r in range(rows):
                row_start = r * cols
                # Find max in row
                max_val = max(x._buffer[row_start + c] for c in range(cols))
                # Compute exp(x - max)
                exp_vals = [math.exp(x._buffer[row_start + c] - max_val) for c in range(cols)]
                sum_exp = sum(exp_vals)
                # Normalize
                for c in range(cols):
                    result._buffer[row_start + c] = exp_vals[c] / sum_exp
        else:
            # axis == 0: Softmax over rows (each column independently)
            for c in range(cols):
                # Find max in column
                max_val = max(x._buffer[r * cols + c] for r in range(rows))
                # Compute exp(x - max)
                exp_vals = [math.exp(x._buffer[r * cols + c] - max_val) for r in range(rows)]
                sum_exp = sum(exp_vals)
                # Normalize
                for r in range(rows):
                    result._buffer[r * cols + c] = exp_vals[r] / sum_exp
    else:
        # General ND case: use keepdims broadcast
        x_max = x.max(axis=axis, keepdims=True)

        # Broadcast subtraction for stability
        shifted = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
        for i in range(x.size):
            shifted._buffer[i] = x._buffer[i] - x_max._buffer[i % x_max.size]

        # Compute exp
        exp_x = shifted.exp()

        # Sum along axis with keepdims
        sum_exp = exp_x.sum(axis=axis, keepdims=True)

        # Divide with proper broadcasting
        for i in range(x.size):
            result._buffer[i] = exp_x._buffer[i] / sum_exp._buffer[i % sum_exp.size]

    return result


def silu(x: AlmeidaTensor) -> AlmeidaTensor:
    """
    SiLU (Swish) activation: x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        SiLU output (same shape as input)
    """
    result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
    for i in range(x.size):
        val = x._buffer[i]
        sigmoid = 1.0 / (1.0 + math.exp(-val))
        result._buffer[i] = val * sigmoid
    return result


def gelu(x: AlmeidaTensor) -> AlmeidaTensor:
    """
    GELU activation (Gaussian Error Linear Unit).

    Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        GELU output (same shape as input)
    """
    SQRT_2_PI = 0.7978845608028654  # sqrt(2/pi)

    result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
    for i in range(x.size):
        val = x._buffer[i]
        inner = SQRT_2_PI * (val + 0.044715 * val ** 3)
        result._buffer[i] = 0.5 * val * (1.0 + math.tanh(inner))
    return result


def rms_norm(x: AlmeidaTensor, weight: AlmeidaTensor, eps: float = 1e-6) -> AlmeidaTensor:
    """
    Root Mean Square Layer Normalization.

    Args:
        x: Input tensor (..., hidden_size)
        weight: Learned scale parameter (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor (same shape as input)
    """
    # Compute RMS along last axis
    hidden_size = x.shape[-1]

    if x.ndim == 1:
        # Single vector
        sum_sq = 0.0
        for i in range(hidden_size):
            sum_sq += x._buffer[i] ** 2
        rms = math.sqrt(sum_sq / hidden_size + eps)

        result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
        for i in range(hidden_size):
            result._buffer[i] = (x._buffer[i] / rms) * weight._buffer[i]
        return result

    elif x.ndim == 2:
        # Batch of vectors (seq_len, hidden_size). Work on raw buffers a row at a
        # time: a C-level slice avoids per-element __getitem__/_multi_to_flat.
        seq_len = x.shape[0]
        H = hidden_size
        result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
        xd, rd, wd = x._buffer._data, result._buffer._data, weight._buffer._data

        for s in range(seq_len):
            base = s * H
            row = xd[base:base + H]
            sum_sq = 0.0
            for v in row:
                sum_sq += v * v
            inv_rms = 1.0 / math.sqrt(sum_sq / H + eps)
            for h in range(H):
                rd[base + h] = row[h] * inv_rms * wd[h]

        return result

    else:
        raise NotImplementedError(f"rms_norm not implemented for {x.ndim}D tensors")
    return None


def layer_norm(x: AlmeidaTensor, weight: AlmeidaTensor, bias: Optional[AlmeidaTensor] = None,
               eps: float = 1e-5) -> AlmeidaTensor:
    """
    Standard Layer Normalization.

    Args:
        x: Input tensor (..., hidden_size)
        weight: Learned scale parameter (hidden_size,)
        bias: Optional learned bias (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor (same shape as input)
    """
    hidden_size = x.shape[-1]

    if x.ndim == 1:
        # Compute mean
        mean = 0.0
        for i in range(hidden_size):
            mean += x._buffer[i]
        mean /= hidden_size

        # Compute variance
        var = 0.0
        for i in range(hidden_size):
            diff = x._buffer[i] - mean
            var += diff ** 2
        var /= hidden_size

        std = math.sqrt(var + eps)

        result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
        for i in range(hidden_size):
            normed = (x._buffer[i] - mean) / std
            result._buffer[i] = normed * weight._buffer[i]
            if bias is not None:
                result._buffer[i] += bias._buffer[i]
        return result

    elif x.ndim == 2:
        # Work on raw buffers a row at a time: a C-level slice avoids per-element
        # __getitem__/_multi_to_flat over seq_len * hidden_size elements.
        seq_len = x.shape[0]
        H = hidden_size
        result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
        xd, rd, wd = x._buffer._data, result._buffer._data, weight._buffer._data
        bd = bias._buffer._data if bias is not None else None

        for s in range(seq_len):
            base = s * H
            row = xd[base:base + H]
            mean = sum(row) / H
            var = 0.0
            for v in row:
                d = v - mean
                var += d * d
            inv_std = 1.0 / math.sqrt(var / H + eps)
            if bd is None:
                for h in range(H):
                    rd[base + h] = (row[h] - mean) * inv_std * wd[h]
            else:
                for h in range(H):
                    rd[base + h] = (row[h] - mean) * inv_std * wd[h] + bd[h]

        return result

    else:
        raise NotImplementedError(f"layer_norm not implemented for {x.ndim}D tensors")
    return None


def rope_embed(x: AlmeidaTensor, cos: AlmeidaTensor, sin: AlmeidaTensor,
               position: int = 0) -> AlmeidaTensor:
    """
    Apply Rotary Position Embedding (RoPE).

    Args:
        x: Input tensor (seq_len, n_heads, head_dim)
        cos: Cosine embeddings (max_seq, head_dim)
        sin: Sine embeddings (max_seq, head_dim)
        position: Starting position in sequence

    Returns:
        Tensor with RoPE applied (same shape as input)
    """
    seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2

    result = AlmeidaTensor(shape=x.shape, dtype=x.dtype)
    # Direct base-offset arithmetic on the flat buffers (inference-path hot loop).
    xd, rd = x._buffer._data, result._buffer._data
    cd, sd = cos._buffer._data, sin._buffer._data

    for s in range(seq_len):
        pos = position + s
        cos_base = pos * head_dim
        for h in range(n_heads):
            base = (s * n_heads + h) * head_dim
            for d in range(half_dim):
                x1 = xd[base + d]
                x2 = xd[base + d + half_dim]
                c = cd[cos_base + d]
                si = sd[cos_base + d]
                # Rotate: [cos, -sin; sin, cos] @ [x1, x2]
                rd[base + d] = x1 * c - x2 * si
                rd[base + d + half_dim] = x1 * si + x2 * c

    return result


# =============================================================================
# LINEAR ALGEBRA: SVD, QR, and utilities
# =============================================================================

def qr(A: AlmeidaTensor) -> Tuple[AlmeidaTensor, AlmeidaTensor]:
    """
    QR decomposition using Modified Gram-Schmidt.

    A = Q @ R where Q is orthogonal and R is upper triangular.

    Args:
        A: Matrix of shape (m, n) where m >= n

    Returns:
        Q: Orthogonal matrix (m, n)
        R: Upper triangular matrix (n, n)
    """
    if A.ndim != 2:
        raise ValueError("QR requires 2D matrix")

    m, n = A.shape

    # Modified Gram-Schmidt is column-oriented; extract columns as plain lists
    # once (from the row-major flat buffer) so the O(m*n^2) inner loops run on
    # Python lists instead of paying per-element __getitem__/_multi_to_flat.
    ad = A._buffer._data
    cols = [[ad[i * n + j] for i in range(m)] for j in range(n)]
    Rd = [[0.0] * n for _ in range(n)]

    for j in range(n):
        qj = cols[j]
        norm_sq = 0.0
        for v in qj:
            norm_sq += v * v
        rjj = math.sqrt(norm_sq)
        Rd[j][j] = rjj

        if rjj > 1e-10:
            inv = 1.0 / rjj
            qj = [v * inv for v in qj]
            cols[j] = qj

        for k in range(j + 1, n):
            qk = cols[k]
            dot = 0.0
            for a, b in zip(qj, qk):
                dot += a * b
            Rd[j][k] = dot
            cols[k] = [b - dot * a for a, b in zip(qj, qk)]

    # Write columns back into Q (m, n) and R (n, n) via flat buffers.
    Q = zeros((m, n))
    R = zeros((n, n))
    qd, rd = Q._buffer._data, R._buffer._data
    for j in range(n):
        colj = cols[j]
        for i in range(m):
            qd[i * n + j] = colj[i]
    for r in range(n):
        Rr = Rd[r]
        base = r * n
        for c in range(n):
            rd[base + c] = Rr[c]

    return Q, R


def svd_power_iteration(
    A: AlmeidaTensor,
    k: int = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> Tuple[AlmeidaTensor, AlmeidaTensor, AlmeidaTensor]:
    """
    SVD via power iteration with deflation.

    Compass and ruler approach - no external dependencies.

    Args:
        A: Matrix of shape (m, n)
        k: Number of singular values to compute (default: min(m, n))
        max_iter: Maximum iterations per singular value
        tol: Convergence tolerance

    Returns:
        U: Left singular vectors (m, k)
        S: Singular values (k,)
        Vt: Right singular vectors (k, n)
    """
    if A.ndim != 2:
        raise ValueError("SVD requires 2D matrix")

    m, n = A.shape
    if k is None:
        k = min(m, n)
    k = min(k, min(m, n))

    U = zeros((m, k))
    S = zeros((k,))
    Vt = zeros((k, n))

    # Work with a flat copy for deflation; keep u/v as plain Python lists so the
    # per-iteration matvecs and the deflation rank-1 update avoid per-element
    # __getitem__/__setitem__ over the whole matrix.
    A_work = A.copy()
    awd = A_work._buffer._data

    for i in range(k):
        # Start from a random unit vector.
        v = list(randn((n,))._buffer._data)
        norm = math.sqrt(sum(x * x for x in v))
        inv = 1.0 / norm
        v = [x * inv for x in v]

        sigma_old = 0.0
        u = [0.0] * m

        for _ in range(max_iter):
            # u = A @ v  (row-major, contiguous over awd)
            for row in range(m):
                base = row * n
                acc = 0.0
                for col in range(n):
                    acc += awd[base + col] * v[col]
                u[row] = acc

            sigma = math.sqrt(sum(x * x for x in u))
            if sigma < 1e-10:
                break
            inv = 1.0 / sigma
            u = [x * inv for x in u]

            # v_new = A.T @ u  (accumulate into columns, row-major over awd)
            v_new = [0.0] * n
            for row in range(m):
                base = row * n
                ur = u[row]
                for col in range(n):
                    v_new[col] += awd[base + col] * ur

            norm = math.sqrt(sum(x * x for x in v_new))
            if norm < 1e-10:
                break
            inv = 1.0 / norm
            v = [x * inv for x in v_new]

            if abs(sigma - sigma_old) < tol * sigma:
                break
            sigma_old = sigma

        # Store results: U[:,i] = u, Vt[i,:] = v
        S._buffer[i] = sigma
        ud, vtd = U._buffer._data, Vt._buffer._data
        for j in range(m):
            ud[j * k + i] = u[j]
        vbase = i * n
        for j in range(n):
            vtd[vbase + j] = v[j]

        # Deflate: A_work -= sigma * u (x) v^T
        for row in range(m):
            base = row * n
            su = sigma * u[row]
            for col in range(n):
                awd[base + col] -= su * v[col]

    return U, S, Vt


def svd(A: AlmeidaTensor, full_matrices: bool = False) -> Tuple[AlmeidaTensor, AlmeidaTensor, AlmeidaTensor]:
    """
    Singular Value Decomposition.

    A = U @ diag(S) @ Vt

    Args:
        A: Matrix of shape (m, n)
        full_matrices: If False, return reduced SVD

    Returns:
        U: Left singular vectors
        S: Singular values
        Vt: Right singular vectors
    """
    return svd_power_iteration(A)


def diag(v: AlmeidaTensor) -> AlmeidaTensor:
    """Create diagonal matrix from vector, or extract diagonal from matrix."""
    if v.ndim == 1:
        # Vector -> diagonal matrix
        n = v.shape[0]
        result = zeros((n, n))
        for i in range(n):
            result[i, i] = v._buffer[i]
        return result
    elif v.ndim == 2:
        # Matrix -> diagonal vector
        n = min(v.shape[0], v.shape[1])
        result = zeros((n,))
        for i in range(n):
            result._buffer[i] = v[i, i]
        return result
    else:
        raise ValueError("diag requires 1D or 2D tensor")
    return None


def tensordot(a: AlmeidaTensor, b: AlmeidaTensor, axes) -> AlmeidaTensor:
    """
    Tensor contraction along specified axes.

    For now, only supports common case: axes=([-1], [0])
    which contracts last axis of a with first axis of b.
    """
    if axes == ([-1], [0]) or axes == ([a.ndim - 1], [0]):
        # Contract last axis of a with first axis of b
        # This is like batch matmul

        # Get shapes
        a_shape = list(a.shape)
        b_shape = list(b.shape)

        contract_dim = a_shape[-1]
        assert contract_dim == b_shape[0], f"Contraction dims must match: {contract_dim} vs {b_shape[0]}"

        # Result shape: a.shape[:-1] + b.shape[1:]
        result_shape = tuple(a_shape[:-1] + b_shape[1:])
        result = zeros(result_shape)

        # Compute sizes
        a_outer_size = 1
        for d in a_shape[:-1]:
            a_outer_size *= d

        b_outer_size = 1
        for d in b_shape[1:]:
            b_outer_size *= d

        # Contract
        for i in range(a_outer_size):
            for j in range(b_outer_size):
                acc = 0.0
                for k in range(contract_dim):
                    a_idx = i * contract_dim + k
                    b_idx = k * b_outer_size + j
                    acc += a._buffer[a_idx] * b._buffer[b_idx]
                result._buffer[i * b_outer_size + j] = acc

        return result
    else:
        raise NotImplementedError(f"tensordot only supports axes=([-1], [0]), got {axes}")


def searchsorted(a: AlmeidaTensor, v: float) -> int:
    """Find insertion point for v in sorted array a."""
    if a.ndim != 1:
        raise ValueError("searchsorted requires 1D array")

    for i in range(a.size):
        if a._buffer[i] >= v:
            return i
    return a.size


def argmin(a: AlmeidaTensor) -> int:
    """Return index of minimum value."""
    if a.size == 0:
        raise ValueError("argmin of empty tensor")

    min_val = a._buffer[0]
    min_idx = 0
    for i in range(1, a.size):
        if a._buffer[i] < min_val:
            min_val = a._buffer[i]
            min_idx = i
    return min_idx


def argmax(a: AlmeidaTensor) -> int:
    """Return index of maximum value."""
    if a.size == 0:
        raise ValueError("argmax of empty tensor")

    max_val = a._buffer[0]
    max_idx = 0
    for i in range(1, a.size):
        if a._buffer[i] > max_val:
            max_val = a._buffer[i]
            max_idx = i
    return max_idx


def cumsum(a: AlmeidaTensor) -> AlmeidaTensor:
    """Cumulative sum."""
    result = zeros(a.shape, dtype=a.dtype)
    acc = 0.0
    for i in range(a.size):
        acc += a._buffer[i]
        result._buffer[i] = acc
    return result


def diff(a: AlmeidaTensor) -> AlmeidaTensor:
    """First difference."""
    if a.ndim != 1:
        raise ValueError("diff requires 1D array")
    if a.size < 2:
        return zeros((0,))

    result = zeros((a.size - 1,), dtype=a.dtype)
    for i in range(a.size - 1):
        result._buffer[i] = a._buffer[i + 1] - a._buffer[i]
    return result


def polyfit(x: AlmeidaTensor, y: AlmeidaTensor, deg: int) -> AlmeidaTensor:
    """
    Least squares polynomial fit.

    For deg=1 (linear), returns [slope, intercept].
    """
    if deg != 1:
        raise NotImplementedError("polyfit only supports deg=1 for now")

    n = x.size
    if n != y.size:
        raise ValueError("x and y must have same size")

    # Linear regression: y = slope * x + intercept
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0

    for i in range(n):
        xi = x._buffer[i]
        yi = y._buffer[i]
        sum_x += xi
        sum_y += yi
        sum_xx += xi * xi
        sum_xy += xi * yi

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        slope = 0.0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

    result = zeros((2,))
    result._buffer[0] = slope
    result._buffer[1] = intercept
    return result


def squeeze(a: AlmeidaTensor, axis: int = None) -> AlmeidaTensor:
    """Remove dimensions of size 1."""
    if axis is not None:
        if a.shape[axis] != 1:
            return a.copy()
        new_shape = list(a.shape)
        del new_shape[axis]
        if not new_shape:
            new_shape = (1,)
        return a.reshape(tuple(new_shape))
    else:
        new_shape = tuple(d for d in a.shape if d != 1)
        if not new_shape:
            new_shape = (1,)
        return a.reshape(new_shape)


# =============================================================================
# MAIN
# =============================================================================

