# almeida-numerics

A from-scratch numerical stack — **tensor**, **linear algebra**, and **statistics** — written in
pure Python. The core of every module uses only the standard library: no NumPy, no SciPy, no
compiled extensions. Every operation is checked against NumPy/SciPy on random inputs, and the
test suite reports the actual numerical error per operation.

> **What this is:** a correct, readable, dependency-free implementation you can read end to end and
> drop into an environment where you can't (or don't want to) pull in the scientific-Python stack.
>
> **What this is not:** a performance replacement for NumPy. Pure-Python loops are orders of
> magnitude slower than vectorized C. The value here is *correctness you can audit* and *zero
> dependencies* — not speed. If you need throughput, use NumPy; this exists for the cases where you
> want to see exactly how the number came out, or can't add a binary dependency.

## Validation

The library is only trustworthy if its outputs match a reference. Every public operation is
compared against NumPy/SciPy on random inputs; the suites print the max-abs error per op and exit
non-zero on any mismatch.

| suite | reference | checks | result |
|-------|-----------|-------:|:------:|
| `tests/test_kernels_vs_numpy.py` | NumPy | 16 | ✅ all pass |
| `tests/test_tensor_vs_numpy.py` | NumPy | 64 | ✅ all pass |
| `tests/test_linalg_vs_numpy.py` | NumPy | 38 | ✅ all pass |
| `tests/test_stats_vs_scipy.py`  | NumPy + SciPy | 63 | ✅ all pass |
| **total** | | **181** | **all pass** |

Closed-form operations match to machine precision (max error ~1e-16). The hand-written
decompositions are validated by reconstruction and by their defining invariants:

- **QR** — `A = QR` reconstructs to ~1e-7 (fp32); `Qᵀ Q = I` to 1e-4; `R` upper-triangular.
- **SVD** — one-sided Jacobi (deterministic): `A = UΣVᵀ` reconstructs to fp32 precision; singular values match NumPy to machine precision and descend. `svd_power_iteration` remains available for top-k of large matrices.
- **LU** — `PA = LU`, with `L` lower- and `U` upper-triangular.
- **Cholesky** — `A = LLᵀ` for SPD inputs.
- **eig** (symmetric) — eigenvalues match `numpy.linalg.eigvalsh`.

Iterative routines (power-method eig/SVD) use looser, explicitly-labelled tolerances than the
closed-form ops — the suites say which is which.

## Performance

Pure Python can't touch NumPy's vectorized C for raw throughput — and this library doesn't try to.
What it *does* do is stay out of the interpreter's way: hot paths hand their inner loops to C
builtins (`sum`, `map`, list comprehensions, slice-assignment) and operate on raw `array` storage
via a small internal kernel layer, rather than paying per-element method dispatch. The result is a
reference implementation that's honest about the ceiling but not needlessly slow.

Representative timings — Python 3.12, float32, best-of-5, both libraries timed in the **same
process** (`python bench/benchmark.py`). NumPy is shown only for scale; these are microbenchmarks
and NumPy times at this size are noisy.

| operation | almeida (ms) | numpy (ms) | notes |
|-----------|-------------:|-----------:|-------|
| matmul 128×128 | 36.6 | 0.011 | inner product via `sum(map(mul, …))` |
| qr 64×64 | 6.9 | 0.058 | Modified Gram–Schmidt, column-major workspace |
| svd 32×32 | 27.0 | 0.079 | one-sided Jacobi (deterministic) |
| softmax 64×256 | 1.32 | 0.018 | contiguous row slices |
| layer_norm 64×256 | 1.56 | 0.026 | raw-buffer row slices |
| add 128×128 | 0.73 | 0.001 | bulk zip-comprehension |
| sum(axis=0) 128×128 | 0.85 | 0.003 | outer/reduce/inner strides |
| sum(all) 128×128 | 0.08 | 0.002 | C-level `sum()` over the buffer |

The gap is smallest where the work naturally maps to a C builtin (`sum(all)`) and largest for
matmul, where NumPy additionally uses SIMD and a cache-blocked, multithreaded BLAS that pure Python
has no access to. **If you need throughput, cross to NumPy at the boundary** with `to_numpy` /
`from_numpy` — that's the intended escape hatch, not a hidden backend.

Every optimization is gated by the 181-check suite: the numbers above changed, the outputs did not.

## Install

```bash
pip install -e .            # core only — installs nothing else
pip install -e ".[test]"    # adds numpy + scipy, needed only to run the suites
```

The core has **no runtime dependencies**. NumPy/SciPy are pulled in only by the `[test]` extra,
because the tests compare against them.

## Quickstart

```python
import almeida_numerics as an

# --- tensor: n-dim array with dtypes, matmul, NN ops, QR/SVD ---
a = an.AlmeidaTensor([[1.0, 2.0], [3.0, 4.0]])
b = a @ a
print(b[0, 0])                      # 7.0
Q, R = an.qr(a)                     # hand-written QR
U, S, Vt = an.svd(a)                # hand-written SVD
y = an.softmax(an.AlmeidaTensor([1.0, 2.0, 3.0]))

# --- linalg: list-based linear algebra (no tensor type needed) ---
from almeida_numerics import linalg as la
x = la.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])   # -> [2.0, 3.0]
d = la.det([[1.0, 2.0], [3.0, 4.0]])                  # -> -2.0
Qh, Rh = la.qr([[1.0, 2.0], [3.0, 4.0]], method="householder")

# --- stats: descriptive, distributions, tests, regression ---
from almeida_numerics import stats as st
st.median([3, 1, 2])                                  # 2.0
st.pearsonr([1, 2, 3, 4], [2, 4, 6, 8])               # (r, p)
slope, intercept, r2 = st.linear_regression([1, 2, 3], [2, 4, 6])
```

Optional NumPy/PyTorch interop lives in `tensor` and is imported lazily — the core never touches
them:

```python
an.from_numpy(np_array)     # np.ndarray  -> AlmeidaTensor
an.to_numpy(t)              # AlmeidaTensor -> np.ndarray
an.from_torch(torch_tensor); an.to_torch(t)
```

## Modules

- **`almeida_numerics.tensor`** — `AlmeidaTensor` over Python `array`/`struct` with custom dtypes
  (fp32/fp16/bf16/…): broadcast elementwise math, matmul, transpose/reshape, reductions,
  `softmax`/`silu`/`gelu`/`rms_norm`/`layer_norm`/`rope_embed`, `qr`/`svd`/`diag`/`tensordot`, and
  utilities (`argmin`/`argmax`/`cumsum`/`diff`/`polyfit`/`searchsorted`/`squeeze`).
- **`almeida_numerics.linalg`** — list-based linear algebra: `matmul`/`matvec`/`dot`/`outer`, norms,
  `qr` (Gram-Schmidt & Householder), `lu`, `cholesky`, `eig` (power / QR), `svd` (power / full),
  `solve`, `lstsq`, `det`, `inv`, `pinv`, `rank`, `cond`, `matrix_power`, `trace`.
- **`almeida_numerics.stats`** — descriptive statistics; percentiles/quartiles; covariance and
  Pearson/Spearman correlation; Normal/Student-t distributions (pdf/cdf/ppf); z-test, one- and
  two-sample (Student & Welch) t-tests, chi-square; linear and polynomial regression; histogram,
  bootstrap, jackknife, normalization.

## Running the suites

```bash
pip install -e ".[test]"
python tests/test_kernels_vs_numpy.py
python tests/test_tensor_vs_numpy.py
python tests/test_linalg_vs_numpy.py
python tests/test_stats_vs_scipy.py
```

Each prints a per-operation report and exits `0` only if every check passes. CI runs all four on
Python 3.8 / 3.10 / 3.12.

## License

Apache License 2.0 — © 2026 Almeida Industries. Author: Michael Almeida. See `LICENSE`.
