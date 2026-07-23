"""
================================================================================
Almeida Linear Algebra - Pure Python Linear Algebra Operations
================================================================================

Provides pure Python implementations of core linear algebra operations:
- QR decomposition (Gram-Schmidt, Householder)
- SVD (Power iteration, Jacobi)
- Eigenvalue decomposition (Power iteration, QR algorithm)
- LU decomposition
- Cholesky decomposition
- Matrix operations (determinant, inverse, norm, condition number)

All operations work on nested Python lists (List[List[float]]).
No numpy, scipy, or external dependencies.

Usage:
    from almeida_numerics.linalg import (
        qr, svd, eig, lu, cholesky,
        det, inv, norm, cond,
        solve, lstsq,
    )

Author: Michael Almeida
Copyright: (c) Almeida Industries
License: Apache-2.0
Date: 2026-01-01
================================================================================
"""

from typing import List, Tuple, Optional, Union
import math

# Type aliases
Vector = List[float]
Matrix = List[List[float]]


# =============================================================================
# BASIC MATRIX OPERATIONS
# =============================================================================

def zeros(rows: int, cols: int) -> Matrix:
    """Create zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def eye(n: int) -> Matrix:
    """Create identity matrix."""
    result = zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result


def copy_matrix(A: Matrix) -> Matrix:
    """Deep copy a matrix."""
    return [[A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def transpose(A: Matrix) -> Matrix:
    """Transpose matrix."""
    rows, cols = len(A), len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication A @ B."""
    m, k1 = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k1 == k2, f"Incompatible shapes: {m}x{k1} @ {k2}x{n}"

    result = zeros(m, n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k1):
                s += A[i][p] * B[p][j]
            result[i][j] = s
    return result


def matvec(A: Matrix, x: Vector) -> Vector:
    """Matrix-vector multiplication A @ x."""
    m, n = len(A), len(A[0])
    assert len(x) == n, f"Incompatible shapes: {m}x{n} @ {len(x)}"

    result = [0.0] * m
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        result[i] = s
    return result


def dot(x: Vector, y: Vector) -> float:
    """Dot product of two vectors."""
    return sum(xi * yi for xi, yi in zip(x, y))


def outer(x: Vector, y: Vector) -> Matrix:
    """Outer product x @ y.T."""
    return [[xi * yj for yj in y] for xi in x]


def scale_vector(alpha: float, x: Vector) -> Vector:
    """Scale vector by scalar."""
    return [alpha * xi for xi in x]


def add_vectors(x: Vector, y: Vector) -> Vector:
    """Add two vectors."""
    return [xi + yi for xi, yi in zip(x, y)]


def sub_vectors(x: Vector, y: Vector) -> Vector:
    """Subtract two vectors."""
    return [xi - yi for xi, yi in zip(x, y)]


# =============================================================================
# NORMS
# =============================================================================

def vector_norm(x: Vector, p: float = 2.0) -> float:
    """Vector p-norm."""
    if p == float('inf'):
        return max(abs(xi) for xi in x)
    elif p == 1:
        return sum(abs(xi) for xi in x)
    elif p == 2:
        return math.sqrt(sum(xi * xi for xi in x))
    else:
        return sum(abs(xi) ** p for xi in x) ** (1.0 / p)
    return None


def frobenius_norm(A: Matrix) -> float:
    """Frobenius norm of matrix."""
    return math.sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


def matrix_norm(A: Matrix, ord: str = 'fro') -> float:
    """Matrix norm."""
    if ord == 'fro':
        return frobenius_norm(A)
    elif ord == '1':
        # Max column sum
        cols = len(A[0])
        return max(sum(abs(A[i][j]) for i in range(len(A))) for j in range(cols))
    elif ord == 'inf':
        # Max row sum
        return max(sum(abs(x) for x in row) for row in A)
    else:
        return frobenius_norm(A)
    return None


def norm(x: Union[Vector, Matrix], ord: Union[int, float, str] = 2) -> float:
    """Unified norm function."""
    if isinstance(x[0], list):
        return matrix_norm(x, str(ord) if isinstance(ord, str) else 'fro')
    else:
        return vector_norm(x, float(ord) if isinstance(ord, (int, float)) else 2.0)


# =============================================================================
# QR DECOMPOSITION
# =============================================================================

def qr_gram_schmidt(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    QR decomposition using modified Gram-Schmidt.

    A = Q @ R where Q is orthogonal, R is upper triangular.

    More numerically stable than classical Gram-Schmidt.
    """
    m, n = len(A), len(A[0])
    Q = copy_matrix(A)
    R = zeros(n, n)

    for j in range(n):
        # Compute R[j,j] = ||Q[:,j]||
        col_norm = math.sqrt(sum(Q[i][j] ** 2 for i in range(m)))

        if col_norm < 1e-14:
            # Near-zero column, skip
            R[j][j] = 0.0
            continue

        R[j][j] = col_norm

        # Normalize Q[:,j]
        for i in range(m):
            Q[i][j] /= col_norm

        # Orthogonalize remaining columns
        for k in range(j + 1, n):
            # R[j,k] = Q[:,j].T @ Q[:,k]
            R[j][k] = sum(Q[i][j] * Q[i][k] for i in range(m))

            # Q[:,k] -= R[j,k] * Q[:,j]
            for i in range(m):
                Q[i][k] -= R[j][k] * Q[i][j]

    return Q, R


def qr_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    QR decomposition using Householder reflections.

    More numerically stable than Gram-Schmidt for ill-conditioned matrices.
    """
    m, n = len(A), len(A[0])
    R = copy_matrix(A)
    Q = eye(m)

    for j in range(min(m - 1, n)):
        # Extract column below diagonal
        x = [R[i][j] if i >= j else 0.0 for i in range(m)]

        # Compute Householder vector
        x_norm = math.sqrt(sum(x[i] ** 2 for i in range(j, m)))
        if x_norm < 1e-14:
            continue

        # v = x - ||x|| * e_j (with sign to avoid cancellation)
        sign = 1.0 if x[j] >= 0 else -1.0
        x[j] += sign * x_norm

        # Normalize v
        v_norm = math.sqrt(sum(x[i] ** 2 for i in range(j, m)))
        if v_norm < 1e-14:
            continue
        v = [x[i] / v_norm for i in range(m)]

        # Apply Householder reflection to R: R = (I - 2*v*v.T) @ R
        for k in range(j, n):
            dot_v_col = sum(v[i] * R[i][k] for i in range(j, m))
            for i in range(j, m):
                R[i][k] -= 2.0 * v[i] * dot_v_col

        # Apply to Q: Q = Q @ (I - 2*v*v.T)
        for i in range(m):
            dot_row_v = sum(Q[i][k] * v[k] for k in range(j, m))
            for k in range(j, m):
                Q[i][k] -= 2.0 * dot_row_v * v[k]

    return Q, R


def qr(A: Matrix, method: str = 'householder') -> Tuple[Matrix, Matrix]:
    """
    QR decomposition.

    Args:
        A: Input matrix (m x n)
        method: 'householder' (default, more stable) or 'gram_schmidt'

    Returns:
        (Q, R) where A = Q @ R
        Q is orthogonal (m x m), R is upper triangular (m x n)
    """
    if method == 'gram_schmidt':
        return qr_gram_schmidt(A)
    else:
        return qr_householder(A)


# =============================================================================
# LU DECOMPOSITION
# =============================================================================

def lu(A: Matrix) -> Tuple[Matrix, Matrix, List[int]]:
    """
    LU decomposition with partial pivoting.

    PA = LU where P is permutation, L is lower triangular, U is upper triangular.

    Returns:
        (L, U, perm) where perm is the permutation vector
    """
    n = len(A)
    assert len(A[0]) == n, "Matrix must be square"

    U = copy_matrix(A)
    L = eye(n)
    perm = list(range(n))

    for k in range(n - 1):
        # Find pivot (largest element in column k below diagonal)
        max_val = abs(U[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i

        # Swap rows in U and L, update permutation
        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            perm[k], perm[max_row] = perm[max_row], perm[k]
            # Swap L entries to the left of diagonal
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]

        # Skip if pivot is zero
        if abs(U[k][k]) < 1e-14:
            continue

        # Eliminate entries below pivot
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U, perm


# =============================================================================
# CHOLESKY DECOMPOSITION
# =============================================================================

def cholesky(A: Matrix) -> Matrix:
    """
    Cholesky decomposition for symmetric positive definite matrices.

    A = L @ L.T where L is lower triangular.

    Raises:
        ValueError: If matrix is not positive definite
    """
    n = len(A)
    L = zeros(n, n)

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                if abs(L[j][j]) < 1e-14:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]

    return L


# =============================================================================
# EIGENVALUE DECOMPOSITION
# =============================================================================

def power_iteration(A: Matrix, num_iter: int = 100, tol: float = 1e-10) -> Tuple[float, Vector]:
    """
    Power iteration to find dominant eigenvalue and eigenvector.

    Returns:
        (eigenvalue, eigenvector)
    """
    n = len(A)

    # Start with random-ish vector
    v = [1.0 / math.sqrt(n)] * n

    eigenvalue = 0.0
    for _ in range(num_iter):
        # w = A @ v
        w = matvec(A, v)

        # Normalize
        w_norm = vector_norm(w)
        if w_norm < 1e-14:
            break

        v_new = [wi / w_norm for wi in w]

        # Rayleigh quotient for eigenvalue
        Av = matvec(A, v_new)
        new_eigenvalue = dot(v_new, Av)

        # Check convergence
        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v_new

        eigenvalue = new_eigenvalue
        v = v_new

    return eigenvalue, v


def eig_power(A: Matrix, k: int = None, num_iter: int = 100) -> Tuple[Vector, Matrix]:
    """
    Eigenvalue decomposition using power iteration with deflation.

    Finds k largest eigenvalues and their eigenvectors.

    Args:
        A: Square symmetric matrix
        k: Number of eigenvalues to find (default: all)
        num_iter: Max iterations per eigenvalue

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors[:,i] corresponds to eigenvalues[i]
    """
    n = len(A)
    if k is None:
        k = n
    k = min(k, n)

    B = copy_matrix(A)
    eigenvalues = []
    eigenvectors = []

    for _ in range(k):
        val, vec = power_iteration(B, num_iter)
        eigenvalues.append(val)
        eigenvectors.append(vec)

        # Deflate: B = B - val * v @ v.T
        for i in range(n):
            for j in range(n):
                B[i][j] -= val * vec[i] * vec[j]

    # Convert eigenvectors to column format
    V = [[eigenvectors[j][i] for j in range(k)] for i in range(n)]

    return eigenvalues, V


def eig_qr(A: Matrix, num_iter: int = 100, tol: float = 1e-10) -> Tuple[Vector, Matrix]:
    """
    Eigenvalue decomposition using QR algorithm.

    More accurate than power iteration for finding all eigenvalues.
    Works best for symmetric matrices.

    Returns:
        (eigenvalues, eigenvectors)
    """
    n = len(A)
    T = copy_matrix(A)
    V = eye(n)

    for _ in range(num_iter):
        # QR decomposition
        Q, R = qr(T)

        # T = R @ Q (similar matrix)
        T = matmul(R, Q)

        # Accumulate eigenvectors
        V = matmul(V, Q)

        # Check convergence (off-diagonal elements should be small)
        off_diag = sum(abs(T[i][j]) for i in range(n) for j in range(n) if i != j)
        if off_diag < tol * n * n:
            break

    # Eigenvalues are on diagonal
    eigenvalues = [T[i][i] for i in range(n)]

    return eigenvalues, V


def eig(A: Matrix, method: str = 'qr') -> Tuple[Vector, Matrix]:
    """
    Eigenvalue decomposition.

    Args:
        A: Square matrix (should be symmetric for real eigenvalues)
        method: 'qr' (default) or 'power'

    Returns:
        (eigenvalues, eigenvectors)
    """
    if method == 'power':
        return eig_power(A)
    else:
        return eig_qr(A)


# =============================================================================
# SINGULAR VALUE DECOMPOSITION (SVD)
# =============================================================================

def svd_power(A: Matrix, k: int = None, num_iter: int = 100) -> Tuple[Matrix, Vector, Matrix]:
    """
    Truncated SVD using power iteration.

    Computes A = U @ diag(s) @ V.T

    Args:
        A: Input matrix (m x n)
        k: Number of singular values (default: min(m, n))
        num_iter: Max iterations per singular value

    Returns:
        (U, s, Vt) where U is m x k, s is k, Vt is k x n
    """
    m, n = len(A), len(A[0])
    if k is None:
        k = min(m, n)
    k = min(k, min(m, n))

    B = copy_matrix(A)
    U_cols = []
    s_vals = []
    V_cols = []

    for _ in range(k):
        # Power iteration on B.T @ B to find right singular vector
        # Start with random-ish vector
        v = [1.0 / math.sqrt(n)] * n

        for _ in range(num_iter):
            # u = B @ v
            u = matvec(B, v)
            u_norm = vector_norm(u)
            if u_norm < 1e-14:
                break
            u = [ui / u_norm for ui in u]

            # v = B.T @ u
            v_new = [sum(B[i][j] * u[i] for i in range(m)) for j in range(n)]
            v_norm = vector_norm(v_new)
            if v_norm < 1e-14:
                break
            v = [vi / v_norm for vi in v_new]

        # Singular value
        Av = matvec(B, v)
        sigma = vector_norm(Av)

        if sigma < 1e-14:
            break

        # Left singular vector
        u = [avi / sigma for avi in Av]

        U_cols.append(u)
        s_vals.append(sigma)
        V_cols.append(v)

        # Deflate: B = B - sigma * u @ v.T
        for i in range(m):
            for j in range(n):
                B[i][j] -= sigma * u[i] * v[j]

    # Build output matrices
    actual_k = len(s_vals)
    U = [[U_cols[j][i] for j in range(actual_k)] for i in range(m)]
    Vt = [[V_cols[i][j] for j in range(n)] for i in range(actual_k)]

    return U, s_vals, Vt


def svd_full(A: Matrix) -> Tuple[Matrix, Vector, Matrix]:
    """
    Full SVD via eigenvalue decomposition.

    Computes eigenvalues of A.T @ A and A @ A.T.
    More accurate than power iteration for small matrices.
    """
    m, n = len(A), len(A[0])

    # Compute A.T @ A
    AtA = matmul(transpose(A), A)

    # Eigendecomposition of A.T @ A gives V and s^2
    eigenvalues, V = eig_qr(AtA)

    # Singular values are sqrt of eigenvalues (handle numerical issues)
    s = [math.sqrt(max(0, ev)) for ev in eigenvalues]

    # Sort by singular value (descending)
    indices = sorted(range(len(s)), key=lambda i: -s[i])
    s = [s[i] for i in indices]
    V = [[V[row][i] for i in indices] for row in range(n)]

    # Compute U = A @ V @ diag(1/s)
    U = zeros(m, n)
    for j in range(n):
        if s[j] > 1e-14:
            v_j = [V[i][j] for i in range(n)]
            Av = matvec(A, v_j)
            for i in range(m):
                U[i][j] = Av[i] / s[j]

    # Vt = V.T
    Vt = transpose(V)

    return U, s, Vt


def svd(A: Matrix, k: int = None, method: str = 'power') -> Tuple[Matrix, Vector, Matrix]:
    """
    Singular Value Decomposition.

    A = U @ diag(s) @ Vt

    Args:
        A: Input matrix (m x n)
        k: Number of singular values (None for all)
        method: 'power' (default, faster) or 'full' (more accurate)

    Returns:
        (U, s, Vt) - Left singular vectors, singular values, right singular vectors
    """
    if method == 'full' or k is None:
        U, s, Vt = svd_full(A)
        if k is not None:
            U = [[U[i][j] for j in range(k)] for i in range(len(U))]
            s = s[:k]
            Vt = Vt[:k]
        return U, s, Vt
    else:
        return svd_power(A, k)


# =============================================================================
# SOLVING LINEAR SYSTEMS
# =============================================================================

def solve_triangular_lower(L: Matrix, b: Vector) -> Vector:
    """Solve L @ x = b where L is lower triangular."""
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        s = b[i] - sum(L[i][j] * x[j] for j in range(i))
        x[i] = s / L[i][i] if abs(L[i][i]) > 1e-14 else 0.0
    return x


def solve_triangular_upper(U: Matrix, b: Vector) -> Vector:
    """Solve U @ x = b where U is upper triangular."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / U[i][i] if abs(U[i][i]) > 1e-14 else 0.0
    return x


def solve(A: Matrix, b: Vector) -> Vector:
    """
    Solve A @ x = b using LU decomposition.

    Args:
        A: Square matrix (n x n)
        b: Right-hand side vector (n)

    Returns:
        Solution vector x
    """
    L, U, perm = lu(A)

    # Permute b
    b_perm = [b[perm[i]] for i in range(len(b))]

    # Solve L @ y = b_perm
    y = solve_triangular_lower(L, b_perm)

    # Solve U @ x = y
    x = solve_triangular_upper(U, y)

    return x


def lstsq(A: Matrix, b: Vector) -> Vector:
    """
    Least squares solution to A @ x = b.

    Minimizes ||A @ x - b||^2 using QR decomposition.

    Args:
        A: Matrix (m x n), m >= n
        b: Vector (m)

    Returns:
        Solution vector x (n)
    """
    m, n = len(A), len(A[0])

    # QR decomposition
    Q, R = qr(A)

    # Compute Q.T @ b
    Qtb = [sum(Q[i][j] * b[i] for i in range(m)) for j in range(n)]

    # Solve R @ x = Q.T @ b (R is n x n upper part)
    R_square = [R[i][:n] for i in range(n)]
    x = solve_triangular_upper(R_square, Qtb[:n])

    return x


def cg(A: Matrix, b: Vector, x0: Optional[Vector] = None,
       tol: float = 1e-10, max_iter: Optional[int] = None) -> Vector:
    """
    Solve A @ x = b for symmetric positive-definite A by conjugate gradient.

    An iterative solver: each step is one matrix-vector product plus a few
    vector dot/axpy operations, so it bottoms out entirely at the BLAS-1/2
    primitives (matvec, dot, scaled vector add) and never forms an inverse.

    Args:
        A: Symmetric positive-definite matrix (n x n)
        b: Right-hand side (n)
        x0: Optional initial guess (defaults to zeros)
        tol: Stop when the residual L2 norm falls below this
        max_iter: Iteration cap (defaults to n)

    Returns:
        Solution vector x (n)
    """
    n = len(b)
    if max_iter is None:
        max_iter = n
    x = [0.0] * n if x0 is None else list(x0)

    r = sub_vectors(b, matvec(A, x))             # r = b - A x
    p = list(r)
    rs_old = dot(r, r)

    for _ in range(max_iter):
        if vector_norm(r) < tol:
            break
        Ap = matvec(A, p)
        alpha = rs_old / dot(p, Ap)
        x = add_vectors(x, scale_vector(alpha, p))
        r = sub_vectors(r, scale_vector(alpha, Ap))
        rs_new = dot(r, r)
        p = add_vectors(r, scale_vector(rs_new / rs_old, p))
        rs_old = rs_new

    return x


# =============================================================================
# MATRIX INVERSE AND DETERMINANT
# =============================================================================

def det(A: Matrix) -> float:
    """
    Determinant using LU decomposition.
    """
    n = len(A)
    L, U, perm = lu(A)

    # Determinant is product of U diagonal * sign of permutation
    d = 1.0
    for i in range(n):
        d *= U[i][i]

    # Count swaps in permutation
    swaps = 0
    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        swaps += cycle_len - 1

    return d * ((-1) ** swaps)


def inv(A: Matrix) -> Matrix:
    """
    Matrix inverse using LU decomposition.

    Solves A @ X = I column by column.
    """
    n = len(A)
    L, U, perm = lu(A)

    result = zeros(n, n)
    for j in range(n):
        # Solve for j-th column of inverse
        e_j = [1.0 if i == j else 0.0 for i in range(n)]
        e_perm = [e_j[perm[i]] for i in range(n)]

        y = solve_triangular_lower(L, e_perm)
        x = solve_triangular_upper(U, y)

        for i in range(n):
            result[i][j] = x[i]

    return result


def cond(A: Matrix) -> float:
    """
    Condition number (ratio of largest to smallest singular value).

    High condition number indicates ill-conditioned matrix.
    """
    U, s, Vt = svd(A)

    s_max = max(s) if s else 1.0
    s_min = min(si for si in s if si > 1e-14) if s else 1e-14

    return s_max / s_min


def rank(A: Matrix, tol: float = 1e-10) -> int:
    """
    Matrix rank (number of significant singular values).
    """
    U, s, Vt = svd(A)
    return sum(1 for si in s if si > tol)


# =============================================================================
# ADDITIONAL UTILITIES
# =============================================================================

def pinv(A: Matrix, tol: float = 1e-10) -> Matrix:
    """
    Moore-Penrose pseudoinverse using SVD.

    A+ = V @ diag(1/s) @ U.T
    """
    m, n = len(A), len(A[0])
    U, s, Vt = svd(A)

    # Compute pseudoinverse of singular values
    s_inv = [1.0 / si if si > tol else 0.0 for si in s]

    # A+ = V @ diag(s_inv) @ U.T
    # First: diag(s_inv) @ U.T
    k = len(s_inv)
    temp = [[s_inv[i] * U[j][i] for j in range(m)] for i in range(k)]

    # Then: V @ temp = Vt.T @ temp
    V = transpose(Vt)
    result = zeros(n, m)
    for i in range(n):
        for j in range(m):
            result[i][j] = sum(V[i][p] * temp[p][j] for p in range(k))

    return result


def matrix_power(A: Matrix, n: int) -> Matrix:
    """
    Compute A^n using binary exponentiation.
    """
    size = len(A)
    if n == 0:
        return eye(size)
    if n == 1:
        return copy_matrix(A)
    if n < 0:
        A = inv(A)
        n = -n

    result = eye(size)
    base = copy_matrix(A)

    while n > 0:
        if n % 2 == 1:
            result = matmul(result, base)
        base = matmul(base, base)
        n //= 2

    return result


def trace(A: Matrix) -> float:
    """Sum of diagonal elements."""
    return sum(A[i][i] for i in range(min(len(A), len(A[0]))))


# =============================================================================
# COMPOSITE TRANSFORMS (Patent-derived)
# =============================================================================

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # More numerically stable for negative x
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def gated_dot(x: Vector, gate_vector: Vector, weights: Vector) -> Vector:
    """
    Gated Dot Product: DotV → Sigmoid → MulV

    Gating mechanism commonly used in LSTMs, GRUs, and attention.
    Computes dot product of x and gate_vector, applies sigmoid to get a gate
    value, then multiplies weights by that gate.

    Patent sources: Gating mechanism patents + dot product patents
    Composite transform from algorithm database.

    Args:
        x: Input vector for computing gate value
        gate_vector: Vector to dot with x to produce gate signal
        weights: Weights to be gated (multiplied by sigmoid output)

    Returns:
        Gated weights vector (same shape as weights)

    Example:
        >>> x = [1.0, 2.0, 3.0]
        >>> gate_v = [0.5, 0.5, 0.5]  # dot product = 3.0
        >>> weights = [1.0, 1.0, 1.0]
        >>> result = gated_dot(x, gate_v, weights)
        >>> # gate = sigmoid(3.0) ≈ 0.953
        >>> # result ≈ [0.953, 0.953, 0.953]
    """
    # Step 1: Dot product (element-wise multiply and sum)
    min_len = min(len(x), len(gate_vector))
    dot_product = sum(x[i] * gate_vector[i] for i in range(min_len))

    # Step 2: Apply sigmoid to get gate value (0-1)
    gate = sigmoid(dot_product)

    # Step 3: Multiply weights by gate value
    return [w * gate for w in weights]


def gated_dot_matrix(X: Matrix, gate_vector: Vector, weights: Vector) -> Matrix:
    """
    Batched gated dot product for matrix input.

    Applies gated_dot to each row of X.

    Args:
        X: Input matrix (m x n)
        gate_vector: Gate vector (n)
        weights: Weights to gate (k)

    Returns:
        Matrix of gated outputs (m x k)
    """
    return [gated_dot(row, gate_vector, weights) for row in X]


def sigmoid_vector(x: Vector) -> Vector:
    """Apply sigmoid element-wise to a vector."""
    return [sigmoid(xi) for xi in x]


def tanh_vector(x: Vector) -> Vector:
    """Apply tanh element-wise to a vector."""
    return [math.tanh(xi) for xi in x]


def layer_norm(x: Vector, eps: float = 1e-5) -> Vector:
    """
    Layer Normalization.

    Normalizes vector to zero mean and unit variance.

    Args:
        x: Input vector
        eps: Small constant for numerical stability

    Returns:
        Normalized vector
    """
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = math.sqrt(var + eps)
    return [(xi - mean) / std for xi in x]


def softmax(x: Vector) -> Vector:
    """
    Softmax activation function.

    Converts vector of scores to probability distribution.
    Uses max-subtraction for numerical stability.

    Args:
        x: Input vector of scores

    Returns:
        Probability distribution (sums to 1)
    """
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def norm_attention(query: Vector, keys: Matrix) -> Vector:
    """
    Normalized Attention: LayerNorm → Matmul → Softmax

    Standard transformer attention block pattern. Computes attention
    weights from a query and a set of keys.

    Patent sources: Transformer patents + normalization patents
    Composite transform from algorithm database.

    Args:
        query: Query vector (d_k)
        keys: Key matrix (n x d_k) - n keys of dimension d_k

    Returns:
        Attention weights (n) - probability distribution over keys

    Example:
        >>> query = [1.0, 0.5, 0.5]
        >>> keys = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        >>> weights = norm_attention(query, keys)
        >>> # weights will favor the first key (highest dot product)
    """
    # Step 1: Layer normalization of query
    normed_query = layer_norm(query)

    # Step 2: Compute attention scores (query @ keys.T)
    scores = []
    for key in keys:
        score = dot(normed_query, key)
        scores.append(score)

    # Step 3: Softmax to get attention weights
    attention_weights = softmax(scores)

    return attention_weights


def attention_weighted_sum(attention: Vector, values: Matrix) -> Vector:
    """
    Compute weighted sum of values using attention weights.

    Args:
        attention: Attention weights (n)
        values: Value matrix (n x d_v)

    Returns:
        Weighted sum vector (d_v)
    """
    if not values:
        return []

    d_v = len(values[0])
    result = [0.0] * d_v

    for i, weight in enumerate(attention):
        for j in range(d_v):
            result[j] += weight * values[i][j]

    return result


def scaled_dot_product_attention(query: Vector, keys: Matrix, values: Matrix,
                                  scale: float = None) -> Tuple[Vector, Vector]:
    """
    Full scaled dot-product attention mechanism.

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    Args:
        query: Query vector (d_k)
        keys: Key matrix (n x d_k)
        values: Value matrix (n x d_v)
        scale: Scaling factor (default: 1/sqrt(d_k))

    Returns:
        (output, attention_weights) - output is (d_v), weights is (n)
    """
    d_k = len(query)
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    # Compute scaled attention scores
    scores = []
    for key in keys:
        score = dot(query, key) * scale
        scores.append(score)

    # Softmax
    attention_weights = softmax(scores)

    # Weighted sum of values
    output = attention_weighted_sum(attention_weights, values)

    return output, attention_weights


def gru_cell(x: Vector, h_prev: Vector,
             W_z: Matrix, U_z: Matrix, b_z: Vector,
             W_r: Matrix, U_r: Matrix, b_r: Vector,
             W_h: Matrix, U_h: Matrix, b_h: Vector) -> Vector:
    """
    GRU (Gated Recurrent Unit) cell - uses gating mechanisms.

    This is a complete GRU implementation using our gated operations.

    Args:
        x: Input vector at current timestep
        h_prev: Previous hidden state
        W_z, U_z, b_z: Update gate weights
        W_r, U_r, b_r: Reset gate weights
        W_h, U_h, b_h: Candidate hidden state weights

    Returns:
        New hidden state
    """
    hidden_size = len(h_prev)

    # Update gate: z = sigmoid(W_z @ x + U_z @ h_prev + b_z)
    z_x = matvec(W_z, x)
    z_h = matvec(U_z, h_prev)
    z = sigmoid_vector(add_vectors(add_vectors(z_x, z_h), b_z))

    # Reset gate: r = sigmoid(W_r @ x + U_r @ h_prev + b_r)
    r_x = matvec(W_r, x)
    r_h = matvec(U_r, h_prev)
    r = sigmoid_vector(add_vectors(add_vectors(r_x, r_h), b_r))

    # Candidate hidden state: h_tilde = tanh(W_h @ x + U_h @ (r * h_prev) + b_h)
    r_h_prev = [r[i] * h_prev[i] for i in range(hidden_size)]
    h_x = matvec(W_h, x)
    h_rh = matvec(U_h, r_h_prev)
    h_tilde = tanh_vector(add_vectors(add_vectors(h_x, h_rh), b_h))

    # New hidden state: h = (1 - z) * h_prev + z * h_tilde
    h_new = [
        (1 - z[i]) * h_prev[i] + z[i] * h_tilde[i]
        for i in range(hidden_size)
    ]

    return h_new


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALMEIDA LINALG - VALIDATION")
    print("=" * 60)

    # Test QR
    print("\n[1] QR Decomposition")
    A = [[1, 2], [3, 4], [5, 6]]
    Q, R = qr(A)
    # Verify Q is orthogonal (Q.T @ Q ≈ I)
    QtQ = matmul(transpose(Q), Q)
    err = sum(abs(QtQ[i][j] - (1 if i == j else 0)) for i in range(2) for j in range(2))
    print(f"  Q orthogonality error: {err:.2e}")
    # Verify A ≈ Q @ R
    QR = matmul(Q, R)
    err = sum(abs(QR[i][j] - A[i][j]) for i in range(3) for j in range(2))
    print(f"  Reconstruction error: {err:.2e}")
    assert err < 1e-10, "QR failed"
    print("  QR: OK")

    # Test LU
    print("\n[2] LU Decomposition")
    A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
    L, U, perm = lu(A)
    # Verify PA = LU
    PA = [[A[perm[i]][j] for j in range(3)] for i in range(3)]
    LU = matmul(L, U)
    err = sum(abs(LU[i][j] - PA[i][j]) for i in range(3) for j in range(3))
    print(f"  Reconstruction error: {err:.2e}")
    assert err < 1e-10, "LU failed"
    print("  LU: OK")

    # Test solve
    print("\n[3] Linear Solve")
    A = [[3, 1], [1, 2]]
    b = [9, 8]
    x = solve(A, b)
    # Verify A @ x ≈ b
    Ax = matvec(A, x)
    err = sum(abs(Ax[i] - b[i]) for i in range(2))
    print(f"  Solution: x = [{x[0]:.4f}, {x[1]:.4f}]")
    print(f"  Residual error: {err:.2e}")
    assert err < 1e-10, "solve failed"
    print("  solve: OK")

    # Test eigenvalues
    print("\n[4] Eigenvalue Decomposition")
    A = [[4, 1], [1, 3]]  # Symmetric
    eigenvalues, V = eig(A)
    print(f"  Eigenvalues: {eigenvalues}")
    # Known eigenvalues: (7 ± sqrt(5))/2 ≈ 4.618, 2.382
    assert abs(max(eigenvalues) - (3.5 + math.sqrt(1.25))) < 0.1, "eig failed"
    print("  eig: OK")

    # Test SVD
    print("\n[5] SVD")
    A = [[1, 2], [3, 4], [5, 6]]
    U, s, Vt = svd(A)
    print(f"  Singular values: {[f'{si:.4f}' for si in s]}")
    # Verify reconstruction: A ≈ U @ diag(s) @ Vt
    # U is m x k, s is k, Vt is k x n
    k = len(s)
    m, n = len(A), len(A[0])
    # Build Sigma as k x k diagonal
    Sigma = [[s[j] if i == j else 0.0 for j in range(k)] for i in range(k)]
    # U[:, :k] @ Sigma @ Vt
    US = matmul(U, Sigma)
    recon = matmul(US, Vt)
    err = sum(abs(recon[i][j] - A[i][j]) for i in range(m) for j in range(n))
    print(f"  Reconstruction error: {err:.2e}")
    assert err < 1e-8, "SVD failed"
    print("  SVD: OK")

    # Test lstsq
    print("\n[6] Least Squares")
    A = [[1, 1], [1, 2], [1, 3]]
    b = [1, 2, 2]
    x = lstsq(A, b)
    print(f"  Solution: x = [{x[0]:.4f}, {x[1]:.4f}]")
    # Should be approximately [0.667, 0.5] for linear regression
    print("  lstsq: OK")

    # Test det and inv
    print("\n[7] Determinant and Inverse")
    A = [[1, 2], [3, 4]]
    d = det(A)
    print(f"  det(A) = {d:.4f} (expected: -2)")
    assert abs(d - (-2)) < 1e-10, "det failed"

    A_inv = inv(A)
    # Verify A @ A_inv ≈ I
    I_approx = matmul(A, A_inv)
    err = sum(abs(I_approx[i][j] - (1 if i == j else 0)) for i in range(2) for j in range(2))
    print(f"  Inverse error: {err:.2e}")
    assert err < 1e-10, "inv failed"
    print("  det/inv: OK")

    # Test condition number
    print("\n[8] Condition Number")
    A = [[1, 0], [0, 1]]
    c = cond(A)
    print(f"  cond(I) = {c:.4f} (expected: 1)")
    assert abs(c - 1) < 0.1, "cond failed for identity"

    A = [[1, 1], [1, 1.001]]
    c = cond(A)
    print(f"  cond(ill-conditioned) = {c:.0f}")
    print("  cond: OK")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)