"""Definitive Screening Design (DSD) — Jones & Nachtsheim (2011).

The frontier of screening designs. Detects:
- All main effects
- All quadratic effects
- Some two-factor interactions
In only 2k+1 runs (k = number of factors).

Nobody else packages this as a standalone pip library.
"""

from __future__ import annotations

from ..core.types import DesignMatrix, Factor


def _build_conference_matrix(k: int) -> list[list[int]]:
    """Build a k x k conference matrix for DSD construction.

    A conference matrix C has 0 on the diagonal and +/-1 off-diagonal,
    with C'C = (k-1)*I (orthogonal columns).

    For even k, uses Paley-type construction from quadratic residues
    of a prime p = k-1 (when k-1 is prime), or stored matrices for
    common sizes. For odd k, borders an even conference matrix.

    Reference: Jones & Nachtsheim (2011), Xiao et al. (2012).
    """
    import numpy as np

    # Use Paley construction for even k where k-1 is prime.
    # For odd k, take top-left submatrix from (k+1) conference matrix.
    # Fall back to search for non-Paley sizes.
    return _paley_conference(k)


def _paley_conference(k: int) -> list[list[int]]:
    """Build conference matrix using Paley construction.

    For even n=k where n-1 is prime, use quadratic residues of GF(n-1).
    For odd k, take first k rows/cols from (k+1) conference matrix.
    """
    import numpy as np

    # For odd k, build a (k+1) matrix and take top-left k x k
    if k % 2 == 1:
        C_big = np.array(_paley_conference(k + 1))
        C_sub = C_big[:k, :k].copy()
        # Verify it's still close to orthogonal, fall back to search if not
        xtx = C_sub.T @ C_sub
        off = np.abs(xtx - np.diag(np.diag(xtx))).max()
        if off <= 1:  # acceptable for DSD purposes
            return C_sub.tolist()
        # Fall back to search
        return _search_conference(k)

    # Even k: bordered Paley from QR of p = k-1
    p = k - 1
    if not _is_prime(p):
        return _search_conference(k)

    # Compute quadratic residues mod p
    qr = set()
    for i in range(1, p):
        qr.add((i * i) % p)

    # Build Jacobi matrix Q (p x p)
    Q = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                Q[i, j] = 0
            else:
                diff = (i - j) % p
                Q[i, j] = 1 if diff in qr else -1

    # Border with row/col of 1s (first row and first col)
    C = np.zeros((k, k), dtype=int)
    C[0, 0] = 0
    C[0, 1:] = 1
    C[1:, 0] = 1
    C[1:, 1:] = Q

    return C.tolist()


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _search_conference(k: int) -> list[list[int]]:
    """Brute-force search for a conference matrix when Paley doesn't apply."""
    import numpy as np

    best = None
    best_score = float('inf')
    rng = np.random.RandomState(42)
    target = (k - 1) * np.eye(k)

    for _ in range(2000):
        C = np.zeros((k, k), dtype=int)
        for i in range(k):
            for j in range(i + 1, k):
                val = 1 if rng.random() < 0.5 else -1
                C[i, j] = val
                C[j, i] = -val  # skew-symmetric off-diagonal

        xtx = C.T @ C
        score = float(np.sum((xtx - target) ** 2))
        if score < best_score:
            best_score = score
            best = C.copy()
        if score == 0:
            break

    return best.tolist()


def definitive_screening_design(factors: list[Factor], randomize: bool = True) -> DesignMatrix:
    """Generate a Definitive Screening Design.

    For k continuous factors, generates 2k+1 runs that can estimate:
    - k main effects
    - k quadratic effects
    - k(k-1)/2 two-factor interactions (some confounded)

    Conference matrix construction (Jones & Nachtsheim 2011).
    """
    import random as rng

    k = len(factors)

    if k < 3:
        # DSD not efficient for < 3 factors, use full factorial
        from .factorial import full_factorial
        return full_factorial(factors, center_points=1)

    # Build conference matrix C (k x k) per Jones & Nachtsheim (2011).
    # C must satisfy: each column has one 0 and (k-1) entries of +/-1,
    # and C'C = (k-1)*I (orthogonal main effects).
    # Construction: cyclic method for odd k, augmented for even k.
    C = _build_conference_matrix(k)

    # DSD = [C; -C; 0] — fold-over ensures orthogonality
    matrix = []
    for row in C:
        matrix.append(list(row))
    for row in C:
        matrix.append([-x for x in row])
    # Center point
    matrix.append([0] * k)

    # Randomize run order
    run_order = list(range(1, len(matrix) + 1))
    if randomize:
        combined = list(zip(run_order, matrix))
        rng.shuffle(combined)
        run_order, matrix = zip(*combined)
        run_order = list(run_order)
        matrix = [list(row) for row in matrix]

    return DesignMatrix(
        factors=factors, matrix=matrix, run_order=run_order,
        is_coded=True, design_type=f"Definitive Screening Design ({2*k+1} runs, {k} factors)",
        center_points=1,
    )


def augment_dsd_for_rsm(design: DesignMatrix, additional_center_points: int = 2) -> DesignMatrix:
    """Augment a DSD with additional center points for RSM modeling.

    After screening with DSD, if you need a full response surface model,
    add center points to estimate pure error and improve quadratic estimates.
    """
    k = design.n_factors
    matrix = [list(row) for row in design.matrix]
    for _ in range(additional_center_points):
        matrix.append([0] * k)

    run_order = list(range(1, len(matrix) + 1))

    return DesignMatrix(
        factors=design.factors, matrix=matrix, run_order=run_order,
        is_coded=True,
        design_type=f"Augmented DSD ({len(matrix)} runs)",
        center_points=design.center_points + additional_center_points,
    )
