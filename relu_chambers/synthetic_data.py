"""Data generation and hyperplane-arrangement utilities.

We work with the convexified two-layer ReLU program of Pilanci & Ergen.
A *chamber* / *activation pattern* is a sign vector s in {0,1}^n that is
realizable as s = 1{X u >= 0} for some u in R^d.
"""

from __future__ import annotations

import numpy as np


def relu(z):
    return np.maximum(z, 0.0)


def pattern_of(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Activation pattern s = 1{X u >= 0} in {0,1}^n."""
    return (X @ u >= 0).astype(np.int8)


def pattern_key(s: np.ndarray) -> bytes:
    return np.asarray(s, dtype=np.int8).tobytes()


def make_planted(
    n: int,
    d: int,
    k: int,
    rng: np.random.Generator,
    noise: float = 0.0,
    normalize_rows: bool = False,
):
    """Planted teacher model.

    y = sum_{j=1}^k (X u_j^*)_+ * alpha_j^*  (+ noise), alpha_j^* > 0.

    Returns dict with X, y, teacher directions U_star (d x k),
    teacher coefficients alpha_star, teacher chambers S_star (list of {0,1}^n),
    and the clean signal y_clean.
    """
    X = rng.standard_normal((n, d))
    if normalize_rows:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    U = rng.standard_normal((d, k))
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
    alpha = rng.uniform(0.7, 1.3, size=k)  # positive coefficients
    feats = relu(X @ U)  # n x k
    y_clean = feats @ alpha  # n
    y = y_clean.copy()
    if noise > 0:
        y = y + noise * rng.standard_normal(n)
    S_star = [pattern_of(X, U[:, j]) for j in range(k)]
    return dict(
        X=X,
        y=y,
        y_clean=y_clean,
        U_star=U,
        alpha_star=alpha,
        S_star=S_star,
        n=n,
        d=d,
        k=k,
    )


def margin_of(X: np.ndarray, u: np.ndarray) -> float:
    """Data margin rho = min_i |x_i^T u| / (||x_i|| ||u||) -- sine of the angle
    between u and the nearest data hyperplane."""
    a = X @ u
    rn = np.linalg.norm(X, axis=1) * (np.linalg.norm(u) + 1e-12)
    return float(np.min(np.abs(a) / (rn + 1e-12)))


def estimate_solid_angle(
    X: np.ndarray, s_target: np.ndarray, n_samples: int, rng: np.random.Generator
) -> float:
    """Monte-Carlo estimate of mu(s_target) = Pr_{g~N(0,I)}[ sign(Xg)=s_target ]."""
    st = np.asarray(s_target, dtype=np.int8)[:, None]  # n x 1
    hits = 0
    done = 0
    batch = 50000
    d = X.shape[1]
    while done < n_samples:
        b = min(batch, n_samples - done)
        G = rng.standard_normal((d, b))
        A = (X @ G >= 0).astype(np.int8)  # n x b
        match = np.all(A == st, axis=0)  # vectorized comparison
        hits += int(match.sum())
        done += b
    return hits / n_samples


def sample_patterns(
    X: np.ndarray,
    N: int,
    rng: np.random.Generator,
    seed_directions: np.ndarray | None = None,
):
    """Draw N Gaussian directions, return list of UNIQUE realizable patterns
    (excluding the all-zero pattern). Optionally include extra seed directions."""
    d = X.shape[1]
    G = rng.standard_normal((d, N))
    if seed_directions is not None and seed_directions.size:
        G = np.concatenate([G, seed_directions.reshape(d, -1)], axis=1)
    A = (X @ G >= 0).astype(np.int8)  # n x (N+extra)
    seen = {}
    for c in range(A.shape[1]):
        s = A[:, c]
        if s.sum() == 0:
            continue
        seen[s.tobytes()] = s.copy()
    return list(seen.values())
