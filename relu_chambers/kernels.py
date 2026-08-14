"""Chunked / device-agnostic primitives for the RCG solver.

Design rules (the same rules that make the method shard across devices):
  * Every n-sized operation is expressed as a reduction over ROW CHUNKS of X
    (a chunk is exactly what a device shard would own).  On one device the
    chunk loop bounds peak memory and keeps temporaries cache-sized; on many
    devices the same loop becomes the shard map and the d x P reductions are
    the only communication.
  * Reductions that feed the master's KKT system are accumulated in float64
    even when the data/atom storage is float32 (per-chunk upcast), so the
    active-set system is evaluated more stably while bulk storage and bandwidth
    remain half-width.
"""

from __future__ import annotations

import numpy as np

from .array_backend import to_cpu

DEF_CHUNK = 1 << 16  # 65536 rows per chunk


def as_dev(a, xp, dtype):
    return xp.asarray(a, dtype=dtype)


def relu_XU(X, U, xp, out_dtype=None, chunk=DEF_CHUNK):
    """A = relu(X @ U) computed chunk-wise (n x P)."""
    n = X.shape[0]
    P = U.shape[1]
    dt = out_dtype or X.dtype
    A = xp.empty((n, P), dtype=dt)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        A[s:e] = xp.maximum(X[s:e] @ U, 0.0).astype(dt, copy=False)
    return A


def gemvT64(A, r, xp, chunk=DEF_CHUNK):
    """A^T r accumulated in float64 (screening / KKT correlations).
    A may be float32; per-chunk products are upcast before accumulation."""
    n, P = A.shape
    out = xp.zeros(P, dtype=xp.float64)
    if A.dtype == xp.float64:
        return A.T @ xp.asarray(r, dtype=xp.float64)
    r64 = xp.asarray(r, dtype=xp.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out += A[s:e].astype(xp.float64, copy=False).T @ r64[s:e]
    return out


def gram64(A, idx, xp, chunk=DEF_CHUNK):
    """Float64 Gram block A[:, idx]^T A[:, idx] (chunked upcast)."""
    n = A.shape[0]
    k = len(idx)
    G = xp.zeros((k, k), dtype=xp.float64)
    cols = xp.asarray(np.asarray(idx))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Ac = A[s:e][:, cols].astype(xp.float64, copy=False)
        G += Ac.T @ Ac
    return G


def gram_cross64(A, idx_new, idx_all, xp, chunk=DEF_CHUNK):
    """Float64 cross-Gram A[:, idx_all]^T A[:, idx_new]  (|all| x |new|)."""
    n = A.shape[0]
    out = xp.zeros((len(idx_all), len(idx_new)), dtype=xp.float64)
    ca = xp.asarray(np.asarray(idx_all))
    cn = xp.asarray(np.asarray(idx_new))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out += A[s:e][:, ca].astype(xp.float64, copy=False).T @ A[s:e][:, cn].astype(
            xp.float64, copy=False
        )
    return out


def matvec64(A, beta, xp, chunk=DEF_CHUNK):
    """A @ beta in float64 (prediction; chunked upcast)."""
    n = A.shape[0]
    out = xp.empty(n, dtype=xp.float64)
    b64 = xp.asarray(beta, dtype=xp.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out[s:e] = A[s:e].astype(xp.float64, copy=False) @ b64
    return out


def chamber_keys(X, U, xp, chunk=DEF_CHUNK):
    """Hash key per column of the chamber pattern 1{XU>=0}, without an n x P
    float intermediate: bit-pack each chunk's boolean block and fold it into a
    per-column blake2b hash.  Returns list of bytes keys (len P)."""
    import hashlib

    n = X.shape[0]
    P = U.shape[1]
    hs = [hashlib.blake2b(digest_size=16) for _ in range(P)]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Mc = to_cpu((X[s:e] @ U) >= 0)
        bits = np.packbits(Mc, axis=0)  # ceil(rows/8) x P
        for j in range(P):
            hs[j].update(bits[:, j].tobytes())
    return [h.digest() for h in hs]


def importance_sample(nu, row_norms, m, rng, mix=0.3):
    """Row sketch: indices + inverse-probability weights for unbiased sums.
    p_i ∝ (1-mix) * |nu_i| ||x_i|| / Z + mix / n  (mixture keeps weights bounded).
    Returns (idx, w) with w_i = 1/(m p_i)."""
    n = len(nu)
    if m >= n:
        return np.arange(n), np.ones(n)
    score = np.abs(np.asarray(nu)) * np.asarray(row_norms)
    Z = score.sum()
    if not np.isfinite(Z) or Z <= 0:
        p = np.full(n, 1.0 / n)
    else:
        p = (1.0 - mix) * score / Z + mix / n
    idx = rng.choice(n, size=m, replace=True, p=p)
    w = 1.0 / (m * p[idx])
    return idx, w
