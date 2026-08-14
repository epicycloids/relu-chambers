"""Gated consolidation: the joint move that escapes alternating-scheme traps.

This is a proposal mechanism for moving several active directions together.
It freezes each active atom's mask m_j = 1{X u_j >= 0} and approximately solves
the group-Lasso over the resulting chamber-restricted weight vectors,

    min_Z 1/2 || sum_j m_j ⊙ (X z_j) - y ||^2 + lam sum_j ||z_j||_2

(the gated-ReLU restricted master).  The numerical iterate is projected back to
valid ReLU directions (u_j = z_j/||z_j||).  The caller then performs a full-data
numerical finite-master refit.  Full working-set replacement requires a computed
objective no larger than the incumbent; after a losing replacement trial, the
caller may instead append a few proposal directions and refit.  This mechanism
does not turn the approximate master or the overall production loop into the
ideal exact algorithm.

The proposal runs in float32 on a host row sketch using block-coordinate
proximal updates.  Its purpose is practical consolidation; it carries no global
optimality or certificate guarantee."""

from __future__ import annotations

import numpy as np


def gated_group_lasso(
    Xs, ys, M, lam, Z0=None, iters=120, tol=1e-7, row_w=None, check_every=0, rng_seed=0
):
    """Block coordinate descent on the frozen-mask group-Lasso (host).

    Xs: m x d (float32), M: m x P {0,1} masks.  Each block update is a few
    proximal-gradient steps with a block-specific Lipschitz estimate.  `iters`
    controls the approximate sweep budget.  Returns the numerical iterate and
    its frozen-mask objective value."""
    m, d = Xs.shape
    P = M.shape[1]
    dt = Xs.dtype
    Z = np.zeros((d, P), dtype=dt) if Z0 is None else np.asarray(Z0, dtype=dt).copy()
    # per-block d x d Grams and Lipschitz constants (one batched pass)
    Qs = []
    Ls = np.empty(P)
    for j in range(P):
        Q = Xs.T @ (M[:, j : j + 1] * Xs)
        Qs.append(Q)
        Ls[j] = max(float(np.linalg.eigvalsh(0.5 * (Q + Q.T))[-1]), 1e-12)
    r = ((Xs @ Z) * M).sum(axis=1) - ys

    def fobj(Zc, rc):
        return 0.5 * float(rc @ rc) + lam * float(np.linalg.norm(Zc, axis=0).sum())

    last = None
    sweeps = max(8, iters // 3)  # one CD sweep ~ 3 FISTA iterations of work
    if m > 100_000:  # large sketches are memory-bound: fewer,
        sweeps = min(sweeps, 12)  # cheaper sweeps with a looser tolerance
        tol = max(tol, 1e-5)
    for sw in range(sweeps):
        for j in range(P):
            zj = Z[:, j]
            fj = (Xs @ zj) * M[:, j]
            rj = r - fj
            cj = Xs.T @ (M[:, j] * rj)  # constant over inner steps
            for _inner in range(3):
                g = Qs[j] @ zj + cj
                znew = zj - g / Ls[j]
                nz = float(np.linalg.norm(znew))
                znew *= max(0.0, 1.0 - (lam / Ls[j]) / max(nz, 1e-30))
                if np.array_equal(znew, zj):
                    break
                zj = znew
            Z[:, j] = zj
            r = rj + (Xs @ zj) * M[:, j]
        obj = fobj(Z, r)
        if last is not None and abs(last - obj) <= tol * max(1.0, abs(obj)):
            break
        last = obj
    return Z, fobj(Z, r)


def propose_consolidation(
    engine,
    U_active,
    beta_active,
    lam,
    sketch_m=32768,
    rng=None,
    iters=120,
    rng_seed=0,
    row_norms=None,
):
    """Gated group-Lasso on a host sketch; returns projected unit directions of
    surviving blocks (host float64 d x P') or None.  The caller performs a
    full-data numerical restricted-master refit before accepting them."""
    rng = rng or np.random.default_rng(rng_seed)
    n, _d = engine.n, engine.d
    # honor the requested sketch size exactly (the caller's self-tuner owns the
    # budget); only fall back to full rows when the request reaches n
    m = min(sketch_m, n) if sketch_m else n
    if m < n:
        # Importance reweighting can inflate row scales and slow the proximal
        # proposal.  Use a uniform, unweighted objective subsample and rescale
        # lam to the subsample size; full-data master work follows in the caller.
        idx = rng.choice(n, size=m, replace=False)
        Xs, ys = engine.gather_rows(idx)
        row_w = None
        lam = lam * (m / n)
    else:
        Xs, ys = engine._Xhost, engine._yhost
        row_w = None
    Xs = np.ascontiguousarray(Xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    Ua = np.asarray(U_active, np.float32)
    M = (Xs @ Ua >= 0).astype(np.float32)
    Z0 = Ua * np.asarray(beta_active, np.float32)[None, :]
    Z, _ = gated_group_lasso(
        Xs, ys, M, lam, Z0=Z0, iters=iters, row_w=row_w, rng_seed=rng_seed
    )
    Zc = Z.astype(np.float64)
    nz = np.linalg.norm(Zc, axis=0)
    keep = np.where(nz > 1e-10 * max(1.0, float(nz.max()) if nz.size else 1.0))[0]
    if not len(keep):
        return None
    keep = keep[np.argsort(-nz[keep])]  # strongest blocks first
    return Zc[:, keep] / nz[keep][None, :]
