"""Atom-level master: a plain LASSO over fixed ReLU features.

Column generation for the convex ReLU program can be represented at the
*atom* level instead of the chamber-block level.  Each column is a single
ReLU feature
    a_j = (X u_j)_+  in R^n,   ||u_j|| = 1,
produced by the pricing oracle's direction u_j (which is, by construction, a valid
ReLU feature -- it lives in its own chamber's cone).  The restricted master is then
the ordinary LASSO
    min_{beta in R^P}  1/2 || A beta - y ||^2  +  lam ||beta||_1,   A = [a_1 ... a_P],
with no cone constraints.  For a fixed set of directions this is the finite
restricted master used by the comparison solvers.  The routines support NumPy
and an optional single-device CuPy backend; they use finite iterations,
tolerances, and ridge regularization and should be read as numerical solvers.

Solver: accelerated proximal gradient (FISTA) with adaptive restart for the bulk of
the progress (fully batched / GPU), followed by an optional float64 cyclic
coordinate-descent polish over the active set.  Atoms are stored as
A = relu(X U); A @ beta and A^T r are the only n-sized kernels.
"""

from __future__ import annotations

import numpy as np

from ..array_backend import get_xp, to_cpu


def build_atoms(X, U, device="cpu", dtype=None):
    """A = relu(X U)  (n x P) on the device; U is d x P unit directions."""
    xp = get_xp(device)
    if dtype is None:
        dtype = xp.float32 if device == "gpu" else xp.float64
    Xx = xp.asarray(X, dtype=dtype)
    Uu = xp.asarray(U, dtype=dtype)
    A = xp.maximum(Xx @ Uu, 0.0)
    return A


def _power_L(A, xp, iters=20, seed=0):
    """Largest eigenvalue of A^T A (Lipschitz constant) via power iteration."""
    P = A.shape[1]
    v = xp.asarray(np.random.default_rng(seed).standard_normal(P), dtype=A.dtype)
    v /= xp.linalg.norm(v) + 1e-30
    lam = 1.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        lam = float(xp.linalg.norm(w))
        v = w / (lam + 1e-30)
    return lam


def active_set_lasso(
    A, yx, lam, xp, S0=None, Aty=None, max_outer=None, ridge=1e-9, tol=1e-7
):
    """Warm-startable numerical Lasso by a primal active-set method.

        min_beta 1/2 ||A beta - y||^2 + lam ||beta||_1 .

    Robust where FISTA struggles (correlated atoms): the active blocks are solved
    by a small float64 KKT solve  G_S beta_S = A_S^T y - lam s_S, sign-infeasible
    coordinates are dropped, and the most KKT-violating inactive atom (|a_j^T r| >
    lam) is added; it stops when the computed correlations satisfy the tolerance.
    The only n-sized operation is one A^T r matmul (device) per outer step.
    Returns (beta_dev, nu_dev, obj, S, signs).
    """
    _n, P = A.shape
    if P == 0:
        return xp.zeros(0, dtype=A.dtype), yx.copy(), 0.5 * float(yx @ yx), [], []
    if Aty is None:
        Aty = A.T @ yx
    Aty_c = to_cpu(Aty).astype(np.float64)
    beta = xp.zeros(P, dtype=A.dtype)
    S = list(S0) if S0 else []
    if max_outer is None:
        max_outer = P + 16
    bc = np.zeros(P)
    for _ in range(max_outer):
        # ---- solve reduced KKT on S, dropping sign-infeasible coords ----
        while S:
            idx = xp.asarray(np.asarray(S))
            As = A[:, idx]
            G = to_cpu(As.T @ As).astype(np.float64)
            sgn = np.sign(bc[S])
            sgn[sgn == 0] = 1.0
            rhs = Aty_c[S] - lam * sgn
            try:
                bs = np.linalg.solve(G + ridge * np.eye(len(S)), rhs)
            except np.linalg.LinAlgError:
                bs = np.linalg.lstsq(G, rhs, rcond=None)[0]
            # if signs are consistent and nonzero, accept; else drop worst offender
            bad = (np.sign(bs) != sgn) | (np.abs(bs) <= 1e-14)
            if not bad.any():
                bc[:] = 0.0
                for v, j in zip(bs, S):
                    bc[j] = v
                break
            drop = int(np.argmax(bad * (np.abs(bs) + 1.0)))  # drop a bad coord
            S.pop(drop)
        else:
            bc[:] = 0.0
        beta = xp.asarray(bc, dtype=A.dtype)
        r = A @ beta - yx
        corr = to_cpu(A.T @ r).astype(np.float64)  # a_j^T r
        viol = np.abs(corr) - lam
        if S:
            viol[np.asarray(S)] = -np.inf
        jmax = int(np.argmax(viol))
        if viol[jmax] <= tol * max(1.0, lam):
            break
        S.append(jmax)
        bc[jmax] = -np.sign(corr[jmax]) * 1e-12  # seed sign
    obj = 0.5 * float(r.astype(xp.float64) @ r.astype(xp.float64)) + lam * float(
        np.abs(bc).sum()
    )
    return beta, -r, obj, S, None


def lasso_fista_device(
    A, yx, lam, xp, beta0=None, iters=200, L=None, tol=1e-9, eval_every=20
):
    """Device-native FISTA for the LASSO (no host round-trips, no polish).  Returns
    (beta_dev, nu_dev, obj_float).  Used inside the direction-refinement inner loop,
    where many warm-started solves happen and CPU transfers would dominate."""
    _n, P = A.shape
    if P == 0:
        return xp.zeros(0, dtype=A.dtype), yx.copy(), 0.5 * float(yx @ yx)
    if L is None:
        L = _power_L(A, xp)
    step = 1.0 / (L + 1e-30)
    thr = lam * step
    beta = (
        xp.zeros(P, dtype=A.dtype)
        if beta0 is None
        else xp.asarray(beta0, dtype=A.dtype)
    )
    z = beta.copy()
    t = 1.0
    Az = A @ z
    last = None
    for k in range(iters):
        g = A.T @ (Az - yx)
        bn = z - step * g
        beta_new = xp.sign(bn) * xp.maximum(xp.abs(bn) - thr, 0.0)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_new + ((t - 1.0) / tn) * (beta_new - beta)
        if float(g @ (beta_new - beta)) > 0:
            z = beta_new.copy()
            tn = 1.0
        beta = beta_new
        t = tn
        Az = A @ z
        if eval_every and (k + 1) % eval_every == 0:
            r = A @ beta - yx
            obj = 0.5 * float(r @ r) + lam * float(xp.abs(beta).sum())
            if last is not None and abs(last - obj) <= tol * max(1.0, abs(obj)):
                break
            last = obj
    r = A @ beta - yx
    obj = 0.5 * float(r @ r) + lam * float(xp.abs(beta).sum())
    return beta, -r, obj


def solve_lasso(
    A,
    y,
    lam,
    device="cpu",
    iters=800,
    beta0=None,
    tol=1e-9,
    eval_every=25,
    L=None,
    polish=True,
    polish_sweeps=40,
    verbose=False,
):
    """LASSO  min_beta 1/2||A beta - y||^2 + lam||beta||_1  by FISTA + CD polish.

    Returns dict(obj, nu=y-A beta, beta, active_idx, L).
    """
    xp = get_xp(device)
    yx = xp.asarray(y, dtype=A.dtype)
    _n, P = A.shape
    if P == 0:
        nu = np.asarray(y, dtype=np.float64)
        return dict(
            obj=0.5 * float(nu @ nu), nu=nu, beta=np.zeros(0), active_idx=[], L=1.0
        )
    if L is None:
        L = _power_L(A, xp)
    step = 1.0 / (L + 1e-30)
    beta = (
        xp.zeros(P, dtype=A.dtype)
        if beta0 is None
        else xp.asarray(beta0, dtype=A.dtype)
    )
    z = beta.copy()
    t = 1.0
    Az = A @ z
    last = None
    thr = lam * step
    for k in range(iters):
        r = Az - yx
        g = A.T @ r  # P
        bn = z - step * g
        beta_new = xp.sign(bn) * xp.maximum(xp.abs(bn) - thr, 0.0)  # soft-threshold
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_new + ((t - 1.0) / tn) * (beta_new - beta)
        # adaptive restart (gradient scheme): if overshoot, reset momentum
        if float(g @ (beta_new - beta)) > 0:
            z = beta_new.copy()
            tn = 1.0
        beta = beta_new
        t = tn
        Az = A @ z
        if eval_every and (k + 1) % eval_every == 0:
            rr = A @ beta - yx
            obj = 0.5 * float(
                rr.astype(xp.float64) @ rr.astype(xp.float64)
            ) + lam * float(xp.abs(beta).sum())
            if verbose:
                print(f"   [fista {k + 1:4d}] obj={obj:.8f}")
            if last is not None and abs(last - obj) <= tol * max(1.0, abs(obj)):
                break
            last = obj

    bc = to_cpu(beta).astype(np.float64)

    def obj_of(b):
        bdev = xp.asarray(b, dtype=A.dtype)
        rr = A @ bdev - yx
        return 0.5 * float(rr.astype(xp.float64) @ rr.astype(xp.float64)) + lam * float(
            np.abs(b).sum()
        )

    obj = obj_of(bc)
    if polish:
        bp = _active_set_refine(A, yx, lam, bc.copy(), xp)
        op = obj_of(bp)
        if op <= obj + 1e-12 * max(1.0, abs(obj)):  # guard: polish never worsens
            bc, obj = bp, op

    beta = xp.asarray(bc, dtype=A.dtype)
    nu = yx - A @ beta
    active = [
        j
        for j in range(len(bc))
        if abs(bc[j]) > 1e-9 * max(1.0, float(np.abs(bc).max()))
    ]
    return dict(obj=obj, nu=to_cpu(nu), beta=bc, active_idx=active, L=float(L))


def _active_set_refine(A, yx, lam, beta, xp, max_rounds=8, ridge=1e-10):
    """Float64 active-set polish with sign and feasibility correction.

    On the active set with fixed signs s, the KKT system is
        (A_S^T A_S) beta_S = A_S^T y - lam * s,
    a tiny |S| x |S| solve.  We solve it, drop coordinates whose sign flips or hits
    zero, and repeat (a primal active-set / LARS-style endgame).  A small ridge
    guards near-duplicate atoms; the resulting residual feeds the bound estimate."""
    P = beta.shape[0]
    thr = 1e-9 * max(1.0, float(np.abs(beta).max()) if P else 1.0)
    S = [j for j in range(P) if abs(beta[j]) > thr]
    if not S:
        return beta
    yc = to_cpu(yx).astype(np.float64)
    for _ in range(max_rounds):
        if not S:
            break
        As = to_cpu(A[:, S]).astype(np.float64)  # n x |S|
        G = As.T @ As
        sgn = np.sign(beta[S])
        sgn[sgn == 0] = 1.0
        rhs = As.T @ yc - lam * sgn
        try:
            bs = np.linalg.solve(G + ridge * np.eye(len(S)), rhs)
        except np.linalg.LinAlgError:
            bs = np.linalg.lstsq(G, rhs, rcond=None)[0]
        # drop coords that changed sign or vanished
        keep = np.abs(bs) > thr
        same_sign = np.sign(bs) == sgn
        good = keep & same_sign
        beta[:] = 0.0
        for idx, j in enumerate(S):
            if good[idx]:
                beta[j] = bs[idx]
        newS = [S[i] for i in range(len(S)) if good[i]]
        if len(newS) == len(S):
            break
        S = newS
    return beta
