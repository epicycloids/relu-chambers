"""Scalable pricing oracle:  rho(nu) = max_{||u||<=1} nu^T (Xu)_+ .

A *lower* bound on rho (it returns an actual unit direction u).  Three ingredients,
all O(n d) per step and fully batched (one matmul does all restarts at once), so the
oracle runs on GPU at large n:

  1. Structured seeds:  residual correlation X^T nu, and the top-few left singular
     vectors of X^T diag(nu) (eigvecs of the d x d matrix X^T diag(nu)^2 X).
  2. Factored SDP rounding (Goemans--Williamson / Nesterov) WITHOUT the n x n matrix:
     the Max-Cut SDP for G = B B^T (B = diag(|nu|) X, rank <= d) is solved in factored
     form  max_{L in R^{r x n}, unit columns} ||L B||_F^2  by a few power-iteration
     steps (Burer--Monteiro), then rounded by random hyperplanes  eps = sign(L^T g),
     u ~ B^T eps.  This is motivated by the SDP rounding analysis, but the
     finite-rank numerical solve is used only to propose directions.
  3. Batched best-response / projected-gradient ascent on the TRUE (nonconcave)
     objective, polishing every candidate; the returned value is a numerical
     lower bound because it is the objective at a realized direction.

The combined oracle dominates each part: rounding supplies global structure (it wins
on the hard Max-Cut instances where ascent stalls), ascent supplies local refinement.
"""

from __future__ import annotations

import numpy as np

from ..array_backend import eigh_small, get_xp, to_cpu


def _normalize_cols(U, xp, eps=1e-12):
    return U / (xp.linalg.norm(U, axis=0, keepdims=True) + eps)


def _objective(Xx, nux, U, xp):
    """nu^T (X u)_+ for each column of U (d x R) -> (R,).  Returns (vals, A, mask)."""
    A = Xx @ U  # n x R
    mask = A >= 0
    vals = (nux[:, None] * (A * mask)).sum(axis=0)
    return vals, A, mask


def _subgrad(Xx, nux, mask, xp):
    """X^T (nu ⊙ 1{Xu>=0}) for each column -> (d x R)."""
    return Xx.T @ (nux[:, None] * mask)


def _softplus_polish(Xx, nux, U0, xp, deltas=(1.0, 0.3, 0.1, 0.03), n_iters=25, lr=0.4):
    """SACG idea: ascend the SMOOTHED price  phi_delta(u)=nu^T delta*log(1+exp(Xu/delta))
    over the sphere with delta annealed downward, so the optimizer slides through the
    ReLU kinks that trap plain subgradient ascent, then hand off to the exact objective.
    Gradient: X^T(nu ⊙ sigmoid(Xu/delta)).  Returns polished unit directions (d x R)."""
    U = _normalize_cols(U0, xp)
    # scale delta by the typical magnitude of Xu so the schedule is data-adaptive
    scale = float(xp.sqrt((Xx**2).sum(axis=1).mean())) + 1e-12
    for delta in deltas:
        dl = delta * scale
        for _ in range(n_iters):
            Z = (Xx @ U) / dl
            sig = 1.0 / (1.0 + xp.exp(-xp.clip(Z, -30, 30)))
            G = Xx.T @ (nux[:, None] * sig)  # d x R smoothed gradient
            gn = xp.linalg.norm(G, axis=0, keepdims=True) + 1e-30
            U = _normalize_cols(U + lr * G / gn, xp)
    return U


def _batched_ascent(Xx, nux, U0, xp, n_iters=60, lr=0.3):
    """Projected best-response/gradient ascent on the true objective, all columns
    in parallel.  Returns (best_vals (R,), best_U (d x R))."""
    U = _normalize_cols(U0, xp)
    vals, _A, mask = _objective(Xx, nux, U, xp)
    best_vals = vals.copy()
    best_U = U.copy()
    for _ in range(n_iters):
        g = _subgrad(Xx, nux, mask, xp)  # d x R
        gn = xp.linalg.norm(g, axis=0, keepdims=True) + 1e-12
        br = g / gn  # best-response candidate
        gd = _normalize_cols(U + lr * (g / gn), xp)  # smooth-step candidate
        vbr, _, mbr = _objective(Xx, nux, br, xp)
        vgd, _, mgd = _objective(Xx, nux, gd, xp)
        take_br = vbr >= vgd
        U = xp.where(take_br[None, :], br, gd)
        vals = xp.where(take_br, vbr, vgd)
        mask = xp.where(take_br[None, :], mbr, mgd)
        improved = vals > best_vals
        best_vals = xp.where(improved, vals, best_vals)
        best_U = xp.where(improved[None, :], U, best_U)
    return best_vals, best_U


def _factored_sdp_round(Xx, absnu, xp, rng_seed, r=16, n_power=12, n_hyper=64):
    """Factored Burer--Monteiro solve of max_{L: unit cols} ||L B||_F^2  (B = diag(|nu|)X),
    then Goemans--Williamson hyperplane rounding -> candidate directions (d x m).
    No n x n matrix is ever formed; cost O(n r d) per power step."""
    n, _d = Xx.shape
    B = absnu[:, None] * Xx  # n x d
    rs = np.random.default_rng(rng_seed)
    L = xp.asarray(rs.standard_normal((r, n)), dtype=Xx.dtype)
    L = L / (xp.linalg.norm(L, axis=0, keepdims=True) + 1e-12)  # unit columns
    for _ in range(n_power):
        LB = L @ B  # r x d
        grad = LB @ B.T  # r x n   (= L (B B^T))
        L = grad / (xp.linalg.norm(grad, axis=0, keepdims=True) + 1e-12)
    # rounding: eps = sign(L^T g), direction ~ B^T eps
    Gh = xp.asarray(rs.standard_normal((r, n_hyper)), dtype=Xx.dtype)
    E = xp.sign(L.T @ Gh)  # n x m
    E = xp.where(E == 0, xp.asarray(1.0, dtype=Xx.dtype), E)
    Dirs = B.T @ E  # d x m  = sum_i |nu_i| eps_i x_i
    return _normalize_cols(Dirs, xp)


def _seed_directions(Xx, nux, xp, n_random, rng_seed, svd_k=4):
    """Structured + random seed directions (d x R)."""
    _n, d = Xx.shape
    seeds = []
    c0 = Xx.T @ nux
    nc = float(xp.linalg.norm(c0))
    if nc > 0:
        seeds.append(c0 / nc)
        seeds.append(-c0 / nc)
    # top-k eigvecs of X^T diag(nu)^2 X  (= left singular vecs of X^T diag(nu))
    try:
        M = nux[:, None] * Xx  # diag(nu) X  (n x d)
        S = M.T @ M  # d x d
        _w, Q = eigh_small(S)  # CPU float64
        Qd = xp.asarray(Q, dtype=Xx.dtype)
        for j in range(1, min(svd_k, d) + 1):
            seeds.append(Qd[:, -j])
            seeds.append(-Qd[:, -j])
    except Exception:
        pass
    rs = np.random.default_rng(rng_seed)
    if n_random > 0:
        R = xp.asarray(rs.standard_normal((d, n_random)), dtype=Xx.dtype)
        seeds.append(R)
    Useed = xp.concatenate([s.reshape(d, -1) for s in seeds], axis=1)
    return _normalize_cols(Useed, xp)


def _price_core(
    Xx,
    nu,
    xp,
    n_random,
    n_iters,
    lr,
    use_sdp,
    sdp_r,
    sdp_power,
    sdp_hyper,
    rng_seed,
    use_softplus=False,
):
    """Run seeds (+ optional rounding/softplus) through batched ascent on the true
    objective.  Returns (best_vals (R,), best_U (d x R)) on device, better-sign oriented."""
    nux = xp.asarray(nu, dtype=Xx.dtype)
    absnu = xp.abs(nux)
    cands = [_seed_directions(Xx, nux, xp, n_random, rng_seed)]
    if use_sdp:
        cands.append(
            _factored_sdp_round(
                Xx, absnu, xp, rng_seed, r=sdp_r, n_power=sdp_power, n_hyper=sdp_hyper
            )
        )
    U0 = xp.concatenate(cands, axis=1)
    vals0, _, _ = _objective(Xx, nux, U0, xp)
    valsN, _, _ = _objective(Xx, nux, -U0, xp)
    flip = valsN > vals0
    U0 = xp.where(flip[None, :], -U0, U0)
    if use_softplus:  # add softplus-annealed polishings of the seeds
        Usp = _softplus_polish(Xx, nux, U0, xp)
        U0 = xp.concatenate([U0, Usp], axis=1)
    return _batched_ascent(Xx, nux, U0, xp, n_iters=n_iters, lr=lr)


def _get_Xdev(X, device, xp, _Xdev):
    if _Xdev is not None:
        return _Xdev
    dt = xp.float32 if device == "gpu" else xp.float64
    return xp.asarray(X, dtype=dt)


def price(
    X,
    nu,
    device="cpu",
    n_random=24,
    n_iters=60,
    lr=0.3,
    use_sdp=True,
    sdp_r=16,
    sdp_power=12,
    sdp_hyper=64,
    rng_seed=0,
    _Xdev=None,
    use_softplus=False,
):
    """Lower bound on rho(nu); returns (value, u (np), pattern (np int8))."""
    xp = get_xp(device)
    Xx = _get_Xdev(X, device, xp, _Xdev)
    best_vals, best_U = _price_core(
        Xx,
        nu,
        xp,
        n_random,
        n_iters,
        lr,
        use_sdp,
        sdp_r,
        sdp_power,
        sdp_hyper,
        rng_seed,
        use_softplus,
    )
    bi = int(xp.argmax(best_vals))
    u = to_cpu(best_U[:, bi])
    s = (to_cpu(Xx @ best_U[:, bi]) >= 0).astype(np.int8)
    return float(best_vals[bi]), u, s


def price_topk(
    X,
    nu,
    k=4,
    device="cpu",
    n_random=24,
    n_iters=60,
    lr=0.3,
    use_sdp=False,
    sdp_r=16,
    sdp_power=12,
    sdp_hyper=64,
    rng_seed=0,
    _Xdev=None,
    dedup_cos=0.999,
    use_softplus=False,
):
    """Multiple pricing: return up to k distinct high-value directions.

    Returns dict(dirs=[(value, u_np), ...] sorted desc, best=top value)."""
    xp = get_xp(device)
    Xx = _get_Xdev(X, device, xp, _Xdev)
    best_vals, best_U = _price_core(
        Xx,
        nu,
        xp,
        n_random,
        n_iters,
        lr,
        use_sdp,
        sdp_r,
        sdp_power,
        sdp_hyper,
        rng_seed,
        use_softplus,
    )
    order = to_cpu(xp.argsort(best_vals))[::-1]
    vals = to_cpu(best_vals)
    Ucpu = to_cpu(best_U)
    chosen = []
    for idx in order:
        u = Ucpu[:, idx]
        nrm = np.linalg.norm(u)
        if nrm < 1e-12:
            continue
        u = u / nrm
        if any(abs(float(u @ w)) > dedup_cos for _, w in chosen):
            continue
        chosen.append((float(vals[idx]), u))
        if len(chosen) >= k:
            break
    return dict(dirs=chosen, best=(chosen[0][0] if chosen else 0.0))


def price_pm(X, nu, device="cpu", **kw):
    """Price BOTH +nu and -nu (sharing the device copy of X).
    Returns dict rho_plus,u_plus,s_plus, rho_minus,u_minus,s_minus."""
    xp = get_xp(device)
    dt = xp.float32 if device == "gpu" else xp.float64
    Xx = xp.asarray(X, dtype=dt)
    vp, up, sp = price(X, nu, device=device, _Xdev=Xx, **kw)
    vm, um, sm = price(X, -nu, device=device, _Xdev=Xx, **kw)
    return dict(rho_plus=vp, u_plus=up, s_plus=sp, rho_minus=vm, u_minus=um, s_minus=sm)
