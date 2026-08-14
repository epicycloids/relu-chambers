"""The pricing oracle and the SDP optimality certificate.

By Theorem 1, separation over all chambers collapses to the single problem

    rho(nu) = max_{||u||<=1}  nu^T (X u)_+ .

A restricted solution with residual nu is globally optimal iff
rho(nu) <= lam and rho(-nu) <= lam.

This module provides:
  * pricing_exact   -- exact value+argmax by enumerating the arrangement (small d)
  * pricing_heuristic -- gradient ascent + best-response restarts (any d); a
                         LOWER bound on rho (it returns a real direction u)
  * pricing_sdp_rounded -- randomized Goemans-Williamson/Nesterov rounding of the
                         certificate SDP; a realized LOWER bound motivated by
                         the sqrt(2/pi) expected approximation result for the
                         symmetrized correlation
  * certificate_sdp -- a poly-time UPPER bound UB(nu) >= rho(nu) (Theorem 6)
  * price_and_certify_sdp -- one PSD solve -> rounded lower bounds for +/-nu and
                         the shared upper-bound diagnostic
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..synthetic_data import pattern_of, relu


# --------------------------------------------------------------------------- #
#  Exact pricing by chamber enumeration (intended for small d).
# --------------------------------------------------------------------------- #
def enumerate_patterns(X, n_dense=200000, rng=None):
    """Enumerate realizable patterns by dense Gaussian sampling.

    For small d the number of chambers is O(n^{d-1}); dense sampling recovers
    all of them with high probability, giving an essentially exact oracle.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = X.shape[1]
    seen = {}
    done = 0
    batch = 20000
    while done < n_dense:
        b = min(batch, n_dense - done)
        G = rng.standard_normal((d, b))
        A = (X @ G >= 0).astype(np.int8)
        for c in range(b):
            s = A[:, c]
            seen[s.tobytes()] = s.copy()
        done += b
    return list(seen.values())


def _conic_support(X, c, sigma):
    """max c^T u s.t. ||u||<=1, diag(sigma) X u >= 0   (small SOCP)."""
    d = X.shape[1]
    u = cp.Variable(d)
    cons = [cp.multiply(sigma, X @ u) >= 0, cp.norm(u, 2) <= 1]
    prob = cp.Problem(cp.Maximize(c @ u), cons)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        prob.solve(solver="SCS")
    if prob.value is None:
        return -np.inf, None
    return float(prob.value), (
        np.asarray(u.value).ravel() if u.value is not None else None
    )


def pricing_exact(X, nu, patterns=None, rng=None):
    """Exact rho(nu)=max_{||u||<=1} nu^T (Xu)_+ via chamber enumeration.

    Returns (value, u_argmax, best_pattern).
    """
    if patterns is None:
        patterns = enumerate_patterns(X, rng=rng)
    best_val, best_u, best_s = -np.inf, None, None
    for s in patterns:
        if np.sum(s) == 0:
            continue
        svec = np.asarray(s, dtype=float)
        sigma = np.where(svec > 0, 1.0, -1.0)
        c = X.T @ (svec * nu)  # X^T D_s nu
        val, u = _conic_support(X, c, sigma)
        if val > best_val:
            best_val, best_u, best_s = val, u, s
    return best_val, best_u, best_s


# --------------------------------------------------------------------------- #
#  Heuristic pricing: gradient ascent + best-response local search.
# --------------------------------------------------------------------------- #
def _ascent(X, nu, u0, n_iters=200, lr=0.3):
    """Projected (sub)gradient ascent on g(u)=nu^T (Xu)_+ over the unit ball,
    interleaved with best-response steps. Returns (best_val, best_u)."""
    u = u0 / (np.linalg.norm(u0) + 1e-12)
    best_val = float(nu @ relu(X @ u))
    best_u = u.copy()
    for t in range(n_iters):
        a = X @ u
        mask = (a >= 0).astype(float)
        g = X.T @ (nu * mask)  # subgradient = X^T D_s nu
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        # best-response candidate (normalize the gradient direction)
        u_br = g / gn
        v_br = float(nu @ relu(X @ u_br))
        # small gradient step candidate
        u_gd = u + lr * g / gn
        u_gd = u_gd / (np.linalg.norm(u_gd) + 1e-12)
        v_gd = float(nu @ relu(X @ u_gd))
        # pick the better move
        if v_br >= v_gd:
            u, v = u_br, v_br
        else:
            u, v = u_gd, v_gd
        if v > best_val:
            best_val, best_u = v, u.copy()
    return best_val, best_u


def pricing_heuristic(X, nu, n_restarts=30, n_iters=200, rng=None, lr=0.3):
    """Lower bound on rho(nu) via multi-start local search.

    Returns (value, u_argmax, pattern). Seeds include the raw residual
    correlation direction X^T nu and the leading singular vector of X^T diag(nu).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = X.shape[1]
    seeds = []
    c0 = X.T @ nu
    if np.linalg.norm(c0) > 0:
        seeds.append(c0)
        seeds.append(-c0)
    # leading singular direction of X^T diag(nu)
    try:
        M = X.T * nu[None, :]
        U_, _, _ = np.linalg.svd(M, full_matrices=False)
        seeds.append(U_[:, 0])
        seeds.append(-U_[:, 0])
    except Exception:
        pass
    n_rand = max(0, n_restarts - len(seeds))
    for _ in range(n_rand):
        seeds.append(rng.standard_normal(d))

    best_val, best_u = -np.inf, None
    for s0 in seeds:
        v, u = _ascent(X, nu, s0, n_iters=n_iters, lr=lr)
        if v > best_val:
            best_val, best_u = v, u
    return best_val, best_u, pattern_of(X, best_u)


# --------------------------------------------------------------------------- #
#  SDP relaxation of the pricing problem (Theorem 6).
#
#  Both the optimality CERTIFICATE (an upper bound on rho) and the RANDOMIZED
#  ROUNDING oracle (a lower bound: an actual direction u) come from the *same*
#  PSD program, so the chamber reference solves it once per residual and gets
#  both bounds.
# --------------------------------------------------------------------------- #
def _sdp_relax(X, nu, solver="SCS"):
    """Solve the Boolean-quadratic SDP relaxation underlying the pricing bound.

    With Gt = diag(|nu|) X X^T diag(|nu|) >= 0,
        SDP(Gt) = max_{Y>=0, diag(Y)=1} <Gt, Y>  >=  max_{eps in {+-1}^n} eps^T Gt eps,
    and by Nesterov SDP(Gt) <= (pi/2) max_eps eps^T Gt eps.  Returns (sdp_val, Y, Gt).
    Gt depends on nu only through |nu|, so the same solve serves +nu and -nu.
    """
    n = X.shape[0]
    Xn = np.abs(nu)[:, None] * X  # diag(|nu|) X
    Gt = Xn @ Xn.T  # n x n PSD
    Gt = 0.5 * (Gt + Gt.T)
    Y = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(cp.trace(Gt @ Y)), [cp.diag(Y) == 1])
    try:
        prob.solve(solver=solver)
    except Exception:
        prob.solve(solver="CLARABEL")
    if prob.value is None:
        return np.inf, None, Gt
    sdp_val = max(prob.value, 0.0)
    Yval = np.asarray(Y.value) if Y.value is not None else None
    return float(sdp_val), Yval, Gt


def _ub_from_sdp(X, nu, sdp_val):
    """UB(nu) = 0.5 ||X^T nu|| + 0.5 sqrt(SDP)  >= rho(nu)  (Theorem 6)."""
    lin = 0.5 * np.linalg.norm(X.T @ nu)
    return float(lin + 0.5 * np.sqrt(sdp_val))


def certificate_sdp(X, nu, solver="SCS"):
    """Upper bound UB(nu) >= rho(nu) = max_{||u||<=1} nu^T (Xu)_+  (Theorem 6).

    Valid for ALL nu and tight up to Nesterov's pi/2 constant. UB(nu)=UB(-nu).
    Returns (ub, sdp_val).
    """
    sdp_val, _, _ = _sdp_relax(X, nu, solver=solver)
    return _ub_from_sdp(X, nu, sdp_val), float(sdp_val)


# --------------------------------------------------------------------------- #
#  Randomized rounding oracle (Goemans-Williamson / Nesterov).
# --------------------------------------------------------------------------- #
def _factor_psd(Y, tol=1e-9):
    """Return L (r x n) with L^T L = Y_clip, columns the rounding vectors l_i.
    Y from an SCS solve may be slightly indefinite; clip eigenvalues at 0."""
    Y = 0.5 * (Y + Y.T)
    w, Q = np.linalg.eigh(Y)
    w = np.clip(w, 0.0, None)
    keep = w > tol * (w.max() + 1e-18)
    L = (np.sqrt(w[keep])[:, None]) * Q[:, keep].T  # (r x n)
    return L


def _round_directions(X, absnu, Y, n_rounds, rng):
    """Goemans-Williamson hyperplane rounding of the SDP solution Y.

    For each random Gaussian g, eps = sign(L^T g) in {+-1}^n; the induced unit
    direction is u_eps ~ X^T diag(|nu|) eps = sum_i |nu_i| eps_i x_i, which
    achieves symmetrized correlation sum_i |nu_i| |x_i^T u| >= sqrt(eps^T Gt eps).
    Returns a (m x d) array of unique candidate unit directions.
    """
    if Y is None:
        return np.zeros((0, X.shape[1]))
    L = _factor_psd(Y)  # r x n
    r = L.shape[0]
    if r == 0:
        return np.zeros((0, X.shape[1]))
    G = rng.standard_normal((r, n_rounds))  # r x m
    E = np.sign(L.T @ G)  # n x m  in {-1,0,+1}
    E[E == 0] = 1.0
    Dirs = X.T @ (absnu[:, None] * E)  # d x m  = sum_i |nu_i| eps_i x_i
    nrm = np.linalg.norm(Dirs, axis=0)
    good = nrm > 1e-12
    Dirs = (Dirs[:, good] / nrm[good]).T  # m' x d  (unit rows)
    return Dirs


def pricing_sdp_rounded(
    X,
    nu,
    n_rounds=64,
    n_polish=6,
    polish_iters=80,
    lr=0.3,
    sdp=None,
    solver="SCS",
    rng=None,
    extra_seeds=True,
):
    """Lower bound on rho(nu) via SDP + randomized hyperplane rounding.

    One SDP solve (Theorem 6) produces, by Goemans-Williamson rounding, a set of
    globally-informed candidate directions; each is evaluated on the EXACT
    objective nu^T (Xu)_+ (always a valid lower bound), and the best `n_polish`
    are refined by the same local ascent used in the heuristic oracle.

    `sdp` may be a precomputed (sdp_val, Y) tuple to avoid re-solving (the SDP
    depends only on |nu|, so it is shared between +nu and -nu).
    Returns (value, u_argmax, pattern).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    absnu = np.abs(nu)
    if sdp is None:
        _, Y, _ = _sdp_relax(X, nu, solver=solver)
    else:
        _sdp_val, Y = sdp

    cands = list(_round_directions(X, absnu, Y, n_rounds, rng))
    if extra_seeds:
        # The structured seeds the heuristic oracle also uses (residual
        # correlation X^T nu and the leading singular vector of X^T diag(nu)).
        # Including them makes the rounded oracle dominate the heuristic's
        # deterministic search; GW rounding is the extra randomized ingredient
        # that escapes the local optima those seeds fall into on hard instances.
        c0 = X.T @ nu
        if np.linalg.norm(c0) > 0:
            cands.append(c0 / np.linalg.norm(c0))
            cands.append(-c0 / np.linalg.norm(c0))
        try:
            U_, _, _ = np.linalg.svd(X.T * nu[None, :], full_matrices=False)
            cands.append(U_[:, 0])
            cands.append(-U_[:, 0])
        except np.linalg.LinAlgError:
            pass

    if not cands:
        return pricing_heuristic(X, nu, rng=rng)

    C = np.array(cands)  # m x d, unit rows
    # exact objective on every candidate AND its negation (rounding is sign-free)
    A = C @ X.T  # m x n
    vals_pos = relu(A) @ nu  # value at +u
    vals_neg = relu(-A) @ nu  # value at -u
    flip = vals_neg > vals_pos
    vals = np.where(flip, vals_neg, vals_pos)
    C = np.where(flip[:, None], -C, C)

    order = np.argsort(vals)[::-1]
    best_val, best_u = float(vals[order[0]]), C[order[0]].copy()
    for idx in order[: max(1, n_polish)]:
        v, u = _ascent(X, nu, C[idx], n_iters=polish_iters, lr=lr)
        if v > best_val:
            best_val, best_u = v, u
    return best_val, best_u, pattern_of(X, best_u)


def price_and_certify_sdp(
    X,
    nu,
    n_rounds=64,
    n_polish=6,
    polish_iters=80,
    lr=0.3,
    solver="SCS",
    rng=None,
    hybrid=True,
    heur_restarts=8,
    heur_iters=60,
):
    """One SDP solve -> lower bounds for BOTH +nu and -nu plus the shared
    upper-bound certificate UB(nu)=UB(-nu).  This is used by the chamber
    reference with oracle='sdp_rounded'.

    With hybrid=True (default) the lower bound is max(rounded, heuristic): the
    randomized rounding is combined with multi-start local search, so the
    returned score is the larger of the two realized scores.  The single SDP
    solve is reused for the certificate and (via Y) for rounding both signs.

    Returns dict: rho_plus,u_plus,s_plus, rho_minus,u_minus,s_minus, ub, sdp_val.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sdp_val, Y, _ = _sdp_relax(X, nu, solver=solver)
    ub = _ub_from_sdp(X, nu, sdp_val)

    def price_one(sign_nu):
        vr, ur, sr = pricing_sdp_rounded(
            X,
            sign_nu,
            n_rounds=n_rounds,
            n_polish=n_polish,
            polish_iters=polish_iters,
            lr=lr,
            sdp=(sdp_val, Y),
            rng=rng,
        )
        if hybrid:
            vh, uh, sh = pricing_heuristic(
                X, sign_nu, n_restarts=heur_restarts, n_iters=heur_iters, rng=rng
            )
            if vh > vr:
                return vh, uh, sh
        return vr, ur, sr

    vp, up, sp = price_one(nu)
    vm, um, sm = price_one(-nu)
    return dict(
        rho_plus=vp,
        u_plus=up,
        s_plus=sp,
        rho_minus=vm,
        u_minus=um,
        s_minus=sm,
        ub=float(ub),
        sdp_val=float(sdp_val),
    )


def certificate_cheap(X, nu):
    """A cheap O(nd^2) VALID upper bound on rho(nu), the spectral relaxation of
    the same Boolean quadratic:
        max_eps eps^T Gt eps <= n * ||diag(|nu|) X||_op^2,
    hence rho(nu) <= 0.5||X^T nu|| + 0.5 sqrt(n) * sigma_max(diag(|nu|) X).
    Looser than the SDP by a factor up to sqrt(n) (the SDP replaces it by a
    constant via Nesterov), but needs no conic solve."""
    n = X.shape[0]
    Xn = np.abs(nu)[:, None] * X
    sigma_max = np.linalg.norm(Xn, 2)
    lin = 0.5 * np.linalg.norm(X.T @ nu)
    return float(lin + 0.5 * np.sqrt(n) * sigma_max)
