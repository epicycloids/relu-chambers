"""Scalable absolute-residual upper bound for ReLU pricing.

Goal: an always-valid upper bound  UB(nu) >= rho(nu) = max_{||u||<=1} nu^T (Xu)_+
that costs O(n d^2) (matrix-free in n), so it replaces the baseline's n x n
Goemans--Williamson SDP (an O(n^2) variable, O(n^3) solve via SCS) with something
that runs at n = 10^6 on a single GPU.

Derivation.  As in the paper,
    rho(nu) <= 1/2 ||X^T nu||_2 + 1/2 * M*,
    M* = max_{eps in {+-1}^n} || X^T diag(|nu|) eps ||_2 = sqrt( max_eps eps^T G eps ),
with G = B B^T, B = diag(|nu|) X (rows b_i = |nu_i| x_i in R^d), so G is PSD of
RANK <= d.  The Boolean maximum is bounded by the Max-Cut SDP and *its dual*:
    M*^2 <= SDP(G) = min_{eta > 0 : diag(eta) >= B B^T} 1^T eta.
Any dual-feasible eta gives a valid exact-arithmetic bound.  We never form the
n x n primal Y.
Instead we work with the d x d multiplier Z >= 0: by Lagrangian duality the dual
SDP equals
    SDP(G) = max_{Z >= 0}  g(Z),   g(Z) = 2 * sum_i sqrt(b_i^T Z b_i) - tr(Z),
a *concave* program over the small d x d PSD cone (matrix-free in n: it needs only
diag(B Z B^T) and B^T diag(.) B, both batched matmuls).  Maximizing g(Z) drives Z
to the optimum; and from ANY Z we recover a dual-feasible eta in exact arithmetic
by the
rescaling
    c_i = sqrt(b_i^T Z b_i),  M = sum_i b_i b_i^T / c_i,  eta_i = lambda_max(M) * c_i,
which satisfies diag(eta) >= B B^T by construction, so 1^T eta = lambda_max(M) *
sum_i c_i  >=  SDP(G)  >=  M*^2.  Hence

    UB(nu) = 1/2 ||X^T nu|| + 1/2 sqrt( lambda_max(M) * sum_i c_i )   >=  rho(nu),

valid for every Z in exact arithmetic, and tight (-> the SDP, i.e. within
Nesterov's sqrt(pi/2)) as Z -> Z*.  The implementation uses ordinary floating
point arithmetic, so its returned value is a numerical estimate rather than an
outward-rounded certificate.  Cost per ascent step: O(n d^2 + d^3).
"""

from __future__ import annotations

import numpy as np

from ..array_backend import eigvalsh_small, get_xp, psd_project_small


def _diag_BZB(B, Z, xp):
    """c2_i = b_i^T Z b_i = diag(B Z B^T), in O(n d^2)."""
    BZ = B @ Z  # n x d
    return xp.einsum("ij,ij->i", BZ, B)  # n


def estimate_absolute_pricing_bound(
    X, nu, device="cpu", n_steps=25, eps=1e-12, return_Z=False, Z0=None
):
    """Estimate an exact-arithmetic upper bound; O(n d^2) per step.

    Maximizes the d x d concave dual g(Z) by projected gradient ascent and returns
    the best bound found.  Set n_steps=0 for the closed-form scalar-Z bound
    (still valid in exact arithmetic, but looser).  Returns ``(ub, info)``,
    where ``info["sdp_bound"]`` is the numerical value of ``1^T eta``.
    """
    xp = get_xp(device)
    cdt = xp.float32 if device == "gpu" else xp.float64  # fp32 on GPU (robust here)
    Xx = xp.asarray(X, dtype=cdt)
    nux = xp.asarray(nu, dtype=cdt)
    absnu = xp.abs(nux)
    B = absnu[:, None] * Xx  # n x d, rows |nu_i| x_i
    n, d = B.shape
    lin = 0.5 * float(xp.linalg.norm(Xx.T @ nux))

    # b_i^T b_i = ||b_i||^2
    bnorm2 = xp.einsum("ij,ij->i", B, B)  # n
    sum_bnorm = float(xp.sqrt(bnorm2).sum())

    # --- initialise Z = alpha I with the optimal scalar alpha ---
    # g(alpha I) = 2 sqrt(alpha) sum||b_i|| - alpha d  -> alpha* = (sum||b_i||/d)^2
    dt = Xx.dtype
    if Z0 is not None:
        Z = xp.asarray(Z0, dtype=dt)
    else:
        alpha = (sum_bnorm / max(d, 1)) ** 2
        Z = (alpha * xp.eye(d)).astype(dt)
    Id = xp.eye(d, dtype=dt)

    def valid_bound(Zc):
        c2 = _diag_BZB(B, Zc, xp)
        c = xp.sqrt(xp.clip(c2, eps, None))
        invc = 1.0 / c
        M = (B * invc[:, None]).T @ B  # d x d  = sum_i b_i b_i^T / c_i
        lam_max = float(eigvalsh_small(M)[-1])  # CPU float64
        sdp_bound = lam_max * float(c.sum())  # 1^T eta, valid upper bound on SDP(G)
        return sdp_bound, c

    best_sdp, _ = valid_bound(Z)
    # --- projected-gradient ascent on g(Z) = 2 sum sqrt(b^T Z b) - tr Z ---
    for t in range(n_steps):
        c2 = _diag_BZB(B, Z, xp)
        c = xp.sqrt(xp.clip(c2, eps, None))
        grad = (B * (0.5 / c)[:, None]).T @ B - Id  # ∇g = sum b b^T/(2 sqrt) - I
        gnorm = float(xp.linalg.norm(grad)) + 1e-12
        s = (float(xp.trace(Z)) / max(d, 1) + 1.0) / gnorm
        Znew = psd_project_small(Z + s * grad, xp, Z.dtype)
        sdp_new, _ = valid_bound(Znew)
        if sdp_new < best_sdp:
            best_sdp = sdp_new
            Z = Znew
        else:
            # gradient overshoot: backtrack a couple of times
            s2, improved = s, False
            for _ in range(3):
                s2 *= 0.4
                Zt = psd_project_small(Z + s2 * grad, xp, Z.dtype)
                sdp_t, _ = valid_bound(Zt)
                if sdp_t < best_sdp:
                    best_sdp, Z, improved = sdp_t, Zt, True
                    break
            if not improved:
                break

    ub = lin + 0.5 * float(np.sqrt(max(best_sdp, 0.0)))
    info = dict(sdp_bound=float(best_sdp), lin=lin, n=int(n), d=int(d))
    if return_Z:
        info["Z"] = Z
    return ub, info


def certificate_cheap(X, nu, device="cpu"):
    """Closed-form O(n d^2) valid bound (spectral relaxation), the loose backstop:
    M*^2 <= n * sigma_max(diag(|nu|) X)^2  =>  UB = 1/2||X^T nu|| + 1/2 sqrt(n) sigma."""
    xp = get_xp(device)
    Xx = xp.asarray(X)
    nux = xp.asarray(nu)
    Bn = xp.abs(nux)[:, None] * Xx
    sigma = float(xp.linalg.norm(Bn, 2))
    lin = 0.5 * float(xp.linalg.norm(Xx.T @ nux))
    n = X.shape[0]
    return lin + 0.5 * np.sqrt(n) * sigma
