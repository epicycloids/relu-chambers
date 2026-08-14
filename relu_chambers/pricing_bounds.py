"""Low-rank SDP estimates for upper-bounding ReLU pricing.

For nonnegative weights w >= 0,
    rho(w) = max_{||u||<=1} w^T (Xu)_+  =  max_{s in {0,1}^n} || X^T (s ⊙ w) ||_2
with NO relaxation: for w >= 0 the realizable pattern s = 1{Xu > 0} attains the
free Boolean maximum.  Hence with nu_+ = max(nu, 0):

    rho(nu) <= rho(nu_+) = sqrt( max_{s in {0,1}^n} s^T G s ),
    G = diag(nu_+) X X^T diag(nu_+).

Via s = (1+eps)/2 this is a {+-1} problem on the AUGMENTED factored matrix
Bhat = [ X^T nu_+ ; diag(nu_+) X ] in R^{(n+1) x d}:

    max_s s^T G s = (1/4) max_{z in {+-1}^{n+1}} z^T Bhat Bhat^T z
                  <= (1/4) SDP(Bhat Bhat^T)        [Nesterov: <= (pi/2) max_z]

so  UB_pos(nu) = (1/2) sqrt(SDP(Bhat Bhat^T)) >= rho(nu), within sqrt(pi/2)
~ 1.2533 of rho(nu_+); the only true loss is rho(nu) <= rho(nu_+).  The
absolute-residual comparison bound in ``reference.absolute_pricing_bound``
also drops the sign, splits the linear and Boolean parts, and incurs the pi/2
factor.  The positive-part bound loses only once.  ``estimate_pricing_bounds``
reports the minimum enabled bound for each sign.

Evaluation: the SDP is bounded through the d x d concave dual
    SDP(B B^T) = max_{Z >= 0} 2 sum_i sqrt(b_i^T Z b_i) - tr(Z),
and in exact arithmetic, from positive c_i, the rescaling
M = sum_i b_i b_i^T / c_i gives SDP <= lambda_max(M) * sum_i c_i.
The Z-ascent runs on a HOST row sketch (it only proposes Z); the final
rescaling is one full-data engine pass, so the mathematical construction never
depends on the sketch.  The current implementation uses ordinary round-to-nearest
accumulation and ``eigvalsh``; it does not outward-enclose their errors.  Returned
values are therefore numerical estimates of the exact-arithmetic bound, not
rigorous floating-point certificates.  Cost: O(m d^2) per ascent step +
O(n d^2) per validation."""

from __future__ import annotations

import numpy as np

from .kernels import importance_sample


def _psd_project(M):
    w, Q = np.linalg.eigh(0.5 * (M + M.T))
    w = np.clip(w, 0.0, None)
    return (Q * w) @ Q.T


def _valid_bound_host(B, Z, eps=1e-30):
    c = np.sqrt(np.clip(np.einsum("ij,ij->i", B @ Z, B), eps, None))
    M = (B / c[:, None]).T @ B
    lam_max = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
    return lam_max * float(c.sum())


def _dual_value_grad_host(B, Z, eps=1e-30):
    """Exact-form small-cone objective and its numerical gradient."""
    c = np.sqrt(np.clip(np.einsum("ij,ij->i", B @ Z, B), eps, None))
    value = 2.0 * float(c.sum()) - float(np.trace(Z))
    grad = (B * (1.0 / c)[:, None]).T @ B - np.eye(B.shape[1])
    return value, grad


def _dual_ascent_host(B, n_steps=20, Z0=None, eps=1e-30):
    _n, d = B.shape
    if Z0 is not None:
        Z = np.asarray(Z0, float)
    else:
        bn = np.sqrt(np.clip(np.einsum("ij,ij->i", B, B), eps, None))
        Z = ((bn.sum() / max(d, 1)) ** 2) * np.eye(d)
    best = _valid_bound_host(B, Z)
    for _ in range(n_steps):
        _, grad = _dual_value_grad_host(B, Z, eps=eps)
        gn = float(np.linalg.norm(grad)) + 1e-12
        s = (float(np.trace(Z)) / max(d, 1) + 1.0) / gn
        improved = False
        for _ls in range(4):
            Zt = _psd_project(Z + s * grad)
            v = _valid_bound_host(B, Zt)
            if v < best - 1e-15 * max(1.0, abs(best)):
                best, Z, improved = v, Zt, True
                break
            s *= 0.4
        if not improved:
            break
    return best, Z


def estimate_pricing_bounds(
    engine,
    nu_host,
    n_steps=20,
    warm=None,
    sketch_m=32768,
    rng=None,
    forms=("joint",),
    row_norms=None,
):
    """Numerical estimates of exact-arithmetic upper bounds for both prices.

    Returns dict(ub_plus, ub_minus, warm).  forms ⊆ {"joint","split"}; reported
    bound per sign = min over enabled forms.  The Z-ascent runs on a host
    sketch and every value receives a full-data engine pass.  Ordinary floating
    arithmetic is not outward-certified, so the returned quantities are
    diagnostic rather than rigorous numerical certificates."""
    warm = warm or {}
    rng = rng or np.random.default_rng(0)
    n, d = engine.n, engine.d
    out = {}
    for sign, key in ((1.0, "plus"), (-1.0, "minus")):
        w = np.maximum(sign * nu_host, 0.0)
        if w.sum() <= 0:
            out[f"ub_{key}"] = 0.0
            continue
        b0 = engine.xtv(engine.scatter_vec(w))  # X^T w (full data, f64)
        # ---- host sketch for the Z-ascent ----
        m = sketch_m if (sketch_m and 2 * sketch_m < n) else n
        if m < n:
            if row_norms is None:
                row_norms = np.sqrt(np.einsum("ij,ij->i", engine._Xhost, engine._Xhost))
            idx, swt = importance_sample(w, row_norms, m, rng)
            Xr, _ = engine.gather_rows(idx)
            Bs = (w[idx] * swt)[:, None] * Xr  # 1-homog. in row scale
        else:
            Bs = w[:, None] * engine._Xhost
        ubs = []
        if "joint" in forms:
            Bsk = np.concatenate([b0[None, :], Bs], axis=0)
            _, Z = _dual_ascent_host(Bsk, n_steps=n_steps, Z0=warm.get(("joint", key)))
            warm[("joint", key)] = Z
            csum, M = engine.cert_validate(w, Z, b0)
            lam_max = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
            ubs.append(0.5 * np.sqrt(max(lam_max * csum, 0.0)))
        if "split" in forms:
            _, Z = _dual_ascent_host(Bs, n_steps=n_steps, Z0=warm.get(("split", key)))
            warm[("split", key)] = Z
            # full-data validation WITHOUT the augmented row: pass b0=0
            csum, M = engine.cert_validate(w, Z, np.zeros(d))
            # remove the (eps-guarded) b0=0 contribution: c0 ~ 1e-150, negligible
            lam_max = float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])
            ubs.append(
                0.5 * float(np.linalg.norm(b0))
                + 0.5 * np.sqrt(max(lam_max * csum, 0.0))
            )
        out[f"ub_{key}"] = float(min(ubs))
    out["warm"] = warm
    return out


def dual_gap(y, nu, lam, ub_plus, ub_minus, obj):
    """Numerical primal-dual gap estimate from supplied pricing bounds.

    ``theta * nu`` maximizes the ray dual objective over the conservatively
    certified segment ``0 <= theta <= lam / max(ub_plus, ub_minus)``.  A loose
    upper bound can make that segment smaller than the full dual-feasible ray.
    Returns ``(gap, theta)``.  The bound is rigorous in exact arithmetic when
    both supplied prices are true upper bounds; this routine does not direct
    floating-point rounding.
    """
    nn = float(nu @ nu)
    if nn <= 0:
        return max(obj, 0.0), 0.0
    ub = max(ub_plus, ub_minus, 1e-300)
    theta = max(min(float(nu @ y) / nn, lam / ub), 0.0)
    D = theta * float(nu @ y) - 0.5 * theta * theta * nn
    return max(obj - D, 0.0), theta
