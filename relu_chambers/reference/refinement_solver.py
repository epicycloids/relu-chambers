"""Single-column refinement baseline for the planted comparison.

Each round alternates a numerical Lasso over the current ReLU atoms with
projected direction refinement, prunes inactive atoms, searches both residual
signs for new directions, and evaluates an absolute-residual bound.  The method
uses finite iteration budgets, heuristic pricing, and ordinary floating-point
arithmetic; its stopping diagnostics are not a global optimality certificate.
It supports NumPy or an optional single-device CuPy backend.
"""

from __future__ import annotations

import time

import numpy as np

from ..array_backend import get_xp, to_cpu, to_device
from .absolute_pricing_bound import estimate_absolute_pricing_bound
from .lasso import active_set_lasso
from .pricing_search import price_topk


def _dual_value(y, nu, lam, ub):
    theta = 1.0 if ub <= 1e-12 else min(1.0, lam / ub)
    return theta * float(nu @ y) - 0.5 * theta**2 * float(nu @ nu)


def _chamber_compress(Xdev, U, beta, xp, dtype):
    """Exact chamber compression (CAPE-CG idea): atoms that landed in the SAME
    chamber with the SAME sign are merged by a conic combination
        u_G = (sum_j |beta_j| u_j)/||.||,   beta_G = sign * ||sum_j |beta_j| u_j||,
    which leaves the prediction identical (ReLU is linear inside a chamber) and can
    only *decrease* the penalty (||sum|beta_j|u_j|| <= sum|beta_j|).  An exact
    improvement, not a heuristic -- it recovers the chamber-block group-norm behaviour
    while keeping the atom representation, and shrinks the working set."""
    P = U.shape[1]
    if P < 2:
        return U, beta, False
    Sf = (Xdev @ U >= 0).astype(dtype)  # n x P chamber indicators
    c = Sf.sum(axis=0)  # column bit-counts
    Gram = Sf.T @ Sf
    D = c[:, None] + c[None, :] - 2.0 * Gram  # pairwise Hamming distance
    Dc = to_cpu(D)
    bc = to_cpu(beta)
    Uc = to_cpu(U)
    sgn = np.sign(bc)
    sgn[sgn == 0] = 1.0
    same = Dc < 0.5  # identical chamber (integer Hamming==0)
    used = np.zeros(P, dtype=bool)
    newU, newbeta = [], []
    changed = False
    for i in range(P):
        if used[i]:
            continue
        grp = [j for j in range(P) if (not used[j]) and same[i, j] and sgn[j] == sgn[i]]
        used[grp] = True
        if len(grp) == 1:
            newU.append(Uc[:, i])
            newbeta.append(bc[i])
            continue
        vG = (np.abs(bc[grp])[None, :] * Uc[:, grp]).sum(axis=1)
        nrm = np.linalg.norm(vG)
        if nrm < 1e-30:
            continue
        newU.append(vG / nrm)
        newbeta.append(sgn[i] * nrm)
        changed = True
    if not changed:
        return U, beta, False
    Un = xp.asarray(np.stack(newU, axis=1), dtype=dtype)
    bn = xp.asarray(np.asarray(newbeta), dtype=dtype)
    return Un, bn, True


def _refine(Xdev, ydev, lam, U, A, S, xp, dtype, n_steps, lr0):
    """Device-resident fully-corrective refinement; returns (U, A, beta, nu, obj, S)."""
    beta, nu, obj, S, _ = active_set_lasso(A, ydev, lam, xp, S0=S)
    for _ in range(n_steps):
        Sm = (Xdev @ U >= 0).astype(dtype)  # n x P masks
        Gdir = (Xdev.T @ (nu[:, None] * Sm)) * xp.sign(beta)[None, :]
        gnrm = xp.linalg.norm(Gdir, axis=0, keepdims=True)
        if float(xp.linalg.norm(Gdir)) < 1e-10:
            break
        Gn = Gdir / (gnrm + 1e-30)
        improved, lr = False, lr0
        for _ls in range(5):
            Un = U + lr * Gn
            Un = Un / (xp.linalg.norm(Un, axis=0, keepdims=True) + 1e-30)
            An = xp.maximum(Xdev @ Un, 0.0)
            bn, nun, on, Sn, _ = active_set_lasso(An, ydev, lam, xp, S0=S)
            if on < obj - 1e-12 * max(1.0, abs(obj)):
                U, A, beta, nu, obj, S = Un, An, bn, nun, on, Sn
                improved = True
                break
            lr *= 0.4
        if not improved:
            break
    return U, A, beta, nu, obj, S


def solve_refinement_baseline(
    X,
    y,
    lam,
    device="cpu",
    max_iter=80,
    eps=1e-4,
    dtype=None,
    add_per_round=8,
    add_min=8,
    cos_dedup=0.9995,
    certify_every=3,
    cert_steps=10,
    oracle_kwargs=None,
    max_cols=None,
    refine_steps=2,
    refine_lr=0.5,
    stall_rtol=2e-4,
    patience=3,
    price_tol=0.02,
    compress=True,
    homotopy=0,
    homotopy_shrink=0.6,
    rng_seed=0,
    verbose=False,
):
    """Run refinement baseline.  Returns dict(obj, nu, beta, U, n_active, gap, history, ...)."""
    xp = get_xp(device)
    if dtype is None:
        # The master's active-set LASSO reads KKT correlations a_j^T r, which suffer
        # catastrophic cancellation in float32 once the residual is small (verified:
        # float32 master diverges from float64 by >10%).  So the master runs in
        # float64 on both CPU and GPU; the pricing oracle and the certificate -- the
        # n-heavy, cancellation-free, embarrassingly parallel parts -- run in float32
        # on the GPU (validated to agree with float64 to ~1e-6), which is where the
        # device throughput actually pays off.
        dtype = xp.float64
    n, d = X.shape
    ok = dict(n_random=20, n_iters=60, use_sdp=False)
    if oracle_kwargs:
        ok.update(oracle_kwargs)

    Xdev = to_device(X, device).astype(dtype)
    ydev = to_device(np.asarray(y, dtype=np.float64), device).astype(dtype)
    yv = np.asarray(y, dtype=np.float64)

    U = xp.zeros((d, 0), dtype=dtype)
    A = xp.zeros((n, 0), dtype=dtype)
    beta = xp.zeros(0, dtype=dtype)
    nu = ydev.copy()
    obj = 0.5 * float(yv @ yv)
    S = []
    hist = {
        "iter": [],
        "n_cols": [],
        "n_active": [],
        "obj": [],
        "rho_lb": [],
        "ub": [],
        "gap": [],
        "time": [],
    }
    t0 = time.time()
    # lambda-homotopy schedule (CAPE-CG idea): anneal lam from large (sparse, well-
    # conditioned) down to the target over `homotopy` rounds, then hold at target.
    mult0 = (1.0 / homotopy_shrink) ** homotopy if homotopy > 0 else 1.0
    for it in range(max_iter):
        lam_it = lam * max(1.0, mult0 * (homotopy_shrink**it)) if homotopy > 0 else lam
        at_target = lam_it <= lam * (1.0 + 1e-9)
        # ---- master: numerical Lasso + direction refinement ----
        if A.shape[1]:
            U, A, beta, nu, obj, S = _refine(
                Xdev, ydev, lam_it, U, A, S, xp, dtype, refine_steps, refine_lr
            )
            # prune zero-weight atoms; remap the active set to the kept columns
            bc = to_cpu(beta)
            keep = np.where(np.abs(bc) > 1e-9 * max(1.0, float(np.abs(bc).max())))[0]
            if 0 < len(keep) < A.shape[1]:
                idx = xp.asarray(keep)
                U, A, beta = U[:, idx], A[:, idx], beta[idx]
            # algebraic chamber compression, then one numerical re-fit
            if compress and U.shape[1] > 1:
                U, beta, did = _chamber_compress(Xdev, U, beta, xp, dtype)
                if did:
                    A = xp.maximum(Xdev @ U, 0.0)
                    beta, nu, obj, S, _ = active_set_lasso(A, ydev, lam_it, xp)
                else:
                    S = list(range(A.shape[1]))
            else:
                S = list(range(A.shape[1]))
        n_active = len(S)

        # ---- pricing (multiple) ----
        kk = max(1, add_per_round)
        rp = price_topk(
            X, nu, k=kk, device=device, _Xdev=Xdev, rng_seed=rng_seed + it, **ok
        )
        rm = price_topk(
            X, -nu, k=kk, device=device, _Xdev=Xdev, rng_seed=rng_seed + 1000 + it, **ok
        )
        cand = rp["dirs"] + rm["dirs"]
        rho_lb = max([v for v, _ in cand], default=0.0)
        # batch width: fixed at add_per_round by default (add_min == add_per_round).
        # Lowering add_min tapers the batch toward add_min as rho_lb -> lam, which keeps
        # the working set leaner but -- measured -- costs accuracy in hard regimes
        # (sustained diverse exploration matters even near convergence), so it is off.
        frac = min(1.0, max(0.0, (rho_lb / max(lam_it, 1e-12) - 1.0) / 3.0))
        eff_k = max(add_min, round(add_per_round * frac))

        # ---- bound estimate (periodic) ----
        if certify_every and (it % certify_every == 0 or it == max_iter - 1):
            ub, _ = estimate_absolute_pricing_bound(
                X,
                nu,
                device=device,
                n_steps=cert_steps,
            )
        else:
            ub = max(rho_lb, lam_it)
        gap = max(obj - _dual_value(yv, to_cpu(nu), lam_it, ub), 0.0)

        hist["iter"].append(it)
        hist["n_cols"].append(int(A.shape[1]))
        hist["n_active"].append(n_active)
        hist["obj"].append(obj)
        hist["rho_lb"].append(float(rho_lb))
        hist["ub"].append(float(ub))
        hist["gap"].append(float(gap))
        hist["time"].append(time.time() - t0)
        if verbose:
            print(
                f"[refinement {it:02d}] cols={A.shape[1]:4d} act={n_active:3d} "
                f"obj={obj:.6f} rho_lb={rho_lb:.4f} ub={ub:.4f} lam={lam_it:.3f} gap={gap:.3e}"
            )

        # ---- stop checks (only at the target lambda) ----
        if at_target:
            if len(hist["obj"]) > patience:
                recent = hist["obj"][-(patience + 1) :]
                rel = (recent[0] - recent[-1]) / max(1.0, abs(recent[-1]))
                if 0 <= rel < stall_rtol:
                    if verbose:
                        print(f"  -> objective plateau (rel={rel:.1e}); stop")
                    break
            if rho_lb <= lam_it * (1.0 + price_tol):
                if verbose:
                    print(f"  -> no strong violator (rho_lb={rho_lb:.4f}); stop")
                break
            if gap <= eps:
                if verbose:
                    print(f"  -> certified gap {gap:.2e} <= eps; stop")
                break

        # ---- add violating columns (dedup vs current + each other) ----
        Ucpu = to_cpu(U)
        Un = Ucpu / (np.linalg.norm(Ucpu, axis=0) + 1e-30) if Ucpu.shape[1] else Ucpu
        newU = []
        for v, u in sorted(cand, key=lambda z: -z[0]):
            if v <= lam_it * (1.0 + 1e-3):
                continue
            uu = np.asarray(u, dtype=np.float64)
            uu /= np.linalg.norm(uu) + 1e-30
            if Un.shape[1] and float(np.max(np.abs(uu @ Un))) > cos_dedup:
                continue
            if any(abs(float(uu @ w)) > cos_dedup for w in newU):
                continue
            newU.append(uu)
            if len(newU) >= 2 * eff_k:
                break
        if not newU:
            if verbose:
                print("  -> no new column; stop")
            break
        Unew = xp.asarray(np.stack(newU, axis=1), dtype=dtype)
        U = xp.concatenate([U, Unew], axis=1)
        A = xp.concatenate([A, xp.maximum(Xdev @ Unew, 0.0)], axis=1)
        beta = xp.concatenate([beta, xp.zeros(Unew.shape[1], dtype=dtype)])
        if max_cols is not None and A.shape[1] >= max_cols:
            break

    # final numerical solve at the target lambda
    if A.shape[1]:
        beta, nu, obj, S, _ = active_set_lasso(A, ydev, lam, xp)
        n_active = len(S)
    else:
        n_active = 0
    out = dict(
        obj=obj,
        nu=to_cpu(nu),
        beta=to_cpu(beta),
        U=to_cpu(U),
        n_cols=int(A.shape[1]),
        n_active=n_active,
        gap=hist["gap"][-1] if hist["gap"] else np.nan,
        history=hist,
    )
    return out
