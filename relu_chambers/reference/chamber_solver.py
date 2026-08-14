"""Finite-chamber column generation and the sample-only baseline.

The chamber reference alternates:
  (1) solve the restricted convex master over the current chambers,
  (2) PRICE: (approximately) solve  max_{||u||<=1} +/- nu^T (X u)_+,
  (3) if the best score exceeds lam, ADD that chamber and repeat; otherwise stop.

Stopping is globally conclusive only when the supplied pricing set is exhaustive.
The heuristic and rounded modes return feasible directions, not proofs that no
violating chamber remains.  SDP values are retained as bound diagnostics.

Sample-only draws N Gaussian directions, collects their unique chambers, and
solves the master ONCE on that fixed set.
"""

from __future__ import annotations

import time

import numpy as np

from ..synthetic_data import pattern_key, sample_patterns
from .convex_master import solve_master
from .exact_pricing import (
    certificate_sdp,
    price_and_certify_sdp,
    pricing_exact,
    pricing_heuristic,
    pricing_sdp_rounded,
)


def dual_value(y, nu, lam, ub):
    """Lower bound on P* from the dual: rescale nu to be dual-feasible
    (rho(theta nu) <= theta*UB <= lam) and evaluate D(theta nu)=theta nu^T y
    - 0.5 theta^2 ||nu||^2.  Returns (D, theta)."""
    if ub <= 1e-12:
        theta = 1.0
    else:
        theta = min(1.0, lam / ub)
    D = theta * float(nu @ y) - 0.5 * theta**2 * float(nu @ nu)
    return D, theta


def solve_chamber_reference(
    X,
    y,
    lam,
    max_iter=40,
    tol=1e-4,
    oracle="heuristic",
    enum_patterns=None,
    seed_samples=0,
    compute_sdp=True,
    sdp_every=1,
    compute_true=False,
    compute_round=False,
    round_kwargs=None,
    pricing_kwargs=None,
    max_chambers=None,
    rng=None,
    verbose=False,
):
    """Run finite-chamber column generation.

    oracle: 'exact' (enumerate arrangement; small d), 'heuristic' (multi-start
            gradient ascent), or 'sdp_rounded' (randomized Goemans-Williamson
            rounding of the certificate SDP; the certificate UB comes free from
            the same solve).
    enum_patterns: precomputed full pattern list for exact pricing / true gap.
    seed_samples: number of Gaussian directions used to seed the chamber set.
    compute_sdp: also compute the SDP certificate UB at each iteration.
    compute_true: also compute exact rho(nu) (requires enum_patterns) for
                  measuring the TRUE optimality gap.
    compute_round: also compute the randomized-rounding lower bound each iter
                  (diagnostic; e.g. to show the LB sandwich vs the exact oracle).
    Returns dict with the per-iteration history and the final master solution.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    _n, _d = X.shape
    pk = pricing_kwargs or {}
    rk = round_kwargs or {}

    patterns = []
    keys = set()
    if seed_samples > 0:
        for s in sample_patterns(X, seed_samples, rng):
            k = s.tobytes()
            if k not in keys:
                keys.add(k)
                patterns.append(s)

    hist = {
        "iter": [],
        "n_chambers": [],
        "n_active": [],
        "obj": [],
        "rho_lb": [],
        "rho_ub": [],
        "true_rho": [],
        "rho_round": [],
        "cert_gap": [],
        "time": [],
    }
    t0 = time.time()
    sol = None
    for it in range(max_iter):
        sol = solve_master(X, y, patterns, lam)
        nu = sol["nu"]

        # ---- pricing (find a violating chamber) ----
        sdp_ub = None
        if oracle == "exact":
            vp, _, sp = pricing_exact(X, nu, patterns=enum_patterns, rng=rng)
            vm, _, sm = pricing_exact(X, -nu, patterns=enum_patterns, rng=rng)
            rho_lb = max(vp, vm)
            true_rho = rho_lb
        elif oracle == "sdp_rounded":
            # one PSD solve -> rounded lower bounds for +/-nu AND the certificate
            res = price_and_certify_sdp(X, nu, rng=rng, **pk)
            vp, _up, sp = res["rho_plus"], res["u_plus"], res["s_plus"]
            vm, _um, sm = res["rho_minus"], res["u_minus"], res["s_minus"]
            rho_lb = max(vp, vm)
            sdp_ub = res["ub"]  # certificate is free from this solve
            true_rho = np.nan
            if compute_true and enum_patterns is not None:
                tvp, _, _ = pricing_exact(X, nu, patterns=enum_patterns)
                tvm, _, _ = pricing_exact(X, -nu, patterns=enum_patterns)
                true_rho = max(tvp, tvm)
        else:
            vp, _up, sp = pricing_heuristic(X, nu, rng=rng, **pk)
            vm, _um, sm = pricing_heuristic(X, -nu, rng=rng, **pk)
            rho_lb = max(vp, vm)
            true_rho = np.nan
            if compute_true and enum_patterns is not None:
                tvp, _, _ = pricing_exact(X, nu, patterns=enum_patterns)
                tvm, _, _ = pricing_exact(X, -nu, patterns=enum_patterns)
                true_rho = max(tvp, tvm)

        # ---- certificate ----
        do_sdp = compute_sdp and (it % max(1, sdp_every) == 0)
        if sdp_ub is not None:
            ub = sdp_ub  # already computed by the rounded oracle
        elif do_sdp:
            ub, _ = certificate_sdp(X, nu)  # UB(nu)=UB(-nu)
        else:
            ub = np.nan
        if np.isfinite(ub):
            Dval, _ = dual_value(y, nu, lam, ub)
            cert_gap = sol["obj"] - Dval
        else:
            cert_gap = np.nan

        # ---- optional rounding-oracle LB diagnostic (the recommended oracle:
        #      randomized rounding combined with a few ascent restarts) ----
        rho_round = np.nan
        if compute_round:
            rvp, _, _ = pricing_sdp_rounded(X, nu, rng=rng, **rk)
            rvm, _, _ = pricing_sdp_rounded(X, -nu, rng=rng, **rk)
            hvp, _, _ = pricing_heuristic(X, nu, rng=rng, n_restarts=6, n_iters=60)
            hvm, _, _ = pricing_heuristic(X, -nu, rng=rng, n_restarts=6, n_iters=60)
            rho_round = max(rvp, rvm, hvp, hvm)

        hist["iter"].append(it)
        hist["n_chambers"].append(len(patterns))
        hist["n_active"].append(len(sol["active_idx"]))
        hist["obj"].append(sol["obj"])
        hist["rho_lb"].append(float(rho_lb))
        hist["rho_ub"].append(float(ub))
        hist["true_rho"].append(float(true_rho))
        hist["rho_round"].append(float(rho_round))
        hist["cert_gap"].append(
            float(max(cert_gap, 0.0)) if np.isfinite(cert_gap) else np.nan
        )
        hist["time"].append(time.time() - t0)
        if verbose:
            print(
                f"[chamber {it:02d}] |A|={len(patterns):3d} active={len(sol['active_idx']):2d} "
                f"obj={sol['obj']:.5f} rho_lb={rho_lb:.4f} ub={ub:.4f} "
                f"lam={lam:.3f} certgap={cert_gap:.2e}"
            )

        # ---- stop / add ----
        added = False
        thresh = lam * (1 + tol)
        # add the violating chamber(s)
        for val, s_pat in [(vp, sp), (vm, sm)]:
            if val > thresh and s_pat is not None and np.sum(s_pat) > 0:
                k = pattern_key(s_pat)
                if k not in keys:
                    keys.add(k)
                    patterns.append(np.asarray(s_pat, dtype=np.int8))
                    added = True
        if not added:
            break
        if max_chambers is not None and len(patterns) >= max_chambers:
            # one more master solve over the (capped) chamber set, then stop
            sol = solve_master(X, y, patterns, lam)
            break

    sol["history"] = hist
    sol["n_chambers_final"] = len(patterns)
    return sol


def sample_only(X, y, lam, N, rng=None):
    """Sample-only baseline: N Gaussian directions -> unique chambers -> one solve."""
    if rng is None:
        rng = np.random.default_rng(0)
    pats = sample_patterns(X, N, rng)
    sol = solve_master(X, y, pats, lam)
    sol["n_chambers_final"] = len(pats)
    sol["N_samples"] = N
    return sol


def sample_only_until(X, y, lam, n_target, rng=None, max_draws=200000):
    """Sample-only with a CHAMBER budget: keep drawing Gaussian directions until
    `n_target` unique realizable chambers are collected, then solve once.
    Returns also the number of directions drawn to reach the budget."""
    if rng is None:
        rng = np.random.default_rng(0)
    d = X.shape[1]
    seen = {}
    drawn = 0
    while len(seen) < n_target and drawn < max_draws:
        b = min(2000, max_draws - drawn)
        G = rng.standard_normal((d, b))
        A = (X @ G >= 0).astype(np.int8)
        for c in range(b):
            s = A[:, c]
            if s.sum() == 0:
                continue
            seen[s.tobytes()] = s.copy()
            if len(seen) >= n_target:
                break
        drawn += b
    pats = list(seen.values())
    sol = solve_master(X, y, pats, lam)
    sol["n_chambers_final"] = len(pats)
    sol["n_directions_drawn"] = drawn
    return sol
