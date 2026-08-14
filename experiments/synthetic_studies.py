"""Small synthetic studies supporting the paper's mathematical experiments.

Usage:
    python -m experiments.synthetic_studies exact-enumeration
    python -m experiments.synthetic_studies dimension-sampling
    python -m experiments.synthetic_studies pricing-methods
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from relu_chambers.reference.chamber_solver import (
    sample_only_until,
    solve_chamber_reference,
)
from relu_chambers.reference.exact_pricing import (
    enumerate_patterns,
    pricing_heuristic,
    pricing_sdp_rounded,
)
from relu_chambers.synthetic_data import estimate_solid_angle, make_planted, margin_of

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESDIR = os.path.join(HERE, "results")
os.makedirs(RESDIR, exist_ok=True)


# --------------------------------------------------------------------------- #
def run_exact_enumeration(seed=3):
    """Low-d: validate the SDP certificate (Thm 6) and the anytime gap."""
    print("\n===== exact enumerable instance =====")
    rng = np.random.default_rng(seed)
    n, d, k, lam = 30, 3, 2, 0.08
    P = make_planted(n, d, k, rng, noise=0.0)
    X, y = P["X"], P["y"]
    pats = enumerate_patterns(X, n_dense=120000, rng=rng)
    print(f"n={n} d={d} k={k} lam={lam}  |chambers|={len(pats)}")

    sol = solve_chamber_reference(
        X,
        y,
        lam,
        max_iter=25,
        tol=1e-5,
        oracle="exact",
        enum_patterns=pats,
        compute_sdp=True,
        sdp_every=1,
        compute_round=True,
        round_kwargs=dict(n_rounds=64, n_polish=6),
        rng=rng,
        verbose=True,
    )
    h = sol["history"]
    # global optimum P* = converged objective (exact oracle => global optimum)
    Pstar = sol["obj"]
    # validity check: SDP UB >= true rho at every iter
    viol = [ub - tr for ub, tr in zip(h["rho_ub"], h["true_rho"])]
    min_margin = min(viol)
    print(
        f"P* = {Pstar:.6f};  min(UB - true_rho) over iters = {min_margin:.4e} (>=0 => valid)"
    )

    # how close is the rounded LB to the exact rho? (sandwich from below)
    rr = [r for r, t in zip(h["rho_round"], h["true_rho"]) if t > 1e-9]
    tt = [t for t in h["true_rho"] if t > 1e-9]
    if tt:
        ratio = float(np.mean([r / t for r, t in zip(rr, tt)]))
        print(f"  rounded-LB / exact-rho (mean over iters) = {ratio:.4f}")

    out = dict(
        n=n,
        d=d,
        k=k,
        lam=lam,
        n_chambers_total=len(pats),
        Pstar=Pstar,
        history=h,
        ub_minus_rho_min=min_margin,
    )
    path = os.path.join(RESDIR, "exact_enumerable_instance.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(out), handle, indent=2)
    return out


# --------------------------------------------------------------------------- #
def run_dimension_sampling(reps=3, seed=40):
    """Curse of dimensionality: solid-angle decay + fixed-chamber-budget comparison."""
    print("\n===== dimension and chamber sampling =====")
    n, k, lam = 50, 2, 0.05
    dims = [2, 3, 4, 6, 8, 10, 12, 16]
    budget = 40  # both methods get this many chambers
    rows = []
    for d in dims:
        sa_list, marg_list = [], []
        adaptive_obj, samp_obj = [], []
        samp_dirs = []
        for r in range(reps):
            rng = np.random.default_rng(seed + 13 * d + r)
            P = make_planted(n, d, k, rng, noise=0.0)
            X, y = P["X"], P["y"]
            # solid angle of teacher chambers
            sas = [estimate_solid_angle(X, s, 200000, rng) for s in P["S_star"]]
            sa_list.append(min(sas))
            marg_list.append(min(margin_of(X, P["U_star"][:, j]) for j in range(k)))
            # Residual-guided chamber generation up to the shared budget.
            solC = solve_chamber_reference(
                X,
                y,
                lam,
                max_iter=80,
                tol=1e-3,
                oracle="heuristic",
                compute_sdp=False,
                max_chambers=budget,
                pricing_kwargs=dict(n_restarts=10, n_iters=70),
                rng=rng,
            )
            adaptive_obj.append(solC["obj"])
            # sample-only until `budget` chambers (fair)
            solS = sample_only_until(
                X, y, lam, budget, rng=np.random.default_rng(seed + 555 + d + r)
            )
            samp_obj.append(solS["obj"])
            samp_dirs.append(solS["n_directions_drawn"])
        rows.append(
            {
                "d": d,
                "solid_angle": float(np.mean(sa_list)),
                "solid_angle_std": float(np.std(sa_list)),
                "margin": float(np.mean(marg_list)),
                "adaptive_objective": float(np.mean(adaptive_obj)),
                "adaptive_objective_std": float(np.std(adaptive_obj)),
                "sampling_objective": float(np.mean(samp_obj)),
                "sampling_objective_std": float(np.std(samp_obj)),
                "sampling_directions": float(np.mean(samp_dirs)),
            }
        )
        print(
            f"d={d:2d}  solid_angle(min teacher)={rows[-1]['solid_angle']:.2e}  "
            f"adaptive_obj={rows[-1]['adaptive_objective']:.4f}  "
            f"sampling_obj={rows[-1]['sampling_objective']:.4f}  "
            f"dirs_to_{budget}_chambers={rows[-1]['sampling_directions']:.0f}"
        )

    payload = dict(n=n, k=k, lam=lam, budget=budget, rows=rows)
    path = os.path.join(RESDIR, "dimension_sampling.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    return rows


# --------------------------------------------------------------------------- #
def _maxcut_exact(L):
    """Exact max_{z in {+-1}^n} z^T L z = 4*MaxCut, by vectorized enumeration
    (fix z_0=+1 by symmetry). For small n only."""
    n = L.shape[0]
    m = 1 << (n - 1)
    bits = ((np.arange(m)[:, None] >> np.arange(n - 1)[None, :]) & 1).astype(float)
    Z = np.empty((m, n))
    Z[:, 0] = 1.0
    Z[:, 1:] = 1.0 - 2.0 * bits  # {0,1} -> {+1,-1}
    q = np.einsum("mi,ij,mj->m", Z, L, Z)  # z^T L z for every z
    return float(q.max())


def _maxcut_instance(A):
    """Thm 5 reduction: X=[R;-R], nu=1 with R=L^{1/2}; rho(1)=2 sqrt(MaxCut)."""
    L = np.diag(A.sum(1)) - A
    w, Q = np.linalg.eigh(L)
    R = (Q * np.sqrt(np.clip(w, 0, None))) @ Q.T
    X = np.vstack([R, -R])
    nu = np.ones(X.shape[0])
    return X, nu, L


def _rand_graph(nv, p, rng):
    A = (rng.random((nv, nv)) < p).astype(float)
    A = np.triu(A, 1)
    return A + A.T


def run_pricing_method_comparison(seed=2024):
    """Pricing-oracle quality on the paper's own hard instances (Thm 5).

    On the Max-Cut reduction, gradient-ascent pricing maximizes an L1 norm over
    the ball and stalls in local optima, whereas randomized SDP rounding
    (Goemans-Williamson) recovers the global optimum. (a) approximation ratio
    vs problem size with exact ground truth; (b) ratio vs compute budget,
    showing the heuristic's deficit is structural, not a tuning artifact.
    """
    print("\n===== pricing method comparison (hard instances) =====")
    rng = np.random.default_rng(seed)

    # ---- (a) approximation ratio vs graph size ----
    sizes = [8, 10, 12, 14, 16]
    graphs_per = 8
    a_heur_mean, a_heur_std, a_rnd_mean, a_rnd_std = [], [], [], []
    for nv in sizes:
        rh, rr = [], []
        for g in range(graphs_per):
            p = float(rng.choice([0.3, 0.5, 0.7]))
            A = _rand_graph(nv, p, rng)
            X, nu, L = _maxcut_instance(A)
            exact = 2.0 * np.sqrt(max(_maxcut_exact(L), 0.0) / 4.0)  # 2*sqrt(MaxCut)
            vh, _, _ = pricing_heuristic(
                X, nu, n_restarts=20, n_iters=150, rng=np.random.default_rng(seed + g)
            )
            vr, _, _ = pricing_sdp_rounded(
                X,
                nu,
                n_rounds=96,
                n_polish=6,
                extra_seeds=True,
                rng=np.random.default_rng(seed + g),
            )
            rh.append(vh / exact)
            rr.append(vr / exact)
        a_heur_mean.append(np.mean(rh))
        a_heur_std.append(np.std(rh))
        a_rnd_mean.append(np.mean(rr))
        a_rnd_std.append(np.std(rr))
        print(
            f"  nv={nv:2d}: heuristic={np.mean(rh):.4f}+-{np.std(rh):.4f}  "
            f"rounded={np.mean(rr):.4f}+-{np.std(rr):.4f}"
        )

    # ---- (b) approximation ratio vs compute budget (fixed instance set) ----
    insts = []
    for _ in range(10):
        nv = int(rng.integers(13, 16))
        p = float(rng.choice([0.3, 0.5]))
        A = _rand_graph(nv, p, rng)
        X, nu, L = _maxcut_instance(A)
        insts.append((X, nu, 2.0 * np.sqrt(max(_maxcut_exact(L), 0.0) / 4.0)))
    heur_budgets = [5, 10, 25, 50, 100, 200]
    rnd_budgets = [8, 16, 32, 64, 128]
    heur_b_ratio, heur_b_time = [], []
    for nr in heur_budgets:
        rs, ts = [], []
        for X, nu, ex in insts:
            t0 = time.time()
            v, _, _ = pricing_heuristic(
                X, nu, n_restarts=nr, n_iters=150, rng=np.random.default_rng(seed)
            )
            ts.append(time.time() - t0)
            rs.append(v / ex)
        heur_b_ratio.append(np.mean(rs))
        heur_b_time.append(np.mean(ts))
    rnd_b_ratio, rnd_b_time = [], []
    for nr in rnd_budgets:
        rs, ts = [], []
        for X, nu, ex in insts:
            t0 = time.time()
            v, _, _ = pricing_sdp_rounded(
                X,
                nu,
                n_rounds=nr,
                n_polish=4,
                extra_seeds=True,
                rng=np.random.default_rng(seed),
            )
            ts.append(time.time() - t0)
            rs.append(v / ex)
        rnd_b_ratio.append(np.mean(rs))
        rnd_b_time.append(np.mean(ts))
    print(
        f"  heuristic ratio vs restarts {heur_budgets}: "
        f"{[round(float(x), 4) for x in heur_b_ratio]}"
    )
    print(
        f"  rounded   ratio vs rounds   {rnd_budgets}: "
        f"{[round(float(x), 4) for x in rnd_b_ratio]}"
    )

    out = dict(
        sizes=sizes,
        graphs_per=graphs_per,
        a_heur_mean=a_heur_mean,
        a_heur_std=a_heur_std,
        a_rnd_mean=a_rnd_mean,
        a_rnd_std=a_rnd_std,
        heur_budgets=heur_budgets,
        heur_b_ratio=heur_b_ratio,
        heur_b_time=heur_b_time,
        rnd_budgets=rnd_budgets,
        rnd_b_ratio=rnd_b_ratio,
        rnd_b_time=rnd_b_time,
    )
    path = os.path.join(RESDIR, "pricing_method_comparison.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(out), handle, indent=2)
    return out


# --------------------------------------------------------------------------- #
def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        default="all",
        choices=(
            "exact-enumeration",
            "dimension-sampling",
            "pricing-methods",
            "all",
        ),
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.study in ("exact-enumeration", "all"):
        run_exact_enumeration()
    if args.study in ("dimension-sampling", "all"):
        run_dimension_sampling()
    if args.study in ("pricing-methods", "all"):
        run_pricing_method_comparison()
    print(f"\nDONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
