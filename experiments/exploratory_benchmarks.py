"""Exploratory validation and scaling experiments for the RCG solver.

python -m experiments.exploratory_benchmarks planted
python -m experiments.exploratory_benchmarks bounds
python -m experiments.exploratory_benchmarks scaling
python -m experiments.exploratory_benchmarks optimizers
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from relu_chambers.reference.lasso import active_set_lasso
from relu_chambers.synthetic_data import make_planted

RES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RES, exist_ok=True)


def teacher_reference(X, y, problem, lam):
    """Exact Lasso restricted to the true teacher atoms (upper bound on F*)."""
    At = np.maximum(X @ problem["U_star"], 0.0)
    _, _, obj, _, _ = active_set_lasso(At, y, lam, np)
    return obj


# --------------------------------------------------------------------------- #
def run_planted_validation(device="cpu"):
    from relu_chambers.reference.refinement_solver import solve_refinement_baseline
    from relu_chambers.solver import solve_rcg

    print("== planted validation: RCG, refinement baseline, teacher reference ==")
    print(
        f"{'n':>7} {'d':>3} | {'teacher':>9} | {'RCG obj':>9} {'RCG s':>6} "
        f"{'act':>3} {'rho/lam':>7} | {'ref obj':>9} {'ref s':>7}"
    )
    rows = []
    for n, d, k, lam, seed in [
        (500, 12, 3, 0.1, 0),
        (2000, 16, 3, 0.1, 0),
        (2000, 16, 3, 0.02, 1),
        (5000, 32, 5, 0.1, 2),
        (2000, 16, 8, 0.05, 3),
    ]:
        rng = np.random.default_rng(seed)
        problem = make_planted(n, d, k, rng)
        X, y = problem["X"], problem["y"]
        teacher = teacher_reference(X, y, problem, lam)
        t0 = time.time()
        rcg = solve_rcg(X, y, lam, device=device, max_iter=60)
        rcg_seconds = time.time() - t0
        # The refinement baseline is run only where its n*d cost is affordable.
        if n * d <= 40000:
            t0 = time.time()
            reference = solve_refinement_baseline(
                X,
                y,
                lam,
                device=device,
                max_iter=40,
            )
            reference_objective = reference["obj"]
            reference_seconds = time.time() - t0
        else:
            reference_objective = reference_seconds = float("nan")
        print(
            f"{n:7d} {d:3d} | {teacher:9.5f} | {rcg['obj']:9.5f} "
            f"{rcg_seconds:6.1f} {rcg['n_active']:3d} "
            f"{rcg['rho_lb'] / lam:7.3f} | {reference_objective:9.5f} "
            f"{reference_seconds:7.1f}",
            flush=True,
        )
        rows.append(
            {
                "n": n,
                "d": d,
                "k": k,
                "lambda": lam,
                "teacher_objective": teacher,
                "rcg_objective": rcg["obj"],
                "rcg_wall_seconds": rcg_seconds,
                "rcg_active_atoms": rcg["n_active"],
                "rcg_price_over_lambda": rcg["rho_lb"] / lam,
                "rcg_heuristic_gap": rcg["gap_heur"],
                "rcg_bound_gap": rcg["gap_cert"],
                "refinement_baseline_objective": (
                    None if np.isnan(reference_objective) else reference_objective
                ),
                "refinement_baseline_wall_seconds": (
                    None if np.isnan(reference_seconds) else reference_seconds
                ),
            }
        )
    with open(os.path.join(RES, "planted_validation.json"), "w") as handle:
        json.dump(rows, handle, indent=2)


# --------------------------------------------------------------------------- #
def run_pricing_bound_comparison():
    """Compare the positive-part and absolute-residual pricing bounds."""
    from relu_chambers.pricing_bounds import estimate_pricing_bounds
    from relu_chambers.reference.absolute_pricing_bound import (
        estimate_absolute_pricing_bound,
    )
    from relu_chambers.reference.exact_pricing import enumerate_patterns, pricing_exact
    from relu_chambers.solver import make_engine

    print("== pricing bounds: positive-part vs absolute-residual ==")
    rng = np.random.default_rng(3)
    rows, bad = [], 0
    for trial in range(10):
        ns, ds = 60, 3
        Xs = rng.standard_normal((ns, ds))
        nus = rng.standard_normal(ns)
        pats = enumerate_patterns(Xs, n_dense=100000, rng=rng)
        rho, _, _ = pricing_exact(Xs, nus, patterns=pats)
        eng = make_engine(Xs, nus * 0.0)
        bounds = estimate_pricing_bounds(
            eng,
            nus,
            n_steps=40,
            forms=("joint", "split"),
            sketch_m=0,
        )
        absolute_bound, _ = estimate_absolute_pricing_bound(
            Xs,
            nus,
            device="cpu",
            n_steps=40,
        )
        positive_bound = bounds["ub_plus"]
        valid = positive_bound >= rho - 1e-9
        bad += not valid
        rows.append(
            {
                "exact_price": rho,
                "positive_part_bound": positive_bound,
                "absolute_residual_bound": absolute_bound,
                "valid": bool(valid),
            }
        )
        print(
            f"  rho={rho:8.4f}  positive={positive_bound:8.4f} "
            f"({positive_bound / rho:5.2f}x)  absolute={absolute_bound:8.4f} "
            f"({absolute_bound / rho:5.2f}x)  valid={valid}"
        )
    # mid-run residual tightness at larger n
    for n, d in [(2000, 16), (20000, 16)]:
        P = make_planted(n, d, 3, rng)
        X, y = P["X"], P["y"]
        nu = y - 0.6 * y + 0.1 * rng.standard_normal(n)
        eng = make_engine(X, y)
        bounds = estimate_pricing_bounds(
            eng,
            nu,
            n_steps=30,
            forms=("joint",),
            sketch_m=0,
        )
        positive_bound = max(bounds["ub_plus"], bounds["ub_minus"])
        absolute_bound, _ = estimate_absolute_pricing_bound(
            X,
            nu,
            device="cpu",
            n_steps=30,
        )
        ratio = positive_bound / absolute_bound
        print(f"  n={n} d={d} mid-run residual: bound ratio={ratio:.3f}")
        rows.append(
            {
                "n": n,
                "d": d,
                "kind": "midrun",
                "positive_part_bound": positive_bound,
                "absolute_residual_bound": absolute_bound,
            }
        )
    print(f"  validity violations: {bad}")
    with open(os.path.join(RES, "pricing_bound_comparison.json"), "w") as handle:
        json.dump(rows, handle, indent=2)


# --------------------------------------------------------------------------- #
def run_scaling_sweep(device="cpu", ns=(20000, 100000, 500000, 1000000)):
    from relu_chambers.solver import solve_rcg

    print(f"== scaling sweep (device={device}) ==")
    rows = []
    for n in ns:
        d, k, lam = 16, 3, 0.1
        rng = np.random.default_rng(0)
        problem = make_planted(n, d, k, rng)
        X, y = problem["X"], problem["y"]
        teacher = teacher_reference(X, y, problem, lam)
        t0 = time.time()
        s = solve_rcg(X, y, lam, device=device, max_iter=60)
        t = time.time() - t0
        rel = (s["obj"] - teacher) / abs(teacher)
        print(
            f"  n={n:8d}: obj={s['obj']:.6f} (teacher {teacher:.6f}, rel {rel:+.2e}) "
            f"act={s['n_active']} rho/lam={s['rho_lb'] / lam:.3f} "
            f"iters={s['n_iter']} t={t:.1f}s "
            f"[{', '.join(f'{k}:{v:.0f}s' for k, v in s['timings'].items())}]"
        )
        rows.append(
            {
                "n": n,
                "objective": s["obj"],
                "teacher_objective": teacher,
                "relative_excess": rel,
                "wall_seconds": t,
                "iterations": s["n_iter"],
                "active_atoms": s["n_active"],
                "price_over_lambda": s["rho_lb"] / lam,
                "heuristic_gap": s["gap_heur"],
                "bound_gap": s["gap_cert"],
                "timings": s["timings"],
            }
        )
        with open(os.path.join(RES, "scaling_sweep.json"), "w") as handle:
            json.dump(rows, handle, indent=2)


# --------------------------------------------------------------------------- #
def run_tuned_baseline(
    X,
    y,
    lam,
    opt,
    m,
    lrs,
    seeds,
    epochs,
    device="cpu",
    batch=None,
    X_te=None,
    y_te=None,
    time_budget=None,
    keep_best_net=False,
    wds=None,
):
    """Grid over lr (x wd, for AdamW) x seeds; results sorted by objective.
    With keep_best_net the best run's (A, b) are attached to runs[0]."""
    from experiments.baselines import train_net

    runs = []
    best_net = None
    best_obj = np.inf
    wlist = list(wds) if wds is not None else [None]
    for lr in lrs:
        for wd in wlist:
            for s in seeds:
                r = train_net(
                    X,
                    y,
                    m=m,
                    beta=lam,
                    opt=opt,
                    lr=lr,
                    epochs=epochs,
                    batch=batch,
                    device=device,
                    rng_seed=s,
                    wd=wd,
                    X_te=X_te,
                    y_te=y_te,
                    time_budget=time_budget,
                )
                if np.isfinite(r["obj"]):
                    rec = dict(
                        lr=lr, seed=s, obj=r["obj"], test_mse=r["test_mse"], t=r["time"]
                    )
                    if wd is not None:
                        rec["wd"] = wd
                    runs.append(rec)
                    if keep_best_net and r["obj"] < best_obj:
                        best_obj = r["obj"]
                        best_net = (r["A"], r["b"])
    runs.sort(key=lambda z: z["obj"])
    if keep_best_net and runs and best_net is not None:
        runs[0]["net"] = best_net
    return runs


def run_optimizer_comparison(device="cpu", n=100000, d=16, k=3, lam=0.1):
    from relu_chambers.solver import solve_rcg

    print(f"== baselines at n={n}, d={d}, k={k}, lam={lam} ==")
    rng = np.random.default_rng(0)
    P = make_planted(n, d, k, rng, noise=0.0)
    X, y = P["X"], P["y"]
    Pte = make_planted(4000, d, k, np.random.default_rng(99))
    X_te = Pte["X"]
    y_te = np.maximum(X_te @ P["U_star"], 0) @ P["alpha_star"]
    teacher = teacher_reference(X, y, P, lam)
    out = {"setup": dict(n=n, d=d, k=k, lam=lam, teacher_objective=teacher)}
    t0 = time.time()
    s = solve_rcg(X, y, lam, device=device, max_iter=60)
    tc = time.time() - t0
    pred = np.maximum(X_te @ s["U"], 0) @ s["beta"]
    mse = float(np.mean((pred - y_te) ** 2))
    out["rcg"] = dict(
        obj=s["obj"],
        t=tc,
        test_mse=mse,
        active=s["n_active"],
        gapH=s["gap_heur"],
        gapC=s["gap_cert"],
    )
    print(f"  RCG: obj={s['obj']:.6f} t={tc:.0f}s test_mse={mse:.2e}")
    budget = max(tc, 10.0)
    for opt in ("adam", "adamw", "sgd"):
        lrs = (3e-4, 1e-3, 3e-3) if opt != "sgd" else (1e-7, 1e-6, 1e-5)
        runs = run_tuned_baseline(
            X,
            y,
            lam,
            opt,
            m=64,
            lrs=lrs,
            seeds=(0, 1),
            epochs=100000,
            device=device,
            X_te=X_te,
            y_te=y_te,
            time_budget=budget,
        )
        if runs:
            best, med = runs[0], runs[len(runs) // 2]
            out[opt] = dict(best=best, median=med, n_runs=len(runs), budget=budget)
            print(
                f"  {opt:6s}: best obj={best['obj']:.6f} (lr={best['lr']}) "
                f"med={med['obj']:.6f}  test_mse(best)={best['test_mse']:.2e}"
            )
    with open(os.path.join(RES, "planted_optimizer_comparison.json"), "w") as handle:
        json.dump(out, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study",
        nargs="?",
        default="all",
        choices=("planted", "bounds", "scaling", "optimizers", "all"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t0 = time.time()
    if args.study in ("planted", "all"):
        run_planted_validation(device=args.device)
    if args.study in ("bounds", "all"):
        run_pricing_bound_comparison()
    if args.study in ("scaling", "all"):
        run_scaling_sweep(device=args.device)
    if args.study in ("optimizers", "all"):
        run_optimizer_comparison(device=args.device)
    print(f"\nexploratory benchmarks [{args.study}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
