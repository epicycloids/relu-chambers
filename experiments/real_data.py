"""Real-data comparison of RCG with tuned SGD, Adam, and AdamW baselines.

Datasets (cached under data_cache/):
    california  20,640 x 8     regression (median house value)
    covtype     581,012 x 54   binary LS-classification (lodgepole pine vs rest)
    msd         515,345 x 90   regression (song year)

Protocol:
  * train/test split, features standardized by train stats, bias column added,
    target standardized by train stats (so test MSE is in target-variance units).
  * lambda = frac * lambda_max with lambda_max = max(rho(y), rho(-y)) estimated
    by the pricing oracle on the training residual y (the smallest lambda whose
    solution is empty); frac defaults to 0.01.
  * every method minimizes F = 1/2||relu(XA)b - y||^2 + lam/2(||A||_F^2+||b||^2)
    == the convex program objective at the Pilanci scaling; baselines get the
    same wall-clock budget used by RCG (and their own tuned lr grid x seeds).

    python -m experiments.real_data california
    python -m experiments.real_data covtype --frac 0.01 --budget-mult 1.0
    python -m experiments.real_data msd
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

RES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache"
)


def load_dataset(name, cache_dir=None):
    cache = CACHE if cache_dir is None else os.fspath(cache_dir)
    if name == "california":
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing(data_home=cache)
        return np.asarray(d.data, float), np.asarray(d.target, float)
    if name == "covtype":
        from sklearn.datasets import fetch_covtype

        d = fetch_covtype(data_home=cache)
        y = (d.target == 2).astype(float) * 2.0 - 1.0  # +-1 labels
        return np.asarray(d.data, float), y
    if name == "msd":
        f = os.path.join(cache, "msd.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(
                "YearPredictionMSD is not cached; run "
                "`python -m experiments.prepare_data msd` first"
            )
        z = np.load(f)
        return np.asarray(z["X"], np.float64), np.asarray(z["y"], np.float64)
    raise ValueError(name)


def prep(X, y, test_frac=0.25, seed=0, max_n=None):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    if max_n:
        idx = idx[: int(max_n / (1 - test_frac))]
    ntr = round(len(idx) * (1 - test_frac))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ymu, ysd = ytr.mean(), ytr.std() + 1e-12
    ytr = (ytr - ymu) / ysd
    yte = (yte - ymu) / ysd
    Xtr = np.c_[Xtr, np.ones(len(Xtr))]
    Xte = np.c_[Xte, np.ones(len(Xte))]
    return np.ascontiguousarray(Xtr), ytr, np.ascontiguousarray(Xte), yte


def lambda_max(X, y, device="cpu"):
    from relu_chambers.adaptive_pricing import search_signed_atoms
    from relu_chambers.solver import make_engine

    eng = make_engine(X, y, device=device)
    nu_sh = [y_ * 1.0 for y_ in eng.ys]
    pr = search_signed_atoms(
        eng,
        nu_sh,
        np.asarray(y, float),
        0.0,
        k=1,
        sketch_m=min(32768, len(y) // 2),
        rng=np.random.default_rng(1),
    )
    return pr["best"]


def run(
    name,
    frac=0.01,
    budget_mult=1.0,
    device="cpu",
    max_n=None,
    m_net=128,
    max_iter=30,
    out_suffix="",
    cache_dir=None,
):
    from experiments.exploratory_benchmarks import run_tuned_baseline
    from relu_chambers.solver import solve_rcg

    X, y = load_dataset(name, cache_dir=cache_dir)
    Xtr, ytr, Xte, yte = prep(X, y, max_n=max_n)
    n, d = Xtr.shape
    lmax = lambda_max(Xtr, ytr, device)
    lam = frac * lmax
    print(f"[{name}] n={n} d={d}  lambda_max~{lmax:.2f} -> lam={lam:.4f}")
    out = {"setup": dict(name=name, n=n, d=d, lam=lam, frac=frac, m_net=m_net)}

    t0 = time.time()
    s = solve_rcg(Xtr, ytr, lam, device=device, max_iter=max_iter)
    tc = time.time() - t0
    pred = np.maximum(Xte @ s["U"], 0) @ s["beta"]
    mse = float(np.mean((pred - yte) ** 2))
    extra = {}
    if name == "covtype":
        # y was standardized by train stats: the two label values are distinct
        # constants; classify by the midpoint between them
        thr = float(np.unique(yte).mean())
        extra["test_acc"] = float(np.mean((pred > thr) == (yte > thr)))
    out["rcg"] = dict(
        obj=s["obj"],
        t=tc,
        test_mse=mse,
        active=s["n_active"],
        rho_rel=s["rho_lb"] / lam,
        gapH=s["gap_heur"],
        gapC=s["gap_cert"],
        iters=s["n_iter"],
        stop=s["stop_reason"],
        **extra,
    )
    print(
        f"  RCG: obj={s['obj']:.4f} act={s['n_active']} rho/lam={s['rho_lb'] / lam:.3f} "
        f"gapH={s['gap_heur']:.2e} t={tc:.0f}s test_mse={mse:.4f} {extra}"
    )

    budget = max(tc * budget_mult, 20.0)
    batch = min(n, 16384) if n > 100000 else None
    for opt in ("adam", "adamw", "sgd"):
        if opt == "adam":
            lrs, wds = (1e-4, 1e-3, 1e-2), None
        elif opt == "adamw":
            # AdamW's decoupled decay rate must be tuned independently of lambda
            # (wd=lambda collapses the net to the zero model) and its optimum
            # varies by dataset, so sweep a broad grid spanning the non-collapsing
            # regime.  Each run gets Adam's per-run budget (no dilution), so the
            # extra hyperparameter costs extra TOTAL search rather than
            # time-starved runs; all runs scored on the same F.
            lrs, wds = (
                (1e-3,),
                (0.05 * lam, 0.018 * lam, 0.006 * lam, 0.002 * lam, 0.0007 * lam),
            )
        else:
            lrs, wds = (10 ** -np.arange(4, 9, 1.0)).tolist(), None
        per_run = budget / 9.0 if opt == "adamw" else budget / (3 * len(lrs))
        runs = run_tuned_baseline(
            Xtr,
            ytr,
            lam,
            opt,
            m=m_net,
            lrs=lrs,
            wds=wds,
            seeds=(0, 1, 2),
            epochs=10**9,
            device=device,
            batch=batch,
            X_te=Xte,
            y_te=yte,
            time_budget=per_run,
            keep_best_net=(name == "covtype"),
        )
        if not runs:
            out[opt] = None
            continue
        best, med = runs[0], runs[len(runs) // 2]
        e2 = {}
        if name == "covtype" and "net" in best:
            An, bn = best.pop("net")
            pb = np.maximum(Xte @ An, 0) @ bn
            thr = float(np.unique(yte).mean())
            e2["test_acc"] = float(np.mean((pb > thr) == (yte > thr)))
        out[opt] = dict(
            best=dict(best, **e2), median=med, n_runs=len(runs), per_run_budget=per_run
        )
        wd_s = f" wd={best['wd']:.3g}" if "wd" in best else ""
        print(
            f"  {opt:6s}: best obj={best['obj']:.4f} (lr={best['lr']:.0e}{wd_s}, "
            f"mse={best['test_mse']:.4f}) {e2}  median obj={med['obj']:.4f}"
        )
    fn = os.path.join(RES, f"real_data_{name}{out_suffix}.json")
    with open(fn, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(f"  -> {fn}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["california", "covtype", "msd"])
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--budget-mult", type=float, default=1.0)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    run(
        a.dataset,
        frac=a.frac,
        budget_mult=a.budget_mult,
        device=a.device,
        max_n=a.max_n,
        max_iter=a.max_iter,
        out_suffix=a.suffix,
        cache_dir=a.cache_dir,
    )
