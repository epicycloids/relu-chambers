"""Reproduce the paper's standalone million-row scale artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from experiments.exploratory_benchmarks import teacher_reference
from relu_chambers.solver import solve_rcg
from relu_chambers.synthetic_data import make_planted


def _solve(n: int, device: str, max_iter: int) -> dict:
    d, k, lam = 16, 3, 0.1
    problem = make_planted(n, d, k, np.random.default_rng(0))
    X, y = problem["X"], problem["y"]
    reference = teacher_reference(X, y, problem, lam)
    started = time.perf_counter()
    solution = solve_rcg(X, y, lam, device=device, max_iter=max_iter)
    elapsed = time.perf_counter() - started
    row = {
        "device": device,
        "n": n,
        "objective": solution["obj"],
        "teacher_objective": reference,
        "relative_excess": (solution["obj"] - reference) / abs(reference),
        "wall_seconds": elapsed,
        "iterations": solution["n_iter"],
        "active_atoms": solution["n_active"],
        "price_over_lambda": solution["rho_lb"] / lam,
        "heuristic_gap": solution["gap_heur"],
        "bound_gap": solution["gap_cert"],
        "stop_reason": solution["stop_reason"],
        "timings": solution["timings"],
    }
    del solution, X, y, problem
    gc.collect()
    return row


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="one large CPU/device run")
    single.add_argument("--n", type=int, default=2_000_000)
    single.add_argument("--device", default="cpu")
    single.add_argument("--max-iter", type=int, default=60)
    single.add_argument(
        "--output", type=Path, default=Path("results/scaling_n2000000.json")
    )

    compare = subparsers.add_parser(
        "compare", help="paired device runs on the same planted instance"
    )
    compare.add_argument("--n", type=int, default=1_000_000)
    compare.add_argument(
        "--device",
        action="append",
        required=True,
        help="repeat in execution order, e.g. cuda then cpu",
    )
    compare.add_argument("--max-iter", type=int, default=60)
    compare.add_argument(
        "--output",
        type=Path,
        default=Path("results/scaling_device_comparison_n1000000.json"),
    )

    args = parser.parse_args()
    if args.mode == "single":
        row = _solve(args.n, args.device, args.max_iter)
        row.pop("device")
        _write(args.output, row)
    else:
        rows = []
        for device in args.device:
            row = _solve(args.n, device, args.max_iter)
            row.pop("relative_excess")
            row.pop("bound_gap")
            row["timings"] = {
                key: round(value, 1) for key, value in row["timings"].items()
            }
            rows.append(row)
        _write(args.output, rows)


if __name__ == "__main__":
    main()
