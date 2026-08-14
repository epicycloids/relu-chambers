"""Frozen-state study of exact predicted-decrease candidate ordering.

This pre-registered component benchmark deliberately does not make an
end-to-end performance claim.  Each deterministic state is a
finite-atom Lasso master fitted to planted ReLU data.  Its candidate pool is
then scored on the exact full data in two ways:

* raw KKT score ``abs(a.T @ r)``;
* exact one-coordinate decrease
  ``((abs(a.T @ r) - lambda)_+)**2 / (2 * ||a||**2)``.

For every candidate the second expression is checked against an explicit
soft-threshold coefficient and a direct objective evaluation.  The raw and
decrease winners are separately appended to the same finite master and fully
refitted.  Correlation-only and correlation-plus-norm rescoring receive one
untimed warm-up and a balanced repeated timing schedule.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from relu_chambers.adaptive_pricing import _predicted_decrease
from relu_chambers.data_engine import ShardedData
from relu_chambers.restricted_master import AtomLasso
from relu_chambers.synthetic_data import make_planted

SCHEMA = "relu_chambers.candidate_order_study"
VERSION = 1
METHODS = ("correlation_only", "correlation_and_sqnorm")


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    d: int
    planted_atoms: int
    master_atoms: int
    candidates: int


SHAPES = (
    Shape("small", 1024, 8, 4, 12, 32),
    Shape("medium", 4096, 16, 6, 24, 64),
)
REGIMES = ("standardized", "anisotropic")

# This operationalizes the word "negligible" for the component benchmark.
# An end-to-end claim still requires a matched improvement of at least 20%.
COMPONENT_RULE = {
    "minimum_state_reversal_fraction": 0.10,
    "minimum_median_full_corrected_gain_ratio_on_reversals": 1.0,
    "minimum_median_full_corrected_gain_per_rescore_second_ratio": 1.0,
    "formula_must_pass_for_every_candidate": True,
    "end_to_end_gain_required_for_adoption": 0.20,
}


def _run_text(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if platform.system() == "Darwin" else value * 1024)


def _memory_total_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _numpy_config() -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    raw = stream.getvalue()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return "unavailable (NumPy configuration output was not JSON)"
    for dependency in config.get("Build Dependencies", {}).values():
        if isinstance(dependency, dict):
            for key in ("include directory", "lib directory", "pc file directory"):
                dependency.pop(key, None)
    python_info = config.get("Python Information")
    if isinstance(python_info, dict):
        python_info.pop("path", None)
    return json.dumps(config, indent=2)


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "raw": [float(v) for v in values],
        "min": float(np.min(array)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(array)),
        "iqr": float(q3 - q1),
    }


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-300)


def _state_seed(shape_index: int, regime_index: int, state_index: int) -> int:
    return 21_000 + 1_000 * shape_index + 100 * regime_index + state_index


def _make_state(shape: Shape, regime: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    planted = make_planted(shape.n, shape.d, shape.planted_atoms, rng)

    # Both regimes start with exactly column-standardized Gaussian rows.  The
    # anisotropic regime then applies declared, deterministic feature scales;
    # this is intentional stress, not an accidental preprocessing mismatch.
    z = np.asarray(planted["X"], dtype=np.float64)
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    if regime == "standardized":
        scales = np.ones(shape.d, dtype=np.float64)
    elif regime == "anisotropic":
        scales = np.geomspace(0.1, 10.0, shape.d, dtype=np.float64)
    else:  # pragma: no cover - caller owns the finite regime set
        raise ValueError(regime)
    X = np.ascontiguousarray(z * scales)

    teacher = _unit_columns(planted["U_star"])
    y_clean = np.maximum(X @ teacher, 0.0) @ planted["alpha_star"]
    noise_scale = 0.02 * float(np.std(y_clean))
    y = np.asarray(
        y_clean + noise_scale * rng.standard_normal(shape.n), dtype=np.float64
    )

    master = _unit_columns(rng.standard_normal((shape.d, shape.master_atoms)))
    axes = np.concatenate([np.eye(shape.d), -np.eye(shape.d)], axis=1)
    random_count = shape.candidates - shape.planted_atoms - 2 * shape.d
    if random_count < 0:
        raise AssertionError("candidate pool is too small for declared sources")
    random_candidates = _unit_columns(rng.standard_normal((shape.d, random_count)))
    candidates = np.concatenate([teacher, axes, random_candidates], axis=1)
    labels = (
        [f"teacher:{j}" for j in range(shape.planted_atoms)]
        + [f"axis:+{j}" for j in range(shape.d)]
        + [f"axis:-{j}" for j in range(shape.d)]
        + [f"random:{j}" for j in range(random_count)]
    )
    permutation = rng.permutation(shape.candidates)
    candidates = _unit_columns(candidates[:, permutation])
    labels = [labels[int(j)] for j in permutation]

    # Fixed before fitting: a fraction of the largest zero-model correlation
    # over the exact finite union.  The same 0.05 is used in every regime.
    union_features = np.maximum(X @ np.concatenate([master, candidates], axis=1), 0.0)
    lambda_fraction = 0.05
    lam = lambda_fraction * float(np.max(np.abs(union_features.T @ y)))
    return {
        "X": X,
        "y": y,
        "y_clean": y_clean,
        "master": master,
        "candidates": candidates,
        "candidate_labels": labels,
        "lambda": lam,
        "lambda_fraction": lambda_fraction,
        "noise_scale": noise_scale,
        "feature_scales": scales,
    }


def _solve_master(
    engine: ShardedData,
    directions: np.ndarray,
    lam: float,
    atoms: list[np.ndarray] | None = None,
) -> tuple[AtomLasso, np.ndarray, list[np.ndarray], float, list[int]]:
    master = AtomLasso(engine, ridge=1e-12)
    master.set_atoms(directions, A_sh=atoms)
    beta, nu_sh, objective, active = master.solve(
        lam,
        tol=1e-10,
        batch_add=8,
    )
    return master, beta, nu_sh, float(objective), active


def _direct_objective(
    engine: ShardedData, master: AtomLasso, beta: np.ndarray, lam: float
) -> float:
    residual = engine.gather_vec(engine.residual(master.A, beta))
    return 0.5 * float(residual @ residual) + lam * float(np.abs(beta).sum())


def _kkt_error(
    engine: ShardedData, master: AtomLasso, beta: np.ndarray, lam: float
) -> dict[str, float]:
    residual = engine.residual(master.A, beta)
    correlation = engine.screen(master.A, residual)
    active = np.flatnonzero(np.abs(beta) > 1e-10)
    inactive = np.flatnonzero(np.abs(beta) <= 1e-10)
    active_error = (
        float(np.max(np.abs(correlation[active] + lam * np.sign(beta[active]))))
        if active.size
        else 0.0
    )
    inactive_excess = (
        float(np.max(np.maximum(np.abs(correlation[inactive]) - lam, 0.0)))
        if inactive.size
        else 0.0
    )
    return {
        "active_stationarity_max_abs": active_error,
        "inactive_violation_max": inactive_excess,
        "scaled_max": max(active_error, inactive_excess) / max(1.0, lam),
    }


def _append_and_refit(
    engine: ShardedData,
    base_directions: np.ndarray,
    base_atoms: list[np.ndarray],
    candidate: np.ndarray,
    lam: float,
) -> dict[str, Any]:
    master, beta0, _, objective0, active0 = _solve_master(
        engine,
        base_directions,
        lam,
        atoms=base_atoms,
    )
    direct0 = _direct_objective(engine, master, beta0, lam)
    master.append_atoms(candidate.reshape(-1, 1))
    beta1, _, objective1, active1 = master.solve(
        lam,
        tol=1e-10,
        batch_add=8,
    )
    direct1 = _direct_objective(engine, master, beta1, lam)
    return {
        "objective_before": objective0,
        "objective_after": float(objective1),
        "objective_before_direct": direct0,
        "objective_after_direct": direct1,
        "realized_full_corrected_gain": direct0 - direct1,
        "active_before": len(active0),
        "active_after": len(active1),
        "appended_coefficient": float(beta1[-1]),
        "appended_active": bool(abs(beta1[-1]) > 1e-10),
        "reported_vs_direct_before_abs_error": abs(objective0 - direct0),
        "reported_vs_direct_after_abs_error": abs(float(objective1) - direct1),
        "kkt": _kkt_error(engine, master, beta1, lam),
    }


def _balanced_schedule(repeats: int, seed: int) -> list[list[str]]:
    first = int(seed) % 2
    orders = []
    for repeat in range(repeats):
        if (repeat + first) % 2:
            orders.append([METHODS[1], METHODS[0]])
        else:
            orders.append([METHODS[0], METHODS[1]])
    return orders


def _time_rescore(
    engine: ShardedData,
    candidates: np.ndarray,
    nu_sh: list[np.ndarray],
    repeats: int,
    calls_per_repeat: int,
    seed: int,
) -> dict[str, Any]:
    # Exactly one untimed warm-up for each method.
    warm_corr = engine.rescore(candidates, nu_sh)
    warm_stats_corr, warm_sqnorm = engine.rescore_stats(candidates, nu_sh)
    engine.synchronize()

    schedule = _balanced_schedule(repeats, seed)
    values: dict[str, list[float]] = {method: [] for method in METHODS}
    correlation_errors: list[float] = []
    latest_stats_corr = warm_stats_corr
    latest_sqnorm = warm_sqnorm
    for order in schedule:
        for method in order:
            engine.synchronize()
            started = time.perf_counter_ns()
            for _ in range(calls_per_repeat):
                if method == "correlation_only":
                    output = engine.rescore(candidates, nu_sh)
                else:
                    output, latest_sqnorm = engine.rescore_stats(candidates, nu_sh)
                    latest_stats_corr = output
            engine.synchronize()
            elapsed = (time.perf_counter_ns() - started) * 1e-9 / calls_per_repeat
            values[method].append(float(elapsed))
            correlation_errors.append(float(np.max(np.abs(output - warm_corr))))

    positions = {
        method: [
            sum(order[position] == method for order in schedule)
            for position in range(len(METHODS))
        ]
        for method in METHODS
    }
    corr_stats = _summary(values["correlation_only"])
    fused_stats = _summary(values["correlation_and_sqnorm"])
    return {
        "warmups_per_method": 1,
        "repeats_per_method": repeats,
        "timed_calls_per_repeat": calls_per_repeat,
        "timed_calls_per_method": repeats * calls_per_repeat,
        "schedule": schedule,
        "position_counts": positions,
        "balanced_within_one": all(
            max(counts) - min(counts) <= 1 for counts in positions.values()
        ),
        "seconds": {
            "correlation_only": corr_stats,
            "correlation_and_sqnorm": fused_stats,
        },
        "median_stats_to_correlation_time_ratio": (
            fused_stats["median"] / max(corr_stats["median"], 1e-300)
        ),
        "median_incremental_seconds": (fused_stats["median"] - corr_stats["median"]),
        "warmup_correlation_max_abs_error": float(
            np.max(np.abs(warm_stats_corr - warm_corr))
        ),
        "all_timed_correlation_max_abs_error": max(correlation_errors, default=0.0),
        "stats_correlation": latest_stats_corr,
        "stats_sqnorm": latest_sqnorm,
    }


def _state_result(
    shape: Shape, regime: str, seed: int, repeats: int, calls_per_repeat: int
) -> dict[str, Any]:
    rss_before = _rss_bytes()
    state = _make_state(shape, regime, seed)
    X = state["X"]
    y = state["y"]
    base_directions = state["master"]
    candidates = state["candidates"]
    lam = float(state["lambda"])

    engine = ShardedData(
        X, y, devices=("cpu",), backend="numpy", dtype="float64", threads=False
    )
    base_atoms = engine.build_atoms(base_directions)
    master, beta, nu_sh, reported_objective, active = _solve_master(
        engine,
        base_directions,
        lam,
        atoms=base_atoms,
    )
    base_objective = _direct_objective(engine, master, beta, lam)
    residual = engine.gather_vec(nu_sh)  # nu = y - A beta

    timing = _time_rescore(
        engine,
        candidates,
        nu_sh,
        repeats,
        calls_per_repeat,
        seed,
    )
    correlation = np.asarray(timing.pop("stats_correlation"), dtype=np.float64)
    sqnorm = np.asarray(timing.pop("stats_sqnorm"), dtype=np.float64)
    decrease = _predicted_decrease(correlation, sqnorm, lam)

    # Explicit scalar soft threshold, followed by direct residual/objective
    # evaluation for every candidate (not merely an algebraic restatement).
    alpha = (
        np.sign(correlation)
        * np.maximum(np.abs(correlation) - lam, 0.0)
        / np.maximum(sqnorm, 1e-300)
    )
    candidate_atoms = np.maximum(X @ candidates, 0.0)
    direct_sqnorm = np.einsum("ij,ij->j", candidate_atoms, candidate_atoms)
    updated_residual = residual[:, None] - candidate_atoms * alpha[None, :]
    explicit_objective = 0.5 * np.einsum(
        "ij,ij->j", updated_residual, updated_residual
    ) + lam * (float(np.abs(beta).sum()) + np.abs(alpha))
    explicit_gain = base_objective - explicit_objective
    abs_error = np.abs(decrease - explicit_gain)
    scale = np.maximum.reduce(
        [np.ones_like(decrease), np.abs(decrease), np.abs(explicit_gain)]
    )
    rel_error = abs_error / scale
    tolerance = 2e-10 * scale
    passes = abs_error <= tolerance

    raw_index = int(np.argmax(np.abs(correlation)))
    decrease_index = int(np.argmax(decrease))
    raw_order = np.argsort(-np.abs(correlation), kind="stable")
    decrease_order = np.argsort(-decrease, kind="stable")
    raw_refit = _append_and_refit(
        engine,
        base_directions,
        base_atoms,
        candidates[:, raw_index],
        lam,
    )
    decrease_refit = _append_and_refit(
        engine,
        base_directions,
        base_atoms,
        candidates[:, decrease_index],
        lam,
    )
    raw_gain = float(raw_refit["realized_full_corrected_gain"])
    ordered_gain = float(decrease_refit["realized_full_corrected_gain"])
    gain_ratio = ordered_gain / raw_gain if raw_gain > 1e-12 else None
    time_ratio = float(timing["median_stats_to_correlation_time_ratio"])
    progress_per_rescore_second_ratio = (
        gain_ratio / time_ratio if gain_ratio is not None else None
    )

    checks = []
    for index in range(shape.candidates):
        checks.append(
            {
                "index": index,
                "source": state["candidate_labels"][index],
                "correlation": float(correlation[index]),
                "absolute_correlation": float(abs(correlation[index])),
                "sqnorm": float(sqnorm[index]),
                "soft_threshold_coefficient": float(alpha[index]),
                "predicted_decrease": float(decrease[index]),
                "explicit_objective_gain": float(explicit_gain[index]),
                "absolute_error": float(abs_error[index]),
                "scaled_relative_error": float(rel_error[index]),
                "pass": bool(passes[index]),
            }
        )

    rss_after = _rss_bytes()
    itemsize = X.dtype.itemsize
    result = {
        "shape": asdict(shape),
        "regime": regime,
        "seed": seed,
        "data_construction": {
            "base_columns_centered_and_scaled_to_unit_sample_std": True,
            "feature_scales": [float(v) for v in state["feature_scales"]],
            "feature_scale_ratio": float(
                np.max(state["feature_scales"]) / np.min(state["feature_scales"])
            ),
            "noise_std_fraction_of_clean_signal_std": 0.02,
            "noise_scale": float(state["noise_scale"]),
            "candidate_sources": {
                "planted_teacher": shape.planted_atoms,
                "positive_axes": shape.d,
                "negative_axes": shape.d,
                "random_unit": shape.candidates - shape.planted_atoms - 2 * shape.d,
            },
        },
        "lambda": lam,
        "lambda_fraction_of_max_zero_model_union_correlation": float(
            state["lambda_fraction"]
        ),
        "base_fit": {
            "reported_objective": reported_objective,
            "direct_objective": base_objective,
            "reported_vs_direct_abs_error": abs(reported_objective - base_objective),
            "active_atoms": len(active),
            "residual_l2": float(np.linalg.norm(residual)),
            "kkt": _kkt_error(engine, master, beta, lam),
        },
        "formula_verification": {
            "candidate_count": shape.candidates,
            "pass_count": int(np.count_nonzero(passes)),
            "failure_count": int(np.count_nonzero(~passes)),
            "max_absolute_error": float(np.max(abs_error)),
            "max_scaled_relative_error": float(np.max(rel_error)),
            "tolerance": "abs_error <= 2e-10 * max(1, abs(predicted), abs(explicit))",
            "direct_sqnorm_vs_rescore_max_abs_error": float(
                np.max(np.abs(direct_sqnorm - sqnorm))
            ),
            "candidate_checks": checks,
        },
        "choice_comparison": {
            "raw_index": raw_index,
            "raw_source": state["candidate_labels"][raw_index],
            "decrease_index": decrease_index,
            "decrease_source": state["candidate_labels"][decrease_index],
            "reversal": raw_index != decrease_index,
            "rho_raw": float(np.max(np.abs(correlation))),
            "rho_over_lambda": float(np.max(np.abs(correlation)) / lam),
            "raw_choice_predicted_decrease": float(decrease[raw_index]),
            "decrease_choice_predicted_decrease": float(decrease[decrease_index]),
            "predicted_decrease_ratio": float(
                decrease[decrease_index] / max(decrease[raw_index], 1e-300)
            ),
            "raw_rank_of_decrease_choice": int(
                np.flatnonzero(raw_order == decrease_index)[0]
            )
            + 1,
            "decrease_rank_of_raw_choice": int(
                np.flatnonzero(decrease_order == raw_index)[0]
            )
            + 1,
            "raw_refit": raw_refit,
            "decrease_refit": decrease_refit,
            "full_corrected_gain_ratio_decrease_over_raw": gain_ratio,
            "full_corrected_gain_per_rescore_second_ratio": (
                progress_per_rescore_second_ratio
            ),
            "raw_correlation_retained_for_kkt": True,
        },
        "rescore_timing": timing,
        "memory": {
            "rss_before_state_bytes": rss_before,
            "rss_after_state_bytes": rss_after,
            "process_peak_rss_bytes_so_far": _peak_rss_bytes(),
            "X_bytes": int(X.nbytes),
            "y_bytes": int(y.nbytes),
            "base_feature_storage_bytes": int(shape.n * shape.master_atoms * itemsize),
            "all_candidate_feature_buffer_bytes": int(
                shape.n * shape.candidates * itemsize
            ),
            "one_retained_finalist_feature_bytes": int(shape.n * itemsize),
            "correlation_output_bytes": int(shape.candidates * 8),
            "additional_sqnorm_output_bytes": int(shape.candidates * 8),
            "explicit_check_updated_residual_bytes": int(updated_residual.nbytes),
            "production_rescore_stats_retains_candidate_features": False,
        },
    }
    return result


def _geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return float(math.exp(statistics.fmean(math.log(value) for value in positive)))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reversals = [row for row in rows if row["choice_comparison"]["reversal"]]
    all_gain_ratios = [
        row["choice_comparison"]["full_corrected_gain_ratio_decrease_over_raw"]
        for row in rows
        if row["choice_comparison"]["full_corrected_gain_ratio_decrease_over_raw"]
        is not None
    ]
    reversal_gain_ratios = [
        row["choice_comparison"]["full_corrected_gain_ratio_decrease_over_raw"]
        for row in reversals
        if row["choice_comparison"]["full_corrected_gain_ratio_decrease_over_raw"]
        is not None
    ]
    progress_ratios = [
        row["choice_comparison"]["full_corrected_gain_per_rescore_second_ratio"]
        for row in rows
        if row["choice_comparison"]["full_corrected_gain_per_rescore_second_ratio"]
        is not None
    ]
    candidate_count = sum(
        row["formula_verification"]["candidate_count"] for row in rows
    )
    failures = sum(row["formula_verification"]["failure_count"] for row in rows)
    reversal_fraction = len(reversals) / len(rows)
    median_reversal_gain = (
        float(statistics.median(reversal_gain_ratios)) if reversal_gain_ratios else None
    )
    median_progress = (
        float(statistics.median(progress_ratios)) if progress_ratios else None
    )
    formula_gate = failures == 0
    reversal_gate = (
        reversal_fraction >= COMPONENT_RULE["minimum_state_reversal_fraction"]
    )
    corrected_gain_gate = (
        median_reversal_gain is not None
        and median_reversal_gain
        >= COMPONENT_RULE["minimum_median_full_corrected_gain_ratio_on_reversals"]
    )
    progress_gate = (
        median_progress is not None
        and median_progress
        >= COMPONENT_RULE["minimum_median_full_corrected_gain_per_rescore_second_ratio"]
    )
    component_go = (
        formula_gate and reversal_gate and corrected_gain_gate and progress_gate
    )
    return {
        "state_count": len(rows),
        "small_state_count": sum(row["shape"]["name"] == "small" for row in rows),
        "medium_state_count": sum(row["shape"]["name"] == "medium" for row in rows),
        "standardized_state_count": sum(
            row["regime"] == "standardized" for row in rows
        ),
        "anisotropic_state_count": sum(row["regime"] == "anisotropic" for row in rows),
        "candidate_formula_checks": candidate_count,
        "candidate_formula_failures": failures,
        "max_formula_absolute_error": max(
            row["formula_verification"]["max_absolute_error"] for row in rows
        ),
        "max_formula_scaled_relative_error": max(
            row["formula_verification"]["max_scaled_relative_error"] for row in rows
        ),
        "reversal_count": len(reversals),
        "reversal_fraction": reversal_fraction,
        "reversals_by_regime": {
            regime: sum(
                row["regime"] == regime and row["choice_comparison"]["reversal"]
                for row in rows
            )
            for regime in REGIMES
        },
        "full_corrected_gain_ratio_all_states": _summary(all_gain_ratios),
        "full_corrected_gain_ratio_reversal_states": (
            _summary(reversal_gain_ratios) if reversal_gain_ratios else None
        ),
        "full_corrected_gain_ratio_geometric_mean_reversal_states": (
            _geometric_mean(reversal_gain_ratios)
        ),
        "full_corrected_gain_per_rescore_second_ratio": _summary(progress_ratios),
        "stats_to_correlation_time_ratio_across_states": _summary(
            [
                row["rescore_timing"]["median_stats_to_correlation_time_ratio"]
                for row in rows
            ]
        ),
        "balanced_timing_every_state": all(
            row["rescore_timing"]["balanced_within_one"] for row in rows
        ),
        "max_rescore_correlation_disagreement": max(
            max(
                row["rescore_timing"]["warmup_correlation_max_abs_error"],
                row["rescore_timing"]["all_timed_correlation_max_abs_error"],
            )
            for row in rows
        ),
        "peak_process_rss_bytes": max(
            row["memory"]["process_peak_rss_bytes_so_far"] for row in rows
        ),
        "maximum_all_candidate_feature_buffer_bytes": max(
            row["memory"]["all_candidate_feature_buffer_bytes"] for row in rows
        ),
        "maximum_one_finalist_feature_bytes": max(
            row["memory"]["one_retained_finalist_feature_bytes"] for row in rows
        ),
        "component_gates": {
            "formula": formula_gate,
            "nonnegligible_reversals": reversal_gate,
            "full_corrected_gain_on_reversals": corrected_gain_gate,
            "full_corrected_gain_per_rescore_second": progress_gate,
        },
        "component_acceptance_pass": component_go,
        "go_no_go": (
            "MEETS_COMPONENT_CRITERIA"
            if component_go
            else "DOES_NOT_MEET_COMPONENT_CRITERIA"
        ),
        "interpretation_boundary": (
            "Component evidence only. An end-to-end performance claim would require "
            "an independently repeated matched time-to-target improvement of at least 20%."
        ),
    }


def _self_check() -> dict[str, Any]:
    X = np.asarray([[1.0], [-1.0], [2.0], [-2.0]])
    y = np.asarray([1.0, 0.25, 2.0, 0.5])
    candidates = np.asarray([[1.0, -1.0]])
    engine = ShardedData(X, y, dtype="float64")
    nu_sh = engine.scatter_vec(y)
    correlation, sqnorm = engine.rescore_stats(candidates, nu_sh)
    direct_atoms = np.maximum(X @ candidates, 0.0)
    direct_corr = direct_atoms.T @ y
    direct_sqnorm = np.einsum("ij,ij->j", direct_atoms, direct_atoms)
    lam = 0.5
    predicted = _predicted_decrease(correlation, sqnorm, lam)
    alpha = np.sign(correlation) * np.maximum(np.abs(correlation) - lam, 0.0) / sqnorm
    before = 0.5 * float(y @ y)
    after = np.asarray(
        [
            0.5
            * float(
                (y - alpha[j] * direct_atoms[:, j])
                @ (y - alpha[j] * direct_atoms[:, j])
            )
            + lam * abs(alpha[j])
            for j in range(candidates.shape[1])
        ]
    )
    explicit = before - after
    maximum_error = float(
        max(
            np.max(np.abs(correlation - direct_corr)),
            np.max(np.abs(sqnorm - direct_sqnorm)),
            np.max(np.abs(predicted - explicit)),
        )
    )
    return {
        "pass": maximum_error <= 1e-12,
        "maximum_abs_error": maximum_error,
        "candidate_count": candidates.shape[1],
        "correlation": [float(v) for v in correlation],
        "sqnorm": [float(v) for v in sqnorm],
        "predicted_decrease": [float(v) for v in predicted],
        "explicit_decrease": [float(v) for v in explicit],
    }


def _environment() -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numpy_config": _numpy_config(),
        "cpu_model": _run_text(
            [
                "bash",
                "-lc",
                "lscpu | sed -n 's/^Model name:[[:space:]]*//p'",
            ]
        ),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "memory_total_bytes": _memory_total_bytes(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def run(states_per_cell: int, repeats: int, calls_per_repeat: int) -> dict[str, Any]:
    self_check = _self_check()
    if not self_check["pass"]:
        raise AssertionError(f"self-check failed: {self_check}")
    rows = []
    for shape_index, shape in enumerate(SHAPES):
        for regime_index, regime in enumerate(REGIMES):
            for state_index in range(states_per_cell):
                seed = _state_seed(shape_index, regime_index, state_index)
                rows.append(
                    _state_result(
                        shape,
                        regime,
                        seed,
                        repeats,
                        calls_per_repeat,
                    )
                )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_scope": "frozen-state component benchmark; not end-to-end evidence",
        "objective": (
            "Preserved finite-atom Lasso restriction of the scalar-output ReLU "
            "regularized least-squares objective"
        ),
        "component_rule": COMPONENT_RULE,
        "config": {
            "states_per_shape_regime_cell": states_per_cell,
            "state_count": len(rows),
            "repeats_per_rescore_method": repeats,
            "timed_calls_per_rescore_repeat": calls_per_repeat,
            "warmups_per_rescore_method": 1,
            "shapes": [asdict(shape) for shape in SHAPES],
            "regimes": list(REGIMES),
            "backend": "numpy float64, one logical row shard",
            "master_solve_tolerance": 1e-10,
            "master_ridge": 1e-12,
        },
        "self_check": self_check,
        "environment": _environment(),
        "states": rows,
        "summary": _aggregate(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-per-cell", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--timing-calls-per-repeat", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=Path("results/candidate_order_study.json")
    )
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.states_per_cell < 5:
        raise SystemExit("require states-per-cell>=5 (at least 20 total states)")
    if args.repeats < 3:
        raise SystemExit("require repeats>=3")
    if args.timing_calls_per_repeat < 1:
        raise SystemExit("require timing-calls-per-repeat>=1")
    check = _self_check()
    if args.self_check_only:
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0 if check["pass"] else 1
    if not check["pass"]:
        raise SystemExit(f"self-check failed: {check}")

    report = run(
        args.states_per_cell,
        args.repeats,
        args.timing_calls_per_repeat,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
