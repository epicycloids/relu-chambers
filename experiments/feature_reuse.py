"""Preregistered finalist-feature reuse microbenchmark.

The primary endpoint is the balanced median wall time of one final full-data
rescore, oriented selection/deduplication, and restricted-master append.  The
candidate passes the isolated gate only at a time ratio <= 0.90.  A small
matched end-to-end comparison is run only after that gate passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from experiments.controlled_benchmark import (
    _array_digest,
    _atomic_json,
    _balanced_orders,
    _metadata,
    _position_balance,
    _quartiles,
    _read_meminfo,
    _RSSMonitor,
)
from relu_chambers.adaptive_pricing import _select_distinct
from relu_chambers.restricted_master import AtomLasso
from relu_chambers.solver import _dedup_cols_indexed, make_engine, solve_rcg

PREREGISTRATION = {
    "claim": "finalist-feature reuse",
    "primary_endpoint": "median(reuse rescore+append) / median(rebuild rescore+append)",
    "isolated_gate_max_ratio": 0.90,
    "warmups": 2,
    "measured_repeats": 8,
    "schedule": "seeded cyclic Latin order, separately balanced for warmup and measured runs",
    "dataset": {
        "seed": 18018,
        "n": 120_000,
        "d": 64,
        "base_columns": 8,
        "finalist_columns": 16,
        "selected_columns": 8,
        "dtype": "float64",
    },
    "retained_feature_max_bytes": 128 * 1024 * 1024,
    "available_memory_max_fraction": 0.25,
    "finalist_layout": "two sign blocks, four selected columns per sign",
    "quality_checks": [
        "same selected oriented directions",
        "same appended feature values",
        "same A.T@y and squared norms",
        "same appended column count",
    ],
    "conditional_end_to_end": {
        "run_only_if_isolated_gate_passes": True,
        "warmups": 1,
        "measured_repeats": 3,
        "objective_tolerance": 1e-8,
    },
}


def _normalize(U: np.ndarray) -> np.ndarray:
    return U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-30)


def _features_host(engine, A_sh) -> np.ndarray:
    return np.concatenate([engine._host64(A) for A in A_sh], axis=0)


def _prepare_master(engine, Ubase, Abase, aty_base, sq_base) -> AtomLasso:
    master = AtomLasso(engine)
    master.set_atoms(Ubase, A_sh=Abase, aty=aty_base, sqnorms=sq_base)
    return master


def _concat_feature_blocks(engine, blocks):
    if len(blocks) == 1:
        return blocks[0]
    if engine.backend == "numpy":
        return [np.concatenate(parts, axis=1) for parts in zip(*blocks)]
    import torch  # type: ignore

    return [torch.cat(parts, dim=1) for parts in zip(*blocks)]


def _run_isolated(
    variant, engine, nu_sh, Ubase, Abase, aty_base, sq_base, Ufinal, cap_bytes
):
    master = _prepare_master(engine, Ubase, Abase, aty_base, sq_base)
    gc.collect()
    with _RSSMonitor(interval=0.001) as rss:
        engine.synchronize()
        start = time.perf_counter()
        if variant not in {"rebuild", "reuse"}:  # pragma: no cover
            raise ValueError(variant)

        # Mirror search_signed_atoms's two sign-specific finalist blocks and cache join,
        # followed by solver's global ordering, oriented deduplication, cache
        # slice, and append.  Search itself is deliberately outside this
        # component endpoint.
        sign_blocks = [(1.0, Ufinal[:, :8], 0), (-1.0, Ufinal[:, 8:], 8)]
        additions = []
        feature_blocks = []
        aty_blocks = []
        sqnorm_blocks = []
        for sign, Ublock, offset in sign_blocks:
            if variant == "reuse":
                stats = engine.rescore_finalists(
                    Ublock,
                    nu_sh,
                    retain_features=True,
                    retained_feature_max_bytes=cap_bytes,
                )
                if not stats["retained"]:
                    raise RuntimeError(
                        "preregistered retained buffer unexpectedly exceeded cap"
                    )
                corr = stats["corr"]
            else:
                stats = None
                corr = engine.rescore(Ublock, nu_sh)
            signed_corr = corr * sign
            ranked = _select_distinct(
                signed_corr, Ublock, 4, 0.999, return_indices=True
            )
            indices = [idx for _, _, idx in ranked]
            additions.extend(
                (value, value, u, offset + idx) for value, u, idx in ranked
            )
            if stats is not None and indices:
                feature_blocks.append(engine.slice_cols(stats["features"], indices))
                aty_blocks.append(np.asarray(stats["aty"])[indices])
                sqnorm_blocks.append(np.asarray(stats["sqnorm"])[indices])
            stats = None

        cache = None
        if variant == "reuse":
            cache = {
                "A_sh": _concat_feature_blocks(engine, feature_blocks),
                "aty": np.concatenate(aty_blocks),
                "sqnorm": np.concatenate(sqnorm_blocks),
            }
        ranked_additions = sorted(enumerate(additions), key=lambda item: -item[1][0])
        ordered = [(index, item[2], item[3]) for index, item in ranked_additions]
        fresh, positions = _dedup_cols_indexed(
            [u for _, u, _ in ordered], master.U, normalize=False
        )
        cache_indices = [ordered[pos][0] for pos in positions]
        selected = [ordered[pos][2] for pos in positions]
        if cache is None:
            master.append_atoms(np.stack(fresh, axis=1))
        else:
            master.append_atoms(
                np.stack(fresh, axis=1),
                A_sh=engine.slice_cols(cache["A_sh"], cache_indices),
                aty=np.asarray(cache["aty"])[cache_indices],
                sqnorms=np.asarray(cache["sqnorm"])[cache_indices],
            )
        cache = None
        engine.synchronize()
        elapsed = time.perf_counter() - start
    Ahost = _features_host(engine, master.A)
    rebuilt_host = _features_host(engine, engine.build_atoms(master.U))
    aty_check = engine.screen(master.A, engine.ys)
    sq_check = engine.sqnorms(master.A)
    return {
        "variant": variant,
        "seconds": elapsed,
        "selected_finalist_indices": selected,
        "appended_columns": len(fresh),
        "A_sha256": _array_digest(Ahost),
        "Aty_sha256": _array_digest(master.Aty),
        "norms_sha256": _array_digest(master.norms),
        "validation": {
            "feature_vs_independent_rebuild_max_abs": float(
                np.max(np.abs(Ahost - rebuilt_host))
            ),
            "aty_max_abs": float(np.max(np.abs(master.Aty - aty_check))),
            "sqnorm_max_abs": float(np.max(np.abs(master.norms**2 - sq_check))),
        },
        "memory": {
            "rss_start_bytes": rss.start_bytes,
            "rss_end_bytes": rss.end_bytes,
            "rss_sampled_peak_bytes": rss.peak_bytes,
            "rss_sampled_peak_delta_bytes": (
                None
                if rss.start_bytes is None or rss.peak_bytes is None
                else max(0, rss.peak_bytes - rss.start_bytes)
            ),
            "retained_feature_payload_bytes": (
                engine.feature_buffer_bytes(8) if variant == "reuse" else 0
            ),
            "required_peak_feature_payload_bytes": (
                engine.feature_buffer_bytes(Ufinal.shape[1])
                if variant == "reuse"
                else 0
            ),
        },
    }


def _recomputed_objective(X, y, lam, solution):
    prediction = np.maximum(X @ solution["U"], 0.0) @ solution["beta"]
    residual = prediction - y
    return 0.5 * float(residual @ residual) + lam * float(
        np.abs(solution["beta"]).sum()
    )


def _run_end_to_end(order_seed=18020):
    rng = np.random.default_rng(18019)
    n, d = 3_000, 8
    X = rng.standard_normal((n, d))
    teacher = _normalize(rng.standard_normal((d, 3)))
    y = np.maximum(X @ teacher, 0.0) @ np.array([0.9, -0.6, 0.4])
    lam = 0.1
    solver = dict(
        max_iter=6,
        eps_heur=0,
        add_per_round=4,
        max_cols=64,
        certify_every=0,
        gated_every=0,
        final_verify=False,
    )
    oracle = dict(
        ascent_trial="damped",
        candidate_order="correlation",
        n_random=8,
        n_subspace=4,
        n_perturb=0,
        sketch_m=0,
        iters_sketch=8,
        iters_full=2,
        top_polish=4,
    )
    records = []
    schedules = {
        "warmup": _balanced_orders(["rebuild", "reuse"], 1, order_seed),
        "measured": _balanced_orders(["rebuild", "reuse"], 3, order_seed + 1),
    }
    for kind, schedule in schedules.items():
        for round_index, order in enumerate(schedule):
            for position, variant in enumerate(order):
                gc.collect()
                kwargs = dict(oracle)
                kwargs.update(
                    retain_finalist_features=(variant == "reuse"),
                    retained_feature_max_bytes=128 * 1024 * 1024,
                )
                with _RSSMonitor(interval=0.002) as rss:
                    start = time.perf_counter()
                    solution = solve_rcg(
                        X,
                        y,
                        lam,
                        rng_seed=18021,
                        oracle_kwargs=kwargs,
                        **solver,
                    )
                    seconds = time.perf_counter() - start
                records.append(
                    {
                        "kind": kind,
                        "round": round_index,
                        "position": position,
                        "variant": variant,
                        "seconds": seconds,
                        "objective": _recomputed_objective(X, y, lam, solution),
                        "n_cols": int(solution["n_cols"]),
                        "n_active": int(solution["n_active"]),
                        "feature_reuse": solution["feature_reuse"],
                        "memory": {
                            "rss_start_bytes": rss.start_bytes,
                            "rss_end_bytes": rss.end_bytes,
                            "rss_sampled_peak_bytes": rss.peak_bytes,
                            "rss_sampled_peak_delta_bytes": (
                                None
                                if rss.start_bytes is None or rss.peak_bytes is None
                                else max(0, rss.peak_bytes - rss.start_bytes)
                            ),
                        },
                    }
                )
    measured = [record for record in records if record["kind"] == "measured"]
    summary = {}
    for variant in ("rebuild", "reuse"):
        selected = [record for record in measured if record["variant"] == variant]
        summary[variant] = {
            "seconds": _quartiles([record["seconds"] for record in selected]),
            "objective": _quartiles([record["objective"] for record in selected]),
            "rss_sampled_peak_delta_bytes": _quartiles(
                [
                    record["memory"]["rss_sampled_peak_delta_bytes"]
                    for record in selected
                    if record["memory"]["rss_sampled_peak_delta_bytes"] is not None
                ]
            ),
        }
    ratio = (
        summary["reuse"]["seconds"]["median"] / summary["rebuild"]["seconds"]["median"]
    )
    objective_delta = abs(
        summary["reuse"]["objective"]["median"]
        - summary["rebuild"]["objective"]["median"]
    )
    return {
        "status": "run_after_isolated_gate_pass",
        "schedule": schedules,
        "balance": {key: _position_balance(value) for key, value in schedules.items()},
        "records": records,
        "summary": summary,
        "median_time_ratio_reuse_over_rebuild": ratio,
        "median_objective_absolute_delta": objective_delta,
        "objective_tolerance": 1e-8,
        "quality_pass": bool(objective_delta <= 1e-8),
    }


def run(output: Path) -> dict:
    memory_at_start = _read_meminfo()
    available_memory = memory_at_start.get("MemAvailable")
    config = PREREGISTRATION["dataset"]
    rng = np.random.default_rng(config["seed"])
    X = rng.standard_normal((config["n"], config["d"]))
    y = rng.standard_normal(config["n"])
    nu = rng.standard_normal(config["n"])
    Ubase = _normalize(rng.standard_normal((config["d"], config["base_columns"])))
    Ufinal = _normalize(rng.standard_normal((config["d"], config["finalist_columns"])))
    engine = make_engine(X, y, dtype=config["dtype"])
    nu_sh = engine.scatter_vec(nu)
    base_stats = engine.rescore_finalists(
        Ubase,
        nu_sh,
        retain_features=True,
        retained_feature_max_bytes=PREREGISTRATION["retained_feature_max_bytes"],
    )
    Abase = base_stats["features"]
    schedules = {
        "warmup": _balanced_orders(
            ["rebuild", "reuse"], PREREGISTRATION["warmups"], 18022
        ),
        "measured": _balanced_orders(
            ["rebuild", "reuse"], PREREGISTRATION["measured_repeats"], 18023
        ),
    }
    records = []
    for kind, schedule in schedules.items():
        for round_index, order in enumerate(schedule):
            for position, variant in enumerate(order):
                record = _run_isolated(
                    variant,
                    engine,
                    nu_sh,
                    Ubase,
                    Abase,
                    base_stats["aty"],
                    base_stats["sqnorm"],
                    Ufinal,
                    PREREGISTRATION["retained_feature_max_bytes"],
                )
                record.update(kind=kind, round=round_index, position=position)
                records.append(record)

    measured = [record for record in records if record["kind"] == "measured"]
    summaries = {}
    for variant in ("rebuild", "reuse"):
        selected = [record for record in measured if record["variant"] == variant]
        summaries[variant] = {
            "seconds": _quartiles([record["seconds"] for record in selected]),
            "rss_sampled_peak_delta_bytes": _quartiles(
                [
                    record["memory"]["rss_sampled_peak_delta_bytes"]
                    for record in selected
                    if record["memory"]["rss_sampled_peak_delta_bytes"] is not None
                ]
            ),
            "retained_feature_payload_bytes": max(
                record["memory"]["retained_feature_payload_bytes"]
                for record in selected
            ),
            "required_peak_feature_payload_bytes": max(
                record["memory"]["required_peak_feature_payload_bytes"]
                for record in selected
            ),
        }
    ratio = (
        summaries["reuse"]["seconds"]["median"]
        / summaries["rebuild"]["seconds"]["median"]
    )
    selections = {
        variant: sorted(
            {
                (tuple(record["selected_finalist_indices"]), record["appended_columns"])
                for record in measured
                if record["variant"] == variant
            }
        )
        for variant in ("rebuild", "reuse")
    }
    validation_pass = all(
        record["validation"]["feature_vs_independent_rebuild_max_abs"] == 0.0
        and record["validation"]["aty_max_abs"] <= 1e-9
        and record["validation"]["sqnorm_max_abs"] <= 1e-8
        for record in measured
    )
    feature_digests = {
        variant: {
            record["A_sha256"] for record in measured if record["variant"] == variant
        }
        for variant in ("rebuild", "reuse")
    }
    quality_pass = (
        len(selections["rebuild"]) == len(selections["reuse"]) == 1
        and selections["rebuild"] == selections["reuse"]
        and validation_pass
        and feature_digests["rebuild"] == feature_digests["reuse"]
    )
    observed_reuse_peak = summaries["reuse"]["rss_sampled_peak_delta_bytes"]["max"]
    required_payload_peak = summaries["reuse"]["required_peak_feature_payload_bytes"]
    memory_observed = max(observed_reuse_peak, required_payload_peak)
    memory_limit = (
        None
        if available_memory is None
        else PREREGISTRATION["available_memory_max_fraction"] * available_memory
    )
    memory_pass = bool(memory_limit is not None and memory_observed <= memory_limit)
    isolated_pass = bool(
        quality_pass
        and ratio <= PREREGISTRATION["isolated_gate_max_ratio"]
        and memory_pass
    )
    payload = {
        "schema_version": 1,
        "experiment": "finalist_feature_reuse",
        "preregistration": PREREGISTRATION,
        "metadata": _metadata(),
        "input_digests": {
            "X": _array_digest(X),
            "y": _array_digest(y),
            "nu": _array_digest(nu),
            "Ubase": _array_digest(Ubase),
            "Ufinal": _array_digest(Ufinal),
        },
        "isolated": {
            "schedule": schedules,
            "balance": {
                key: _position_balance(value) for key, value in schedules.items()
            },
            "records": records,
            "summary": summaries,
            "median_time_ratio_reuse_over_rebuild": ratio,
            "speedup_rebuild_over_reuse": 1.0 / ratio,
            "quality_pass": quality_pass,
            "isolated_gate_pass": isolated_pass,
            "memory_gate": {
                "available_memory_bytes_at_start": available_memory,
                "max_fraction": PREREGISTRATION["available_memory_max_fraction"],
                "limit_bytes": memory_limit,
                "observed_or_required_peak_bytes": memory_observed,
                "pass": memory_pass,
            },
            "selection_checks": {
                key: [list(item) for item in value] for key, value in selections.items()
            },
            "feature_digest_checks": {
                key: sorted(value) for key, value in feature_digests.items()
            },
            "validation_pass": validation_pass,
        },
    }
    payload["end_to_end"] = (
        _run_end_to_end()
        if isolated_pass
        else {"status": "not_run_isolated_quality_time_or_memory_gate_failed"}
    )
    _atomic_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/feature_reuse_study.json")
    )
    args = parser.parse_args()
    result = run(args.output)
    isolated = result["isolated"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ratio": isolated["median_time_ratio_reuse_over_rebuild"],
                "quality_pass": isolated["quality_pass"],
                "isolated_gate_pass": isolated["isolated_gate_pass"],
                "end_to_end_status": result["end_to_end"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
