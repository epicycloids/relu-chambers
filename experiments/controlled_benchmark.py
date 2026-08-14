"""Controlled end-to-end timing harness for :func:`relu_chambers.solve_rcg`.

The default invocation is deliberately a smoke test, not performance evidence::

    python -m experiments.controlled_benchmark --output /tmp/rcg-smoke.json

Larger measurements must be requested explicitly (``--large`` for more than
10,000 rows).  Multiple ``--backend`` arguments are interleaved in a balanced,
deterministic order.  ``cpu-shards:N`` means N logical row shards on this host;
it is never described as N physical CPUs or sockets.

The solver does not yet expose transfer, synchronization, or full-data-pass
counters, nor the finer pricing subphases requested by the measurement
contract.  This harness records those fields as unavailable instead of
inventing measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from relu_chambers.solver import solve_rcg
from relu_chambers.synthetic_data import make_planted

SCHEMA_VERSION = 1
LARGE_ROW_THRESHOLD = 10_000
DEFAULT_PHASES = ("master", "gated", "price", "cert", "other", "verify")


@dataclass(frozen=True)
class Backend:
    name: str
    device: str
    devices: tuple[str, ...] | None
    threads: bool
    data_dtype: str
    hardware_claim: str
    physical_multi_device_speedup_claim_eligible: bool
    ascent_trial: str = "both"
    candidate_order: str = "correlation"
    search_backend: str = "host"
    retain_finalist_features: bool = False
    retained_feature_max_bytes: int = 256 * 1024 * 1024


def _parse_backend(text: str, requested_dtype: str) -> Backend:
    spec = text.strip().lower()
    if spec in {"cpu", "numpy"}:
        dtype = "float64" if requested_dtype == "auto" else requested_dtype
        return Backend(
            "cpu", "cpu", None, False, dtype, "single NumPy host path", False
        )
    if spec.startswith("cpu-shards:"):
        try:
            count = int(spec.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid backend {text!r}") from exc
        if count < 1:
            raise ValueError("cpu-shards count must be positive")
        dtype = "float64" if requested_dtype == "auto" else requested_dtype
        return Backend(
            f"cpu-shards:{count}",
            "cpu",
            ("cpu",) * count,
            True,
            dtype,
            f"{count} logical row shards on one host; not {count} physical CPUs",
            False,
        )
    if spec in {"torch-cpu", "torch:cpu"}:
        dtype = "float64" if requested_dtype == "auto" else requested_dtype
        return Backend(
            "torch-cpu",
            "torch:cpu",
            None,
            False,
            dtype,
            "single Torch CPU device",
            False,
        )
    if spec in {"cuda", "rocm"}:
        dtype = "float32" if requested_dtype == "auto" else requested_dtype
        return Backend(
            spec, "cuda", None, False, dtype, "one accelerator selected by Torch", False
        )
    if spec.startswith("cuda:"):
        tail = spec[len("cuda:") :]
        ids = [part.strip() for part in tail.split(",") if part.strip()]
        if not ids or any(not item.isdigit() for item in ids):
            raise ValueError(
                "CUDA backends use cuda, cuda:0, or cuda:0,1 (not repeated prefixes)"
            )
        devices = tuple(f"cuda:{item}" for item in ids)
        dtype = "float32" if requested_dtype == "auto" else requested_dtype
        multi = len(devices) >= 2
        return Backend(
            spec,
            devices[0],
            devices if multi else None,
            False,
            dtype,
            f"{len(devices)} physical accelerator device(s) requested",
            multi,
        )
    raise ValueError(
        f"unknown backend {text!r}; use cpu, cpu-shards:N, torch-cpu, cuda, or cuda:0,1"
    )


def _parse_case(text: str, requested_dtype: str) -> Backend:
    """Parse a labeled backend plus objective-preserving oracle options."""
    if "=" not in text:
        raise ValueError(
            "case syntax is LABEL=BACKEND[;ascent=MODE][;order=MODE]"
            "[;search=MODE][;reuse=on|off][;reuse-cap=BYTES]"
        )
    label, specification = text.split("=", 1)
    label = label.strip()
    parts = [part.strip() for part in specification.split(";") if part.strip()]
    if not label or not parts:
        raise ValueError("case requires a nonempty label and backend")
    base = _parse_backend(parts[0], requested_dtype)
    options: dict[str, str] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ValueError(f"invalid case option {item!r}")
        key, value = (piece.strip().lower() for piece in item.split("=", 1))
        if key in options:
            raise ValueError(f"duplicate case option {key!r}")
        options[key] = value
    unknown = set(options) - {"ascent", "order", "search", "reuse", "reuse-cap"}
    if unknown:
        raise ValueError(f"unknown case options: {sorted(unknown)}")
    ascent = options.get("ascent", "both")
    order = options.get("order", "correlation")
    search = options.get("search", "host")
    reuse_text = options.get("reuse", "off")
    if ascent not in {"both", "damped", "fixed", "packed"}:
        raise ValueError(f"unknown ascent mode {ascent!r}")
    if order not in {"correlation", "decrease"}:
        raise ValueError(f"unknown candidate order {order!r}")
    if search not in {"host", "device", "auto"}:
        raise ValueError(f"unknown search backend {search!r}")
    if reuse_text not in {"on", "off", "true", "false", "1", "0"}:
        raise ValueError(f"unknown finalist-feature reuse mode {reuse_text!r}")
    reuse = reuse_text in {"on", "true", "1"}
    try:
        reuse_cap = int(options.get("reuse-cap", str(256 * 1024 * 1024)))
    except ValueError as exc:
        raise ValueError("reuse-cap must be a nonnegative integer byte count") from exc
    if reuse_cap < 0:
        raise ValueError("reuse-cap must be a nonnegative integer byte count")
    return replace(
        base,
        name=label,
        ascent_trial=ascent,
        candidate_order=order,
        search_backend=search,
        retain_finalist_features=reuse,
        retained_feature_max_bytes=reuse_cap,
    )


def _balanced_orders(names: list[str], rounds: int, seed: int) -> list[list[str]]:
    """Cyclic Latin schedule; position counts differ by at most one."""
    if not names:
        raise ValueError("at least one backend is required")
    base = list(names)
    random.Random(seed).shuffle(base)
    return [base[r % len(base) :] + base[: r % len(base)] for r in range(rounds)]


def _position_balance(orders: list[list[str]]) -> dict[str, Any]:
    names = sorted({name for order in orders for name in order})
    counts = {name: [0] * len(names) for name in names}
    for order in orders:
        for pos, name in enumerate(order):
            counts[name][pos] += 1
    flat = [value for row in counts.values() for value in row]
    return {
        "position_counts": counts,
        "max_position_count_difference": max(flat) - min(flat) if flat else 0,
        "balanced_within_one": (max(flat) - min(flat) <= 1) if flat else True,
    }


def _proc_status_bytes(field: str) -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _vmstat_counters() -> dict[str, int]:
    wanted = {"pswpin", "pswpout", "pgmajfault"}
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, value = line.split()
            if key in wanted:
                result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


class _RSSMonitor:
    """Low-overhead host RSS sampler; peak is sampled, not allocator-exact."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.start_bytes: int | None = None
        self.end_bytes: int | None = None
        self.peak_bytes: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start_bytes = _proc_status_bytes("VmRSS")
        self.peak_bytes = self.start_bytes

        def sample() -> None:
            while not self._stop.wait(self.interval):
                value = _proc_status_bytes("VmRSS")
                self.samples += 1
                if value is not None and (
                    self.peak_bytes is None or value > self.peak_bytes
                ):
                    self.peak_bytes = value

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 2 * self.interval))
        self.end_bytes = _proc_status_bytes("VmRSS")
        if self.end_bytes is not None and (
            self.peak_bytes is None or self.end_bytes > self.peak_bytes
        ):
            self.peak_bytes = self.end_bytes


def _torch_module():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _cuda_devices(backend: Backend) -> list[str]:
    if not backend.device.startswith("cuda"):
        return []
    if backend.devices is not None:
        return list(dict.fromkeys(backend.devices))
    return [backend.device]


def _synchronize(backend: Backend) -> int:
    devices = _cuda_devices(backend)
    if not devices:
        return 0
    torch = _torch_module()
    if torch is None:
        raise RuntimeError("accelerator backend requested but Torch is unavailable")
    for device in devices:
        torch.cuda.synchronize(device)
    return len(devices)


def _reset_device_peaks(backend: Backend) -> None:
    torch = _torch_module()
    if torch is None:
        return
    for device in _cuda_devices(backend):
        torch.cuda.reset_peak_memory_stats(device)


def _device_memory(backend: Backend) -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        return {}
    result: dict[str, Any] = {}
    for device in _cuda_devices(backend):
        result[device] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    return result


def _recompute_objective(
    X: np.ndarray, y: np.ndarray, lam: float, U: Any, beta: Any, chunk: int = 65_536
) -> dict[str, Any]:
    Uv = np.asarray(U, dtype=np.float64)
    bv = np.asarray(beta, dtype=np.float64).reshape(-1)
    if Uv.ndim != 2 or Uv.shape[1] != bv.size:
        return {"valid": False, "reason": "U/beta shape mismatch", "obj": None}
    if not np.all(np.isfinite(Uv)) or not np.all(np.isfinite(bv)):
        return {"valid": False, "reason": "non-finite model", "obj": None}
    residual_sq = 0.0
    for start in range(0, X.shape[0], chunk):
        end = min(X.shape[0], start + chunk)
        pred = np.maximum(np.asarray(X[start:end], np.float64) @ Uv, 0.0) @ bv
        residual = pred - np.asarray(y[start:end], np.float64)
        residual_sq += float(residual @ residual)
    loss = 0.5 * residual_sq
    penalty = float(lam * np.abs(bv).sum())
    value = loss + penalty
    return {
        "valid": bool(math.isfinite(value)),
        "reason": None,
        "obj": value,
        "loss": loss,
        "penalty": penalty,
    }


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _scalar_output(sol: dict[str, Any], verified: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "obj",
        "n_active",
        "n_cols",
        "rho_lb",
        "gap_cert",
        "gap_heur",
        "theta",
        "n_iter",
        "stop_reason",
        "time",
        "ub_plus",
        "ub_minus",
    )
    output = {key: sol.get(key) for key in fields if key in sol}
    output["recomputed"] = verified
    output["history"] = sol.get("history", {})
    output["feature_reuse"] = sol.get("feature_reuse", {})
    if "U" in sol:
        U = np.asarray(sol["U"])
        output["U_shape"] = list(U.shape)
        output["U_sha256"] = _array_digest(U)
    if "beta" in sol:
        beta = np.asarray(sol["beta"])
        output["beta_shape"] = list(beta.shape)
        output["beta_sha256"] = _array_digest(beta)
        output["beta_l1"] = float(np.abs(beta).sum())
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch_module()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _run_once(
    backend: Backend,
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    solver_config: dict[str, Any],
    solver_seed: int,
    kind: str,
    round_index: int,
    sequence_index: int,
) -> dict[str, Any]:
    gc.collect()
    _seed_everything(solver_seed)
    try:
        pre_syncs = _synchronize(backend)
        _reset_device_peaks(backend)
    except Exception as exc:
        return {
            "kind": kind,
            "round": round_index,
            "sequence_index": sequence_index,
            "backend": backend.name,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    sol: dict[str, Any] | None = None
    verify: dict[str, Any] = {"valid": False, "reason": "solver failed", "obj": None}
    syncs = pre_syncs
    vmstat_before = _vmstat_counters()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    self_swap_before = _proc_status_bytes("VmSwap")
    with _RSSMonitor() as rss:
        start = time.perf_counter()
        try:
            effective_config = dict(solver_config)
            effective_config["oracle_kwargs"] = {
                "ascent_trial": backend.ascent_trial,
                "candidate_order": backend.candidate_order,
                "search_backend": backend.search_backend,
                "retain_finalist_features": backend.retain_finalist_features,
                "retained_feature_max_bytes": backend.retained_feature_max_bytes,
            }
            sol = solve_rcg(
                X,
                y,
                lam,
                device=backend.device,
                devices=list(backend.devices) if backend.devices is not None else None,
                dtype_data=backend.data_dtype,
                threads=backend.threads,
                rng_seed=solver_seed,
                **effective_config,
            )
            syncs += _synchronize(backend)
            verify_start = time.perf_counter()
            verify = _recompute_objective(X, y, lam, sol.get("U"), sol.get("beta"))
            harness_verify_seconds = time.perf_counter() - verify_start
            wall_seconds = time.perf_counter() - start
            status, error, tb = "ok", None, None
        except Exception as exc:
            with contextlib.suppress(Exception):
                syncs += _synchronize(backend)
            wall_seconds = time.perf_counter() - start
            harness_verify_seconds = None
            status, error, tb = "error", repr(exc), traceback.format_exc()
    vmstat_after = _vmstat_counters()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    self_swap_after = _proc_status_bytes("VmSwap")
    vmstat_delta = {
        key: vmstat_after[key] - vmstat_before[key]
        for key in vmstat_before.keys() & vmstat_after.keys()
    }
    process_major_faults = int(usage_after.ru_majflt - usage_before.ru_majflt)
    process_swap_change = (
        None
        if self_swap_before is None or self_swap_after is None
        else int(self_swap_after - self_swap_before)
    )

    phases = dict(sol.get("timings", {})) if sol is not None else {}
    for phase in DEFAULT_PHASES:
        phases.setdefault(phase, 0.0 if phase == "verify" else None)
    measured_phase_sum = sum(
        float(v)
        for v in phases.values()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    )
    run = {
        "kind": kind,
        "round": round_index,
        "sequence_index": sequence_index,
        "backend": backend.name,
        "status": status,
        "wall_seconds": wall_seconds,
        "oracle_variant": {
            "ascent_trial": backend.ascent_trial,
            "candidate_order": backend.candidate_order,
            "search_backend": backend.search_backend,
            "retain_finalist_features": backend.retain_finalist_features,
            "retained_feature_max_bytes": backend.retained_feature_max_bytes,
        },
        "solver_phase_seconds": phases,
        "solver_phase_sum_seconds": measured_phase_sum,
        "harness_objective_verification_seconds": harness_verify_seconds,
        "unattributed_seconds": wall_seconds - measured_phase_sum,
        "memory": {
            "host_rss_start_bytes": rss.start_bytes,
            "host_rss_end_bytes": rss.end_bytes,
            "host_rss_sampled_peak_bytes": rss.peak_bytes,
            "host_rss_samples": rss.samples,
            "system_vmstat_delta": vmstat_delta,
            "system_swap_io_observed": bool(
                vmstat_delta.get("pswpin", 0) or vmstat_delta.get("pswpout", 0)
            ),
            "process_major_faults": process_major_faults,
            "process_vm_swap_start_bytes": self_swap_before,
            "process_vm_swap_end_bytes": self_swap_after,
            "process_vm_swap_change_bytes": process_swap_change,
            "process_paging_observed": bool(
                process_major_faults or (process_swap_change not in (None, 0))
            ),
            "device": _device_memory(backend),
        },
        "counters": {
            "harness_device_synchronizations": syncs,
            "solver_device_synchronizations": None,
            "host_device_transfer_bytes": None,
            "full_data_passes": None,
            "status": "unavailable: solve_rcg does not expose these counters",
        },
        "output": _scalar_output(sol, verify) if sol is not None else None,
        "error": error,
        "traceback": tb,
    }
    return run


def _quartiles(values: list[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": len(values),
        "raw": values,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summaries(
    backends: list[Backend], measured: list[dict[str, Any]], quality_rtol: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: dict[str, Any] = {}
    for backend in backends:
        runs = [run for run in measured if run["backend"] == backend.name]
        successful = [run for run in runs if run["status"] == "ok"]
        valid = [run for run in successful if run["output"]["recomputed"].get("valid")]
        times = [float(run["wall_seconds"]) for run in valid]
        objs = [float(run["output"]["recomputed"]["obj"]) for run in valid]
        reasons: list[str] = []
        if len(valid) != len(runs):
            reasons.append(
                f"only {len(valid)}/{len(runs)} runs returned finite verified models"
            )
        if any(run.get("memory", {}).get("process_paging_observed") for run in runs):
            reasons.append(
                "the benchmark process paged during at least one measured run"
            )
        if valid:
            for run in valid:
                returned = run["output"].get("obj")
                recomputed = run["output"]["recomputed"]["obj"]
                tolerance = quality_rtol * max(1.0, abs(float(recomputed)))
                if returned is None or not math.isfinite(float(returned)):
                    reasons.append("a returned objective is missing or non-finite")
                elif abs(float(returned) - float(recomputed)) > tolerance:
                    reasons.append(
                        "a returned objective fails independent recomputation"
                    )
                for gap_name in ("gap_cert", "gap_heur"):
                    gap = run["output"].get(gap_name)
                    if gap is not None and not math.isfinite(float(gap)):
                        reasons.append(f"a reported {gap_name} is non-finite")
        summaries[backend.name] = {
            "backend": asdict(backend),
            "wall_seconds": _quartiles(times) if times else None,
            "recomputed_objective": _quartiles(objs) if objs else None,
            "system_swap_io_warning": any(
                run.get("memory", {}).get("system_swap_io_observed") for run in runs
            ),
            "internal_valid": not reasons,
            "internal_failure_reasons": sorted(set(reasons)),
        }

    baseline_name = backends[0].name
    baseline = summaries[baseline_name]
    comparisons: list[dict[str, Any]] = []
    for backend in backends:
        name = backend.name
        summary = summaries[name]
        reasons: list[str] = []
        quality_pass = bool(baseline["internal_valid"] and summary["internal_valid"])
        if not baseline["internal_valid"]:
            reasons.append("baseline failed finite-model or recomputation checks")
        if not summary["internal_valid"]:
            reasons.extend(summary["internal_failure_reasons"])
        if (
            baseline["recomputed_objective"] is None
            or summary["recomputed_objective"] is None
        ):
            quality_pass = False
            reasons.append("missing verified objective samples")
            baseline_obj = candidate_obj = tolerance = None
        else:
            baseline_obj = float(baseline["recomputed_objective"]["median"])
            candidate_obj = float(summary["recomputed_objective"]["median"])
            tolerance = quality_rtol * max(1.0, abs(baseline_obj))
            candidate_runs = [
                run
                for run in measured
                if run["backend"] == name
                and run["status"] == "ok"
                and run["output"]["recomputed"].get("valid")
            ]
            if any(
                float(run["output"]["recomputed"]["obj"]) > baseline_obj + tolerance
                for run in candidate_runs
            ):
                quality_pass = False
                reasons.append(
                    "at least one candidate objective is worse than baseline tolerance"
                )

        if baseline["wall_seconds"] is None or summary["wall_seconds"] is None:
            ratio = speedup = None
            regression_pass = performance_pass = False
            reasons.append("missing valid timing samples")
        else:
            base_time = float(baseline["wall_seconds"]["median"])
            candidate_time = float(summary["wall_seconds"]["median"])
            ratio = candidate_time / base_time if base_time else None
            speedup = base_time / candidate_time if candidate_time else None
            regression_pass = bool(ratio is not None and ratio <= 1.10)
            performance_pass = bool(
                quality_pass and ratio is not None and ratio <= 0.80
            )

        if name == baseline_name and quality_pass:
            verdict = "baseline"
            reasons.append("reference backend; no speedup decision")
        elif not quality_pass:
            verdict = "fail_quality"
        elif performance_pass:
            verdict = "pass_end_to_end_speed_threshold"
        elif not regression_pass:
            verdict = "fail_runtime_regression_threshold"
            reasons.append("median time exceeds baseline by more than 10%")
        else:
            verdict = "quality_pass_but_no_20_percent_speedup"
            reasons.append("median time does not improve baseline by at least 20%")
        comparisons.append(
            {
                "baseline": baseline_name,
                "candidate": name,
                "baseline_median_objective": baseline_obj,
                "candidate_median_objective": candidate_obj,
                "objective_tolerance": tolerance,
                "quality_pass": quality_pass,
                "median_time_ratio_candidate_over_baseline": ratio,
                "speedup_baseline_over_candidate": speedup,
                "runtime_regression_pass": regression_pass,
                "performance_threshold_pass": performance_pass,
                "verdict": verdict,
                "reasons": sorted(set(reasons)),
            }
        )
    return summaries, comparisons


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _cpu_metadata() -> dict[str, Any]:
    model = None
    physical_pairs: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    try:
        blocks = Path("/proc/cpuinfo").read_text().strip().split("\n\n")
        for block in blocks:
            current = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    current[key.strip()] = value.strip()
            model = model or current.get("model name") or current.get("Hardware")
            if "physical id" in current and "core id" in current:
                physical_pairs.add((current["physical id"], current["core id"]))
    except OSError:
        pass
    affinity = None
    with contextlib.suppress(AttributeError, OSError):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "model": model,
        "logical_count": os.cpu_count(),
        "physical_core_count_from_proc": len(physical_pairs) or None,
        "process_affinity_logical_cpus": affinity,
    }


def _accelerator_metadata() -> dict[str, Any]:
    torch = _torch_module()
    if torch is None:
        return {"torch_available": False, "devices": []}
    result: dict[str, Any] = {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "hip_runtime": getattr(torch.version, "hip", None),
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "multi_processor_count": getattr(
                        props, "multi_processor_count", None
                    ),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return result


def _metadata() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "torch", "cvxpy", "scikit-learn", "matplotlib"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    env_names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    )
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": {"version": sys.version},
        "cpu": _cpu_metadata(),
        "accelerators": _accelerator_metadata(),
        "memory": _read_meminfo(),
        "process_peak_rss_before_benchmark_bytes": int(ru.ru_maxrss) * 1024,
        "software_packages": packages,
        "thread_environment": {name: os.environ.get(name) for name in env_names},
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _clean_json(payload), handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _default_output(root: Path) -> Path:
    stamp = _datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return root / "results" / f"controlled_benchmark_{stamp}.json"


def _self_test() -> None:
    orders = _balanced_orders(["a", "b", "c"], 5, 7)
    assert _position_balance(orders)["balanced_within_one"]
    stats = _quartiles([1.0, 2.0, 3.0])
    assert stats["median"] == 2.0 and stats["iqr"] == 1.0
    case = _parse_case(
        "fast=torch-cpu;ascent=damped;order=decrease;search=device;"
        "reuse=on;reuse-cap=12345",
        "auto",
    )
    assert case.name == "fast" and case.ascent_trial == "damped"
    assert case.candidate_order == "decrease" and case.data_dtype == "float64"
    assert case.search_backend == "device"
    assert case.retain_finalist_features and case.retained_feature_max_bytes == 12345
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "out.json"
        _atomic_json(path, {"finite": 1.0, "nonfinite": float("nan")})
        assert json.loads(path.read_text()) == {"finite": 1.0, "nonfinite": None}
    X = np.array([[1.0], [-1.0]])
    check = _recompute_objective(
        X, np.array([1.0, 0.0]), 0.1, np.array([[1.0]]), np.array([1.0])
    )
    assert check["valid"] and abs(check["obj"] - 0.1) < 1e-14


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        action="append",
        default=None,
        help="repeat for balanced comparison; first is baseline",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=(
            "repeat objective-preserving A/B cases as "
            "LABEL=BACKEND[;ascent=both|damped|fixed|packed]"
            "[;order=correlation|decrease]"
            "[;search=host|device|auto][;reuse=on|off]"
            "[;reuse-cap=BYTES]; cannot be mixed with --backend"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0, help="dataset seed")
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=1729)
    parser.add_argument(
        "--input-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--data-dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--eps-rel", type=float, default=1e-4)
    parser.add_argument("--eps-heur", type=float, default=1e-3)
    parser.add_argument("--price-tol", type=float, default=1e-3)
    parser.add_argument("--add-per-round", type=int, default=4)
    parser.add_argument("--max-cols", type=int, default=64)
    parser.add_argument("--certify-every", type=int, default=0)
    parser.add_argument("--cert-steps", type=int, default=2)
    parser.add_argument("--gated-every", type=int, default=0)
    parser.add_argument("--gated-iters", type=int, default=5)
    parser.add_argument("--sketch-m", default="0", help="integer row count or 'auto'")
    parser.add_argument("--stall-rtol", type=float, default=1e-7)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--final-verify",
        action="store_true",
        help="required for evidence; omitted by cheap default smoke",
    )
    parser.add_argument("--quality-rtol", type=float, default=1e-6)
    parser.add_argument(
        "--large",
        action="store_true",
        help=f"required when n > {LARGE_ROW_THRESHOLD:,}",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="test harness utilities and exit without fitting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print("controlled benchmark self-test: OK")
        return 0
    if args.repeats < 3 or args.warmups < 1:
        raise SystemExit("measurement protocol requires >=3 repeats and >=1 warm-up")
    if min(args.n, args.d, args.k) < 1 or args.k > args.d:
        raise SystemExit("require positive n,d,k and k <= d")
    if args.n > LARGE_ROW_THRESHOLD and not args.large:
        raise SystemExit(f"n={args.n} is a large run; repeat with --large")
    try:
        sketch_m: str | int = "auto" if args.sketch_m == "auto" else int(args.sketch_m)
    except ValueError as exc:
        raise SystemExit("--sketch-m must be an integer or auto") from exc
    if args.case and args.backend:
        raise SystemExit("use either --case or --backend, not both")
    try:
        if args.case:
            backends = [_parse_case(text, args.data_dtype) for text in args.case]
        else:
            backend_texts = args.backend or ["cpu"]
            backends = [_parse_backend(text, args.data_dtype) for text in backend_texts]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    names = [backend.name for backend in backends]
    if len(set(names)) != len(names):
        raise SystemExit("backend names must be unique")

    root = Path(__file__).resolve().parents[1]
    output = args.output or _default_output(root)
    dataset_start = time.perf_counter()
    planted = make_planted(
        args.n, args.d, args.k, np.random.default_rng(args.seed), noise=args.noise
    )
    input_dtype = np.float32 if args.input_dtype == "float32" else np.float64
    X = np.ascontiguousarray(planted["X"], dtype=input_dtype)
    y = np.ascontiguousarray(planted["y"], dtype=np.float64)
    dataset_seconds = time.perf_counter() - dataset_start
    dataset = {
        "kind": "planted",
        "n": args.n,
        "d": args.d,
        "k": args.k,
        "lambda": args.lam,
        "noise": args.noise,
        "seed": args.seed,
        "input_dtype": str(X.dtype),
        "X_sha256": _array_digest(X),
        "y_sha256": _array_digest(y),
        "generation_seconds": dataset_seconds,
    }
    solver_config = {
        "max_iter": args.max_iter,
        "eps_rel": args.eps_rel,
        "eps_heur": args.eps_heur,
        "price_tol": args.price_tol,
        "add_per_round": args.add_per_round,
        "max_cols": args.max_cols,
        "certify_every": args.certify_every,
        "cert_steps": args.cert_steps,
        "gated_every": args.gated_every,
        "gated_iters": args.gated_iters,
        "sketch_m": sketch_m,
        "stall_rtol": args.stall_rtol,
        "patience": args.patience,
        "final_verify": args.final_verify,
        "verbose": False,
    }

    warmup_orders = _balanced_orders(names, args.warmups, args.order_seed + 1)
    measured_orders = _balanced_orders(names, args.repeats, args.order_seed)
    by_name = {backend.name: backend for backend in backends}
    warmups: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    sequence = 0
    for round_index, order in enumerate(warmup_orders):
        for name in order:
            warmups.append(
                _run_once(
                    by_name[name],
                    X,
                    y,
                    args.lam,
                    solver_config,
                    args.solver_seed,
                    "warmup",
                    round_index,
                    sequence,
                )
            )
            sequence += 1
    for round_index, order in enumerate(measured_orders):
        for name in order:
            measured.append(
                _run_once(
                    by_name[name],
                    X,
                    y,
                    args.lam,
                    solver_config,
                    args.solver_seed,
                    "measured",
                    round_index,
                    sequence,
                )
            )
            sequence += 1

    summaries, comparisons = _summaries(backends, measured, args.quality_rtol)
    protocol_eligible = bool(
        args.final_verify and args.repeats >= 3 and args.warmups >= 1
    )
    summaries_valid = all(summary["internal_valid"] for summary in summaries.values())
    evidence_eligible = bool(protocol_eligible and summaries_valid)
    ineligibility = []
    if not args.final_verify:
        ineligibility.append("--final-verify was not enabled")
    for name, summary in summaries.items():
        if not summary["internal_valid"]:
            ineligibility.extend(
                f"{name}: {reason}" for reason in summary["internal_failure_reasons"]
            )
    report = {
        "schema": "relu_chambers.controlled_benchmark",
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "clock": "time.perf_counter",
            "warmups_per_backend": args.warmups,
            "measured_repeats_per_backend": args.repeats,
            "primary_metric": "median end-to-end wall_seconds after warm-up",
            "timed_scope": "solve_rcg engine setup/transfers/solve plus independent objective recomputation",
            "device_timing": "synchronize requested accelerators immediately before and after solve_rcg",
            "quality_contract": "each candidate run <= baseline median objective + quality_rtol*max(1,abs(baseline))",
            "quality_rtol": args.quality_rtol,
            "warmup_orders": warmup_orders,
            "measured_orders": measured_orders,
            "measured_order_balance": _position_balance(measured_orders),
            "evidence_eligible": evidence_eligible,
            "evidence_ineligibility_reasons": sorted(set(ineligibility)),
            "known_measurement_limits": [
                "warm-up excludes first-use compilation from the primary repeated median",
                "host RSS peak is sampled at 10 ms and may miss shorter allocator peaks",
                "process paging uses major-fault and VmSwap deltas; system-wide swap I/O is recorded separately as an interference warning",
                "solve_rcg does not expose transfer bytes, solver synchronization counts, full-data pass counts, or finer pricing subphases",
                "logical CPU row shards are not evidence of multiple physical CPUs or sockets",
            ],
        },
        "metadata": _metadata(),
        "dataset": dataset,
        "solver": {
            "name": "solve_rcg",
            "rng_seed": args.solver_seed,
            "config": solver_config,
        },
        "backends": [asdict(backend) for backend in backends],
        "warmup_runs": warmups,
        "measured_runs": measured,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    _atomic_json(output, report)
    print(f"wrote {output}")
    for comparison in comparisons:
        summary = summaries[comparison["candidate"]]["wall_seconds"]
        timing = (
            "no valid time"
            if summary is None
            else (f"median={summary['median']:.6f}s IQR={summary['iqr']:.6f}s")
        )
        print(f"{comparison['candidate']}: {timing}; {comparison['verdict']}")
    return 0 if all(item["quality_pass"] for item in comparisons) else 2


if __name__ == "__main__":
    raise SystemExit(main())
