"""Equivalence test: sharded engine (1/2/4 shards) vs single-device reference.

python -m tests.test_data_engine            # numpy backend
python -m tests.test_data_engine torch      # torch backend (cpu devices)
"""

from __future__ import annotations

import sys

import numpy as np

from relu_chambers.data_engine import ShardedData


def run(backend="numpy", devices=("cpu", "cpu"), dtype="float64"):
    rng = np.random.default_rng(0)
    n, d, P = 4001, 7, 5
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    U = rng.standard_normal((d, P))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    beta = rng.standard_normal(P)

    ref_A = np.maximum(X @ U, 0.0)
    ref_r = ref_A @ beta - y
    ref_screen = ref_A.T @ ref_r
    ref_sq = (ref_A**2).sum(0)
    ref_gram = ref_A[:, [0, 2, 4]].T @ ref_A[:, [0, 2, 4]]
    nu = -ref_r
    ref_rescore = ref_A.T @ nu
    ref_rescore_sq = (ref_A**2).sum(0)
    ref_spec = X.T @ ((nu**2)[:, None] * X)
    ref_xtv = X.T @ nu
    w = np.maximum(nu, 0.0)
    Z = np.eye(d) * 2.0
    B = w[:, None] * X
    c = np.sqrt(np.clip(np.einsum("ij,ij->i", B @ Z, B), 1e-300, None))
    b0 = X.T @ w
    c0 = float(np.sqrt(b0 @ Z @ b0))
    ref_csum = c.sum() + c0
    ref_M = (B / c[:, None]).T @ B + np.outer(b0, b0) / c0

    sd = ShardedData(
        X, y, devices=devices, backend=backend, dtype=dtype, chunk=1024
    )  # small chunk: exercise chunk loops
    A_sh = sd.build_atoms(U)
    r_sh = sd.residual(A_sh, beta)
    nu_sh = [-v for v in r_sh]

    def err(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.abs(b))))

    checks = {
        "screen": err(sd.screen(A_sh, r_sh), ref_screen),
        "sqnorms": err(sd.sqnorms(A_sh), ref_sq),
        "gram": err(sd.gram(A_sh, [0, 2, 4]), ref_gram),
        "rescore": err(sd.rescore(U, nu_sh), ref_rescore),
        "spectral": err(sd.spectral(nu_sh), ref_spec),
        "xtv": err(sd.xtv(nu_sh), ref_xtv),
        "gather": err(sd.gather_vec(nu_sh), nu),
    }
    score, score_sq = sd.rescore_stats(U, nu_sh)
    checks["rescore_stats"] = max(
        err(score, ref_rescore), err(score_sq, ref_rescore_sq)
    )
    csum, Mv = sd.cert_validate(w, Z, b0)
    checks["cert_csum"] = abs(csum - ref_csum) / (1 + abs(ref_csum))
    checks["cert_M"] = err(Mv, ref_M)
    # chamber keys identical across shardings
    keys = sd.chamber_keys(U)
    sd1 = ShardedData(
        X, y, devices=("cpu",), backend="numpy", dtype="float64", chunk=1024
    )
    checks["chamber_keys"] = 0.0 if keys == sd1.chamber_keys(U) else 1.0
    tolmap = {"spectral": 3e-4}  # storage-dtype kernel (seeds only)
    base_tol = 1e-10 if dtype == "float64" else 2e-5
    ok = all(v < tolmap.get(k2, base_tol) for k2, v in checks.items())
    print(
        f"[{backend} x{len(devices)} {dtype}] "
        + " ".join(f"{k2}={v:.1e}" for k2, v in checks.items())
        + ("  OK" if ok else "  FAIL")
    )
    return ok


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    ok = True
    for k in (1, 2, 4):
        ok &= run(backend, devices=("cpu",) * k, dtype="float64")
        ok &= run(backend, devices=("cpu",) * k, dtype="float32")
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
