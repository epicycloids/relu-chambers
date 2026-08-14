"""Array-backend helpers for NumPy and optional CuPy execution.

The reference solvers express large-data operations as dense products with
X or X^T, so they can run on whichever backend holds X. This module provides
a small compatibility layer:

    xp = get_xp("gpu")          # cupy if available, else numpy
    Xg = to_device(X, "gpu")    # move once; keep it resident

All numerical kernels below take an explicit ``xp`` (the array module) and an X
already on the target device, and use only ``xp`` calls, so the identical code
runs on CPU and GPU.  This is deliberate: the algorithm's arithmetic intensity
(matmuls) is what makes the GPU/parallel story work, and writing device-agnostic
kernels keeps the CPU and GPU paths aligned at the algorithmic level.
"""

from __future__ import annotations

import numpy as np

_CUPY = None


def _try_cupy():
    global _CUPY
    if _CUPY is None:
        try:
            import cupy as cp  # type: ignore

            cp.cuda.Device(0).use()
            _CUPY = cp
        except Exception:
            _CUPY = False
    return _CUPY


def get_xp(device="cpu"):
    """Return the array module for ``device`` ('cpu' -> numpy, 'gpu' -> cupy)."""
    if device == "gpu":
        cp = _try_cupy()
        if cp is False:
            raise RuntimeError("GPU requested but CuPy/CUDA unavailable")
        return cp
    return np


def gpu_available():
    return _try_cupy() is not False


def to_device(a, device):
    xp = get_xp(device)
    if xp is np:
        return np.asarray(a)
    return xp.asarray(a)


def to_cpu(a):
    cp = _try_cupy()
    if cp is not False and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def asfloat(a, xp, dtype):
    return xp.asarray(a, dtype=dtype)


# --- small-matrix linear algebra: always CPU + float64 ---------------------- #
# Spectral work on the d x d / r x r reductions is tiny but precision-critical;
# we do it on the CPU in float64 (LAPACK) even when the big matmuls run on the GPU
# in float32.  This is both a stability choice (the conditioning of the master and
# of the certificate lives in these small factorisations) and a portability one
# (it sidesteps the GPU's cuSOLVER entirely).
def _cpu64(a):
    return to_cpu(a).astype(np.float64)


def eigvalsh_small(M):
    """Ascending eigenvalues of a small symmetric matrix (numpy float64)."""
    A = _cpu64(M)
    return np.linalg.eigvalsh(0.5 * (A + A.T))


def eigh_small(M):
    """(w, Q) of a small symmetric matrix in float64 (numpy)."""
    A = _cpu64(M)
    return np.linalg.eigh(0.5 * (A + A.T))


def psd_project_small(M, xp, dtype):
    """Project a small symmetric matrix onto the PSD cone (CPU float64), return on
    the device with ``dtype``."""
    w, Q = eigh_small(M)
    w = np.clip(w, 0.0, None)
    Z = (Q * w) @ Q.T
    return xp.asarray(Z, dtype=dtype)
