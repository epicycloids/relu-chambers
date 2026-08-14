"""Row-sharded NumPy/Torch data engine for the practical RCG solver.

The solver touches the data only through the kernels below; every one is a
row-wise map followed by a small reduction:

    kernel            map (per shard, O(rows * ...))      reduction (host)
    ---------------   ---------------------------------   -----------------
    build_atoms       A_i = relu(X_i U)                   none (stays sharded)
    screen            A_i^T r_i                            sum -> (P,)   f64
    sqnorms           colsum(A_i^2)                        sum -> (P,)   f64
    gram              A_i[:,S]^T A_i[:,S]                  sum -> (k,k)  f64
    residual          A_i beta - y_i                       none (stays sharded)
    rescore           nu_i^T relu(X_i U)                   sum -> (R,)   f64
    spectral          X_i^T diag(nu_i^2) X_i               sum -> (d,d)  f64
    xtv               X_i^T v_i                            sum -> (d,)   f64
    cert_validate     c_i = sqrt(rowsq(B_i Z)), M_i        sum c, sum M  f64
    chamber_keys      hash(packbits(X_i U >= 0))           digest concat

These local reductions are small, but they are not the whole solver's data
movement.  The outer loop gathers an O(n) residual on the host and some
chamber paths can move O(nP) masks.  No sample-count-independent communication
or physical multi-device scaling claim is made.  Kernels use row chunks so
their working arrays need not span an entire shard.

Backends:
    backend="numpy"  shards are CPU blocks; serial or thread-pool execution is
                     available on one shared-memory host.
    backend="torch"  shards live on torch devices ("cuda:0", "cuda:1", ...;
                     ROCm presents as "cuda").  Operations are dispatched per
                     shard and host reductions synchronize their results.
`weights` sets the shard row fractions for heterogeneous devices.

Search-phase computations (pricing ascent, gated FISTA, certificate ascent)
run on HOST over an importance row sketch obtained via gather_rows(); only the
full-data numerical validation/rescoring passes use the engine by default.
Pricing also has an opt-in single-device sketch path.  Backend speedups must be
measured on the target hardware.
"""

from __future__ import annotations

import hashlib

import numpy as np

try:
    import torch  # type: ignore

    _TORCH = True
except Exception:
    _TORCH = False

CHUNK = 1 << 16


def _split(n, k, weights=None):
    if weights is None:
        edges = np.linspace(0, n, k + 1)
    else:
        w = np.asarray(weights, float)
        edges = np.concatenate([[0.0], np.cumsum(w) / w.sum() * n])
    edges = edges.astype(int)
    edges[-1] = n
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


class ShardedData:
    def __init__(
        self,
        X,
        y,
        devices=("cpu",),
        backend="numpy",
        dtype="float64",
        threads=False,
        weights=None,
        chunk=CHUNK,
    ):
        X = np.ascontiguousarray(X)
        y = np.asarray(y, dtype=np.float64)
        self.n, self.d = X.shape
        self.backend = backend
        self.devices = list(devices)
        self.threads = threads
        self.chunk = chunk
        self.slices = _split(self.n, len(self.devices), weights)
        if backend == "numpy":
            self._dt = np.float32 if dtype == "float32" else np.float64
            # shards are VIEWS of the host array when dtypes already match
            self.Xs = [X[s:e].astype(self._dt, copy=False) for s, e in self.slices]
            self.ys = [y[s:e] for s, e in self.slices]
        elif backend == "torch":
            if not _TORCH:
                raise RuntimeError("torch backend requested but torch missing")
            self._dt = torch.float32 if dtype == "float32" else torch.float64
            self.Xs = [
                torch.as_tensor(X[s:e], dtype=self._dt).to(dev)
                for (s, e), dev in zip(self.slices, self.devices)
            ]
            self.ys = [
                torch.as_tensor(y[s:e], dtype=torch.float64).to(dev)
                for (s, e), dev in zip(self.slices, self.devices)
            ]
        else:
            raise ValueError(backend)
        self._Xhost = X  # host copy kept for sketch gathers
        self._yhost = y

    # ---------------- helpers ----------------
    def _to_dev(self, a, i, f64=False):
        if self.backend == "numpy":
            return np.asarray(a, dtype=np.float64 if f64 else self._dt)
        dt = torch.float64 if f64 else self._dt
        return torch.as_tensor(np.asarray(a), dtype=dt).to(self.devices[i])

    def _host64(self, a):
        if self.backend == "numpy":
            return np.asarray(a, dtype=np.float64)
        return a.detach().to("cpu", torch.float64).numpy()

    def _map(self, fn):
        idx = range(len(self.Xs))
        if self.backend == "numpy" and self.threads and len(self.Xs) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(self.Xs)) as ex:
                return list(ex.map(fn, idx))
        return [fn(i) for i in idx]

    def _relu(self, Z):
        """Apply ReLU in place to a newly formed work array/tensor."""
        if self.backend == "numpy":
            np.maximum(Z, 0.0, out=Z)
            return Z
        return Z.clamp_min_(0.0)

    def _f64(self, a):
        if self.backend == "numpy":
            return a.astype(np.float64, copy=False)
        return a.to(torch.float64)

    def _zeros64(self, shape, i):
        if self.backend == "numpy":
            return np.zeros(shape, dtype=np.float64)
        return torch.zeros(shape, dtype=torch.float64, device=self.devices[i])

    def synchronize(self):
        if self.backend == "torch" and any("cuda" in str(d) for d in self.devices):
            torch.cuda.synchronize()

    # ---------------- sketch support (host) ----------------
    def gather_rows(self, idx):
        """Host (float64) rows of X and y for sketch-phase computations."""
        return self._Xhost[idx], self._yhost[idx]

    def gather_vec(self, v_sh):
        """Concatenate a sharded vector to host float64."""
        return np.concatenate([self._host64(v) for v in v_sh])

    def scatter_vec(self, v):
        """Host vector -> sharded float64 device pieces."""
        out = []
        for i, (s, e) in enumerate(self.slices):
            out.append(self._to_dev(np.asarray(v[s:e], np.float64), i, f64=True))
        return out

    # ---------------- kernels ----------------
    def build_atoms(self, U):
        def k(i):
            Ud = self._to_dev(U, i)
            return self._relu(self.Xs[i] @ Ud)

        return self._map(k)

    def feature_buffer_bytes(self, n_cols):
        """Payload bytes for ``n_cols`` full-height stored feature columns."""
        n_cols = int(n_cols)
        if n_cols < 0:
            raise ValueError("n_cols must be nonnegative")
        if self.backend == "numpy":
            itemsize = np.dtype(self._dt).itemsize
        else:
            itemsize = 4 if self._dt == torch.float32 else 8
        return int(self.n * n_cols * itemsize)

    def append_atoms(self, A_sh, U):
        Anew = self.build_atoms(U)
        if self.backend == "numpy":
            return [
                np.concatenate([A_sh[i], Anew[i]], axis=1) for i in range(len(A_sh))
            ]
        return [torch.cat([A_sh[i], Anew[i]], dim=1) for i in range(len(A_sh))]

    def slice_cols(self, A_sh, idx):
        idxa = np.asarray(idx)
        if self.backend == "numpy":
            return [A[:, idxa] for A in A_sh]
        return [A[:, torch.as_tensor(idxa).to(A.device)] for A in A_sh]

    def screen(self, A_sh, r_sh):
        def k(i):
            A, r = A_sh[i], r_sh[i]
            out = self._zeros64(A.shape[1], i)
            for s in range(0, A.shape[0], self.chunk):
                e = min(A.shape[0], s + self.chunk)
                out += self._f64(A[s:e]).T @ self._f64(r[s:e])
            return out

        return sum(self._host64(v) for v in self._map(k))

    def sqnorms(self, A_sh):
        def k(i):
            A = A_sh[i]
            out = self._zeros64(A.shape[1], i)
            for s in range(0, A.shape[0], self.chunk):
                e = min(A.shape[0], s + self.chunk)
                Ac = self._f64(A[s:e])
                out += (Ac * Ac).sum(0)
            return out

        return sum(self._host64(v) for v in self._map(k))

    def gram(self, A_sh, idx):
        sub = self.slice_cols(A_sh, idx)

        def k(i):
            A = sub[i]
            out = self._zeros64((A.shape[1], A.shape[1]), i)
            for s in range(0, A.shape[0], self.chunk):
                e = min(A.shape[0], s + self.chunk)
                Ac = self._f64(A[s:e])
                out += Ac.T @ Ac
            return out

        return sum(self._host64(v) for v in self._map(k))

    def residual(self, A_sh, beta):
        """Sharded float64 r = A beta - y."""

        def k(i):
            A = A_sh[i]
            out = self._zeros64(A.shape[0], i)
            b = self._to_dev(beta, i, f64=True)
            for s in range(0, A.shape[0], self.chunk):
                e = min(A.shape[0], s + self.chunk)
                out[s:e] = self._f64(A[s:e]) @ b
            return out - self.ys[i]

        return self._map(k)

    def rescore(self, U, nu_sh):
        def k(i):
            Ud = self._to_dev(U, i)
            out = self._zeros64(U.shape[1], i)
            X = self.Xs[i]
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                Ac = self._f64(self._relu(X[s:e] @ Ud))
                out += Ac.T @ self._f64(nu_sh[i][s:e])
            return out

        return sum(self._host64(v) for v in self._map(k))

    def rescore_stats(self, U, nu_sh):
        """Exact candidate correlations and squared feature norms in one pass."""

        def k(i):
            Ud = self._to_dev(U, i)
            corr = self._zeros64(U.shape[1], i)
            sqnorm = self._zeros64(U.shape[1], i)
            X = self.Xs[i]
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                Ac = self._f64(self._relu(X[s:e] @ Ud))
                corr += Ac.T @ self._f64(nu_sh[i][s:e])
                sqnorm += (Ac * Ac).sum(0)
            return corr, sqnorm

        parts = self._map(k)
        corr = sum(self._host64(part[0]) for part in parts)
        sqnorm = sum(self._host64(part[1]) for part in parts)
        return corr, sqnorm

    def rescore_finalists(
        self, U, nu_sh, retain_features=False, retained_feature_max_bytes=None
    ):
        """Rescore finalists and optionally retain the features just formed.

        The opt-in retained path materializes ``relu(X @ U)`` once in the
        engine storage dtype and accumulates correlation, squared norm, and
        ``A.T @ y`` together.  The latter two statistics let the restricted
        master append the retained columns without another full-data scan.
        ``retained_feature_max_bytes`` caps feature payload bytes; when the
        requested matrix does not fit, this method returns the ordinary
        chunked rescore and no feature buffer.
        """
        U = np.asarray(U, np.float64)
        if U.ndim != 2 or U.shape[0] != self.d:
            raise ValueError(f"U must have shape ({self.d}, p)")
        if retained_feature_max_bytes is not None:
            retained_feature_max_bytes = int(retained_feature_max_bytes)
            if retained_feature_max_bytes < 0:
                raise ValueError("retained_feature_max_bytes must be nonnegative")
        payload_bytes = self.feature_buffer_bytes(U.shape[1])
        fits = (
            retained_feature_max_bytes is None
            or payload_bytes <= retained_feature_max_bytes
        )
        retained = bool(retain_features and fits)
        if not retained:
            corr, sqnorm = self.rescore_stats(U, nu_sh)
            return {
                "corr": corr,
                "sqnorm": sqnorm,
                "aty": None,
                "features": None,
                "retained": False,
                "feature_payload_bytes": 0,
                "requested_feature_payload_bytes": payload_bytes,
                "fallback_reason": (
                    "disabled" if not retain_features else "cap_exceeded"
                ),
            }

        A_sh = self.build_atoms(U)

        def k(i):
            A = A_sh[i]
            width = A.shape[1]
            corr = self._zeros64(width, i)
            sqnorm = self._zeros64(width, i)
            aty = self._zeros64(width, i)
            for s in range(0, A.shape[0], self.chunk):
                e = min(A.shape[0], s + self.chunk)
                Ac = self._f64(A[s:e])
                corr += Ac.T @ self._f64(nu_sh[i][s:e])
                sqnorm += (Ac * Ac).sum(0)
                aty += Ac.T @ self._f64(self.ys[i][s:e])
            return corr, sqnorm, aty

        parts = self._map(k)
        corr = sum(self._host64(part[0]) for part in parts)
        sqnorm = sum(self._host64(part[1]) for part in parts)
        aty = sum(self._host64(part[2]) for part in parts)
        return {
            "corr": corr,
            "sqnorm": sqnorm,
            "aty": aty,
            "features": A_sh,
            "retained": True,
            "feature_payload_bytes": payload_bytes,
            "requested_feature_payload_bytes": payload_bytes,
            "fallback_reason": None,
        }

    def spectral(self, nu_sh):
        """X^T diag(nu^2) X (float64 d x d) -- computed in storage dtype with
        float64 cross-chunk accumulation (seed quality does not need f64)."""

        def k(i):
            X = self.Xs[i]
            out = self._zeros64((self.d, self.d), i)
            w = nu_sh[i]
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                Xc = X[s:e]
                if self.backend == "numpy":
                    wc = (np.asarray(w[s:e], self._dt) ** 2)[:, None]
                    out += self._f64(Xc.T @ (wc * Xc))
                else:
                    wc = (w[s:e].to(self._dt) ** 2)[:, None]
                    out += (Xc.T @ (wc * Xc)).to(torch.float64)
            return out

        return sum(self._host64(v) for v in self._map(k))

    def xtv(self, v_sh):
        def k(i):
            X = self.Xs[i]
            out = self._zeros64(self.d, i)
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                out += self._f64(X[s:e]).T @ self._f64(v_sh[i][s:e])
            return out

        return sum(self._host64(v) for v in self._map(k))

    def cert_validate(self, w_host, Z, b0):
        """sum_i c_i and M = sum_i b_i b_i^T / c_i for rows B = diag(w) X plus
        the augmented row b0 (host).  w_host: host nonneg weights (n,)."""
        Zh = np.asarray(Z, np.float64)
        w_sh = self.scatter_vec(w_host)

        def k(i):
            X = self.Xs[i]
            csum = self._zeros64(1, i)
            M = self._zeros64((self.d, self.d), i)
            Zt = self._to_dev(Zh, i, f64=True)
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                B = self._f64(X[s:e]) * w_sh[i][s:e][:, None]
                q = ((B @ Zt) * B).sum(1)
                if self.backend == "numpy":
                    c = np.sqrt(np.clip(q, 1e-300, None))
                else:
                    c = torch.sqrt(torch.clamp(q, min=1e-300))
                csum += c.sum()
                M += (B / c[:, None]).T @ B
            return csum, M

        parts = self._map(k)
        csum = sum(float(self._host64(p[0]).reshape(-1)[0]) for p in parts)
        M = sum(self._host64(p[1]) for p in parts)
        b0 = np.asarray(b0, np.float64)
        c0 = float(np.sqrt(max(b0 @ Zh @ b0, 1e-300)))
        return csum + c0, M + np.outer(b0, b0) / c0

    def chamber_keys(self, U):
        """Per-column hash of the activation pattern 1{XU>=0} (order: shards in
        row order, chunks in row order -> deterministic)."""
        P = np.asarray(U).shape[1]
        hs = [hashlib.blake2b(digest_size=16) for _ in range(P)]
        for i in range(len(self.Xs)):
            Ud = self._to_dev(U, i)
            X = self.Xs[i]
            for s in range(0, X.shape[0], self.chunk):
                e = min(X.shape[0], s + self.chunk)
                Z = X[s:e] @ Ud
                if self.backend == "numpy":
                    Mc = np.asarray(Z >= 0)
                else:
                    Mc = (Z >= 0).to("cpu").numpy()
                bits = np.packbits(Mc, axis=0)
                for j in range(P):
                    hs[j].update(bits[:, j].tobytes())
        return [h.digest() for h in hs]
