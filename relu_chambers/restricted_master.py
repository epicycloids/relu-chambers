"""Numerical atom-Lasso restricted master for the practical RCG solver.

Primal active-set method for  min 1/2||A beta - y||^2 + lam ||beta||_1  with:

  * incremental float64 caches: Gram COLUMNS are computed once per atom (the
    first time the KKT system touches it); A^T y and atom norms once per
    appended atom.  The per-step reduced KKT solve is a tiny host Cholesky.
  * NORMALIZED-column formulation.  For nonzero stored columns the scaling is
    algebraically equivalent to a weighted Lasso, with
    a_j = ||a_j|| a~_j and lam_j = lam/||a_j||.
  * atoms stored sharded across devices in the storage dtype (float32 on GPU);
    KKT quantities are reduced in float64.  Storage precision, ridge
    regularization, tolerances, cycling guards, and finite loops still make the
    returned solution numerical rather than exact.

Screening and any newly requested Gram columns use full-row engine passes."""

from __future__ import annotations

import numpy as np


class AtomLasso:
    """Working set of ReLU atoms plus warm-started numerical Lasso solves."""

    def __init__(self, engine, ridge=1e-10):
        self.eng = engine
        self.ridge = ridge
        self.A = None  # sharded atom blocks
        self.U = np.zeros((engine.d, 0))  # host float64 unit directions
        self.Aty = np.zeros(0)
        self.norms = np.zeros(0)
        self.G = np.zeros((0, 0))
        self.known = np.zeros(0, dtype=bool)
        self.S = []
        self.beta = np.zeros(0)

    @property
    def P(self):
        return self.U.shape[1]

    def set_atoms(self, U, A_sh=None, aty=None, sqnorms=None):
        U = np.asarray(U, np.float64)
        self.A = A_sh if A_sh is not None else self.eng.build_atoms(U)
        self.U = U
        self._validate_prebuilt(self.A, U.shape[1])
        self.Aty = (
            self.eng.screen(self.A, self.eng.ys)
            if aty is None
            else self._validate_stats(aty, U.shape[1], "aty")
        )
        sq = (
            self.eng.sqnorms(self.A)
            if sqnorms is None
            else self._validate_stats(sqnorms, U.shape[1], "sqnorms")
        )
        self.norms = np.sqrt(np.maximum(sq, 1e-300))
        P = U.shape[1]
        self.G = np.zeros((P, P))
        self.known = np.zeros(P, dtype=bool)
        self.S = []
        self.beta = np.zeros(P)

    def append_atoms(self, Unew, A_sh=None, aty=None, sqnorms=None):
        """Append directions, optionally reusing aligned full-data features.

        Omitting the optional arrays rebuilds the features from their
        directions.
        """
        Unew = np.asarray(Unew, np.float64)
        if self.P == 0:
            self.set_atoms(Unew, A_sh=A_sh, aty=aty, sqnorms=sqnorms)
            return
        Anew = self.eng.build_atoms(Unew) if A_sh is None else A_sh
        self._validate_prebuilt(Anew, Unew.shape[1])
        self.A = self._concat(self.A, Anew)
        k = Unew.shape[1]
        P0 = self.P
        self.U = np.concatenate([self.U, Unew], axis=1)
        aty_new = (
            self.eng.screen(Anew, self.eng.ys)
            if aty is None
            else self._validate_stats(aty, k, "aty")
        )
        sq_new = (
            self.eng.sqnorms(Anew)
            if sqnorms is None
            else self._validate_stats(sqnorms, k, "sqnorms")
        )
        self.Aty = np.concatenate([self.Aty, aty_new])
        self.norms = np.concatenate([self.norms, np.sqrt(np.maximum(sq_new, 1e-300))])
        G = np.zeros((P0 + k, P0 + k))
        G[:P0, :P0] = self.G
        self.G = G
        self.known = np.concatenate([self.known, np.zeros(k, dtype=bool)])
        self.beta = np.concatenate([self.beta, np.zeros(k)])

    def _validate_prebuilt(self, A_sh, width):
        if A_sh is None or len(A_sh) != len(self.eng.slices):
            raise ValueError("prebuilt atom shards do not match the engine")
        for A, (start, end) in zip(A_sh, self.eng.slices):
            if tuple(A.shape) != (end - start, width):
                raise ValueError("prebuilt atom shard has the wrong shape")

    @staticmethod
    def _validate_stats(values, width, name):
        values = np.asarray(values, np.float64)
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite vector of length {width}")
        return values

    def _concat(self, A_sh, B_sh):
        if self.eng.backend == "numpy":
            return [np.concatenate([a, b], axis=1) for a, b in zip(A_sh, B_sh)]
        import torch

        return [torch.cat([a, b], dim=1) for a, b in zip(A_sh, B_sh)]

    def keep(self, idx):
        idx = list(idx)
        self.A = self.eng.slice_cols(self.A, idx)
        self.U = self.U[:, idx]
        self.Aty = self.Aty[idx]
        self.norms = self.norms[idx]
        self.G = self.G[np.ix_(idx, idx)]
        self.known = self.known[idx]
        old2new = {j: i for i, j in enumerate(idx)}
        self.S = [old2new[j] for j in self.S if j in old2new]
        self.beta = self.beta[idx]

    def _ensure_gram(self, cols):
        need = [j for j in cols if not self.known[j]]
        if not need:
            return
        # full-height Gram columns: A^T a_j for each needed j (one engine pass)
        sub = self.eng.slice_cols(self.A, need)
        Gn = self._cross(self.A, sub)
        for t, j in enumerate(need):
            self.G[:, j] = Gn[:, t]
            self.G[j, :] = Gn[:, t]
            self.known[j] = True

    def _cross(self, A_sh, B_sh):
        out = None
        for i in range(len(A_sh)):
            A, B = A_sh[i], B_sh[i]
            acc = np.zeros((A.shape[1], B.shape[1]))
            for s in range(0, A.shape[0], self.eng.chunk):
                e = min(A.shape[0], s + self.eng.chunk)
                acc += (
                    self.eng._host64(self.eng._f64(A[s:e]).T @ self.eng._f64(B[s:e]))
                    if self.eng.backend != "numpy"
                    else A[s:e].astype(np.float64).T @ B[s:e].astype(np.float64)
                )
            out = acc if out is None else out + acc
        return out

    def solve(self, lam, max_outer=None, tol=1e-8, batch_add=16):
        """Numerically solve the finite Lasso, warm-started from prior state.

        Each screening pass admits up to `batch_add` violating atoms at once
        (sign-infeasible ones are dropped by the KKT loop), so the number of
        full-row passes usually follows active-set depth rather than adding only
        one coordinate at a time.  Ridge regularization, tolerances, and loop
        guards mean that the returned point need not satisfy exact Lasso KKT
        conditions.

        Returns (beta_host, nu_sh (sharded float64), obj, S)."""
        P = self.P
        if P == 0:
            nu_sh = [y_ * 1.0 for y_ in self.eng.ys]
            yh = self.eng._yhost
            return np.zeros(0), nu_sh, 0.5 * float(yh @ yh), []
        if max_outer is None:
            max_outer = 2 * P + 32
        S = [j for j in self.S if j < P]
        bc = np.zeros(P)
        for j in S:
            bc[j] = self.beta[j] if j < len(self.beta) else 0.0
        r_sh = None
        for _outer in range(max_outer):
            drops = 0
            while S:
                self._ensure_gram(S)
                ns = self.norms[S]
                Gs = self.G[np.ix_(S, S)] / np.outer(ns, ns)
                sgn = np.sign(bc[S])
                sgn[sgn == 0] = 1.0
                rhs = self.Aty[S] / ns - lam * sgn / ns
                try:
                    L = np.linalg.cholesky(Gs + self.ridge * np.eye(len(S)))
                    bt = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
                except np.linalg.LinAlgError:
                    bt = np.linalg.lstsq(
                        Gs + self.ridge * np.eye(len(S)), rhs, rcond=None
                    )[0]
                bs = bt / ns
                bad = (np.sign(bs) != sgn) | (np.abs(bs) * ns <= 1e-14)
                if not bad.any():
                    bc[:] = 0.0
                    bc[np.asarray(S)] = bs
                    break
                drops += 1
                if drops > 4 * P + 64:  # cycling guard (ridge makes this rare)
                    bc[:] = 0.0
                    bc[np.asarray(S)] = bs * (~bad)
                    break
                # drop ALL sign-infeasible coordinates at once
                S = [S[i] for i in range(len(S)) if not bad[i]]
            else:
                bc[:] = 0.0
            r_sh = self.eng.residual(self.A, bc)
            corr = self.eng.screen(self.A, r_sh)
            viol = np.abs(corr) - lam
            if S:
                viol[np.asarray(S)] = -np.inf
            order = np.argsort(-viol)
            adm = [int(j) for j in order[:batch_add] if viol[j] > tol * max(1.0, lam)]
            if not adm:
                break
            for j in adm:
                S.append(j)
                bc[j] = -np.sign(corr[j]) * 1e-12
        self.S, self.beta = list(S), bc.copy()
        rss = sum(float(self.eng._host64((v * v).sum())) for v in r_sh)
        obj = 0.5 * rss + lam * float(np.abs(bc).sum())
        nu_sh = [-v for v in r_sh]
        return bc, nu_sh, obj, list(S)
