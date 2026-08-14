"""Restricted convex ReLU master problem (Pilanci-Ergen group-lasso).

Given a set A = {s_1,...,s_P} of activation patterns, solve

  min_{v_j, w_j}  (1/2)|| sum_j D_j X (v_j - w_j) - y ||^2
                  + lam * sum_j ( ||v_j||_2 + ||w_j||_2 )
  s.t.  (2 D_j - I) X v_j >= 0,   (2 D_j - I) X w_j >= 0     (cone constraints)

where D_j = diag(s_j). This is a second-order cone program. The cone
constraints guarantee D_j X v_j = (X v_j)_+, so each block is a genuine ReLU
feature. The group penalty induces chamber-sparsity.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

_SOLVERS = ["CLARABEL", "SCS"]


def solve_master(X, y, patterns, lam, solver_opts=None, verbose=False):
    """Solve the restricted convex program over `patterns` (list of {0,1}^n).

    Returns dict with objective, prediction yhat, residual nu = y - yhat,
    per-block weights, group norms, and the active (nonzero-block) patterns.
    """
    n, d = X.shape
    P = len(patterns)
    if P == 0:
        yhat = np.zeros(n)
        nu = y - yhat
        obj = 0.5 * float(nu @ nu)
        return dict(
            obj=obj,
            yhat=yhat,
            nu=nu,
            V=np.zeros((0, d)),
            W=np.zeros((0, d)),
            group_norms=np.zeros(0),
            active_idx=[],
            patterns=patterns,
            status="empty",
        )

    sigmas = [np.where(np.asarray(s) > 0, 1.0, -1.0) for s in patterns]  # +-1
    svecs = [np.asarray(s, dtype=float) for s in patterns]

    V = [cp.Variable(d) for _ in range(P)]
    W = [cp.Variable(d) for _ in range(P)]

    pred = 0
    constraints = []
    pen = 0
    for j in range(P):
        Dj_pos = cp.multiply(svecs[j], X @ (V[j] - W[j]))  # D_j X (v-w)
        pred = pred + Dj_pos
        constraints.append(cp.multiply(sigmas[j], X @ V[j]) >= 0)
        constraints.append(cp.multiply(sigmas[j], X @ W[j]) >= 0)
        pen = pen + cp.norm(V[j], 2) + cp.norm(W[j], 2)

    obj = 0.5 * cp.sum_squares(pred - y) + lam * pen
    prob = cp.Problem(cp.Minimize(obj), constraints)

    opts = dict(verbose=verbose)
    if solver_opts:
        opts.update(solver_opts)

    last_err = None
    for sv in _SOLVERS:
        try:
            prob.solve(solver=sv, **opts)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if prob.value is None:
        raise RuntimeError(f"master solve failed: status={prob.status} err={last_err}")

    Vv = np.array(
        [np.asarray(v.value).ravel() if v.value is not None else np.zeros(d) for v in V]
    )
    Ww = np.array(
        [np.asarray(w.value).ravel() if w.value is not None else np.zeros(d) for w in W]
    )
    yhat = np.zeros(n)
    for j in range(P):
        yhat = yhat + svecs[j] * (X @ (Vv[j] - Ww[j]))
    nu = y - yhat
    gnorm = np.linalg.norm(Vv, axis=1) + np.linalg.norm(Ww, axis=1)
    active = [
        j for j in range(P) if gnorm[j] > 1e-6 * max(1.0, gnorm.max() if P else 1.0)
    ]
    return dict(
        obj=float(prob.value),
        yhat=yhat,
        nu=nu,
        V=Vv,
        W=Ww,
        group_norms=gnorm,
        active_idx=active,
        patterns=patterns,
        status=prob.status,
    )
