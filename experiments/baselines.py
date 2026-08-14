"""Baselines for the convex two-layer ReLU program (P) / nonconvex training (1).

All methods are scored on the SAME regularized objective with the same penalty
lambda = beta (the Pilanci scaling under which (P) and (1) share an optimum):

    F(net) = 1/2 || sum_j relu(X a_j) b_j - y ||^2 + (beta/2) sum_j (||a_j||^2 + b_j^2).

A trained network's F is >= the convex optimum F* of (P); the gap measures how far
nonconvex / sampling heuristics land from the certified global optimum that refinement baseline
reaches.  They use the shared array backend so they can run on the same device.

  * train_net  -- SGD or Adam on the nonconvex objective (1) (manual gradients).
  * sample_lasso -- random chamber sampling: m random gate directions g_i, atoms
                  (X g_i)_+, solved by the SAME exact active-set LASSO master as
                  refinement baseline.  This is the Pilanci / Mishkin / Kim--Pilanci "sample the
                  activation patterns" approach with priced columns replaced by random
                  ones, so the only difference from refinement baseline is *how the columns are
                  chosen*.
"""

from __future__ import annotations

import time

import numpy as np

from relu_chambers.array_backend import get_xp, to_cpu, to_device
from relu_chambers.reference.lasso import active_set_lasso


def reg_objective(X, y, A, b, beta, xp=None):
    """F(net) = 1/2||relu(XA)b - y||^2 + beta/2 (||A||_F^2 + ||b||^2), in float64."""
    if xp is None:
        xp = get_xp("cpu")
    Xx = xp.asarray(X)
    yy = xp.asarray(y)
    H = xp.maximum(Xx @ A, 0.0)
    r = H @ b - yy
    data = 0.5 * float(r.astype(xp.float64) @ r.astype(xp.float64))
    reg = 0.5 * beta * (float((A * A).sum()) + float((b * b).sum()))
    return data + reg


def _predict(X, A, b, xp):
    return xp.maximum(X @ A, 0.0) @ b


def train_net(
    X,
    y,
    m,
    beta,
    opt="adam",
    lr=None,
    epochs=400,
    batch=None,
    device="cpu",
    rng_seed=0,
    X_te=None,
    y_te=None,
    init_scale=None,
    record_every=0,
    time_budget=None,
    wd=None,
):
    """Train the 2-layer ReLU net (1) by full-batch/minibatch SGD, Adam or AdamW.

    All optimizers are scored on the SAME regularized objective F (lambda=beta).
    opt="adamw" uses decoupled weight decay applied as a STABLE proximal shrink
    w <- (w - lr*adam_step)/(1 + lr*wd) with rate wd (default beta).  Because the
    decay is decoupled from the adaptive preconditioner, its rate must be tuned
    independently: the AdamW fixed point is w* ~ -adam_step/wd, so wd=beta=lambda
    with lambda large (lambda is a fraction of lambda_max on real data) drives the
    weights to zero regardless of fit.  wd is therefore swept by the caller; every
    run is still scored on the same F (beta), so the comparison is apples-to-apples.

    Returns dict(obj, test_mse, A, b, time, history).  Manual gradients:
      r = relu(XA) b - y    (minibatch gradients are scaled by n/|batch| so the
                             regularizer keeps the same relative weight)
      grad_b = relu(XA)^T r + beta b
      grad_A = X^T( (r[:,None] * 1{XA>=0}) * b[None,:] ) + beta A.
    """
    xp = get_xp(device)
    dtype = xp.float32 if device == "gpu" else xp.float64
    Xx = to_device(X, device).astype(dtype)
    yy = to_device(np.asarray(y, np.float64), device).astype(dtype)
    n, d = X.shape
    rs = np.random.default_rng(rng_seed)
    if init_scale is None:
        init_scale = 1.0 / np.sqrt(max(d, 1))
    A = xp.asarray(rs.standard_normal((d, m)) * init_scale, dtype=dtype)
    b = xp.asarray(rs.standard_normal(m) * init_scale, dtype=dtype)
    if lr is None:
        lr = (
            1e-3
            if opt in ("adam", "adamw")
            else 1e-2 / max(1.0, np.linalg.norm(X, 2) ** 2 / n)
        )
    if batch is None:
        batch = n
    # adam state
    mA = xp.zeros_like(A)
    vA = xp.zeros_like(A)
    mb = xp.zeros_like(b)
    vb = xp.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    step = 0
    hist = {"epoch": [], "obj": [], "time": []}
    idx = np.arange(n)
    scale_full = float(n)  # gradient scale reference for minibatches
    for ep in range(epochs):
        if batch < n:
            rs.shuffle(idx)
        for s in range(0, n, batch):
            bi = idx[s : s + batch]
            Xb = Xx[bi] if batch < n else Xx
            yb = yy[bi] if batch < n else yy
            gscale = scale_full / len(bi) if batch < n else 1.0
            Zb = Xb @ A
            Hb = xp.maximum(Zb, 0.0)
            Sb = (Zb >= 0).astype(dtype)
            r = Hb @ b - yb
            gb_data = gscale * (Hb.T @ r)
            gA_data = gscale * ((Xb.T @ (r[:, None] * Sb)) * b[None, :])
            step += 1
            if opt == "adamw":
                # decoupled weight decay as a STABLE proximal shrink.  The naive
                # multiplicative form A*(1-lr*wd) goes negative and diverges once
                # lr*wd>1 -- which is immediate when wd=beta=lambda is large
                # (lr*lambda was 0.2-17 here), the source of the NaN blowups; the
                # proximal /(1+lr*wd) is in (0,1) for any lr,wd>0.
                wd_eff = beta if wd is None else wd
                mA = b1 * mA + (1 - b1) * gA_data
                vA = b2 * vA + (1 - b2) * gA_data * gA_data
                mb = b1 * mb + (1 - b1) * gb_data
                vb = b2 * vb + (1 - b2) * gb_data * gb_data
                bc1 = 1 - b1**step
                bc2 = 1 - b2**step
                A = (A - lr * (mA / bc1) / (xp.sqrt(vA / bc2) + eps)) / (
                    1.0 + lr * wd_eff
                )
                b = (b - lr * (mb / bc1) / (xp.sqrt(vb / bc2) + eps)) / (
                    1.0 + lr * wd_eff
                )
            elif opt == "adam":
                gA = gA_data + beta * A
                gb = gb_data + beta * b
                mA = b1 * mA + (1 - b1) * gA
                vA = b2 * vA + (1 - b2) * gA * gA
                mb = b1 * mb + (1 - b1) * gb
                vb = b2 * vb + (1 - b2) * gb * gb
                bc1 = 1 - b1**step
                bc2 = 1 - b2**step
                A = A - lr * (mA / bc1) / (xp.sqrt(vA / bc2) + eps)
                b = b - lr * (mb / bc1) / (xp.sqrt(vb / bc2) + eps)
            else:
                A = A - lr * (gA_data + beta * A)
                b = b - lr * (gb_data + beta * b)
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            hist["epoch"].append(ep)
            hist["obj"].append(reg_objective(X, y, A, b, beta, xp))
            hist["time"].append(time.time() - t0)
        if time_budget and time.time() - t0 > time_budget:
            break
    obj = reg_objective(X, y, A, b, beta, xp)
    test_mse = np.nan
    if X_te is not None:
        Xt = to_device(X_te, device).astype(dtype)
        pred = _predict(Xt, A, b, xp)
        rt = pred - to_device(np.asarray(y_te, np.float64), device).astype(dtype)
        test_mse = float((rt.astype(xp.float64) @ rt.astype(xp.float64)) / len(y_te))
    return dict(
        obj=obj,
        test_mse=test_mse,
        A=to_cpu(A),
        b=to_cpu(b),
        time=time.time() - t0,
        history=hist,
        m=m,
    )


def sample_lasso(X, y, lam, m, device="cpu", rng_seed=0, X_te=None, y_te=None):
    """Random chamber sampling: m random unit gate directions -> atoms (X g_i)_+,
    solved by the exact active-set LASSO master (same master refinement baseline uses).  Returns
    dict(obj, test_mse, n_active, time)."""
    xp = get_xp(device)
    dtype = xp.float32 if device == "gpu" else xp.float64
    rs = np.random.default_rng(rng_seed)
    _n, d = X.shape
    G = rs.standard_normal((d, m))
    G /= np.linalg.norm(G, axis=0, keepdims=True) + 1e-30
    Xx = to_device(X, device).astype(dtype)
    Gd = xp.asarray(G, dtype=dtype)
    A = xp.maximum(Xx @ Gd, 0.0)
    ydev = to_device(np.asarray(y, np.float64), device).astype(dtype)
    t0 = time.time()
    beta, _nu, obj, S, _ = active_set_lasso(A, ydev, lam, xp)
    dt = time.time() - t0
    test_mse = np.nan
    if X_te is not None:
        bc = to_cpu(beta)
        At = np.maximum(np.asarray(X_te) @ G, 0.0)
        pred = At @ bc
        rt = pred - np.asarray(y_te)
        test_mse = float(rt @ rt / len(y_te))
    return dict(obj=obj, test_mse=test_mse, n_active=len(S), time=dt, m=m)
