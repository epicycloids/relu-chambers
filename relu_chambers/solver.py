"""Residual-adaptive column generation for two-layer ReLU fitting.

This module is the numerical implementation described in the paper.  It keeps
the scalar-output two-layer ReLU objective fixed while using a finite evolving
working set.  It must not be confused with the ideal method that assumes an
exact restricted master and exact global pricing.

  ENGINE     all data-touching kernels run through data_engine.ShardedData: row
             shards across devices (numpy CPU blocks, or torch CUDA/ROCm
             devices) with chunked reductions.  The loop also performs
             O(n) host residual movement and can move O(nP) chamber data; its
             communication is not independent of n.  numpy x1 is the plain-CPU
             special case.
  MASTER     numerical warm active-set Lasso with a ridge, tolerances, loop
             guards, and incremental float64 Gram caching
             (restricted_master.AtomLasso).
  CONSOLIDATE the per-atom refinement is replaced by the GATED JOINT step
             (consolidation.py): freeze active masks, solve the group-Lasso over all
             neuron vectors jointly on a host sketch, project to unit ReLU
             atoms, and replace the working set only when a full-data numerical
             restricted refit has no larger computed objective.
  PRICING    one joint +-nu batched search on a host importance sketch with
             survivor pruning; full-data numerical rescoring; SDP-rounding
             escalation only when the cheap oracle finds no violator.
  BOUND      pricing_bounds: an exact-arithmetic upper-bound construction evaluated
             through warm-started host-sketch ascent and a full-data numerical
             replay.  Ordinary floating accumulation/eigensolves are not
             outward-certified, so the returned gap_cert field is explicitly
             marked numerical.
  STOPPING   heuristic gap, no-violator (with one escalated re-check), plateau,
             or time budget; numerical-bound stopping is opt-in.  Returned
             quantities receive a final float64 replay, not formal intervals.
"""

from __future__ import annotations

import time

import numpy as np

from .adaptive_pricing import search_signed_atoms
from .consolidation import propose_consolidation
from .data_engine import ShardedData
from .pricing_bounds import dual_gap, estimate_pricing_bounds
from .restricted_master import AtomLasso


def _dedup_cols(Unew_cols, U_existing, cos_tol=0.9995):
    return _dedup_cols_indexed(Unew_cols, U_existing, cos_tol)[0]


def _dedup_cols_indexed(Unew_cols, U_existing, cos_tol=0.9995, normalize=True):
    """Oriented deduplication plus indices into ``Unew_cols``."""
    fresh = []
    indices = []
    for index, u in enumerate(Unew_cols):
        u = np.asarray(u, float)
        if normalize:
            u = u / (np.linalg.norm(u) + 1e-30)
        else:
            # Retained oracle features were formed from this exact vector.
            # The oracle has already normalized it; changing its scale here
            # would misalign the cached column and the direction stored by the
            # restricted master.
            u = u.copy()
        if (
            U_existing is not None
            and U_existing.shape[1]
            and float(np.max(u @ U_existing)) > cos_tol
        ):
            continue
        if any(float(u @ w) > cos_tol for w in fresh):
            continue
        fresh.append(u)
        indices.append(index)
    return fresh, indices


def make_engine(
    X, y, device="cpu", devices=None, dtype=None, threads=False, weights=None
):
    """Build the data engine.  device: "cpu" (numpy), "torch" / "cuda" /
    "cuda:0,cuda:1" (torch backend; ROCm presents as cuda)."""
    if devices is not None:
        backend = (
            "torch" if any("cuda" in d or "torch" in d for d in devices) else "numpy"
        )
        devs = [d.replace("torch:", "") for d in devices]
    elif device in ("cpu", "numpy"):
        backend, devs = "numpy", ["cpu"]
    elif device.startswith(("cuda", "torch")):
        backend = "torch"
        spec = device.replace("torch:", "")
        devs = spec.split(",") if "," in spec else [spec if spec != "torch" else "cuda"]
    else:
        backend, devs = "numpy", ["cpu"]
    if dtype is None:
        dtype = "float32" if backend == "torch" else "float64"
    return ShardedData(
        X,
        y,
        devices=devs,
        backend=backend,
        dtype=dtype,
        threads=threads,
        weights=weights,
    )


def solve_rcg(
    X,
    y,
    lam,
    device="cpu",
    engine=None,
    dtype_data=None,
    max_iter=100,
    eps_rel=1e-4,
    eps_heur=1e-3,
    price_tol=1e-3,
    add_per_round=8,
    max_cols=512,
    certify_every=5,
    cert_steps=15,
    gated_every=1,
    gated_iters=100,
    sketch_m="auto",
    time_budget=None,
    rng_seed=0,
    verbose=False,
    stall_rtol=1e-7,
    patience=6,
    final_verify=True,
    devices=None,
    threads=False,
    oracle_kwargs=None,
    allow_numerical_bound_stop=False,
):
    """Run RCG.  Returns dict(obj, nu, beta, U, n_active, gap_cert,
    gap_heur, rho_lb, history, timings, ...).

    Despite its concise name, ``gap_cert`` is computed with round-to-nearest
    arithmetic and is not outward-certified.  It does not
    trigger stopping unless ``allow_numerical_bound_stop=True``.
    """
    t_start = time.time()
    eng = engine or make_engine(
        X, y, device=device, devices=devices, dtype=dtype_data, threads=threads
    )
    n, _d = eng.n, eng.d
    y64 = eng._yhost
    rng = np.random.default_rng(rng_seed)
    if sketch_m == "auto":
        sketch_m = 0 if n <= 65536 else 32768
    ok = dict(sketch_m=sketch_m or 0)
    if oracle_kwargs:
        ok.update(oracle_kwargs)
    row_norms = np.sqrt(np.einsum("ij,ij->i", eng._Xhost, eng._Xhost))

    master = AtomLasso(eng)
    prev_dirs = []
    cert_warm = None
    gated_m = sketch_m  # self-tuning gated sketch size (see below)
    gated_ok = True
    hist = {
        k: []
        for k in (
            "iter",
            "n_cols",
            "n_active",
            "obj",
            "rho_lb",
            "ub",
            "gap_cert",
            "gap_heur",
            "time",
        )
    }
    timings = {"master": 0.0, "gated": 0.0, "price": 0.0, "cert": 0.0, "other": 0.0}
    feature_reuse = {
        "pricing_calls": 0,
        "requested_calls": 0,
        "retained_calls": 0,
        "fallback_calls": 0,
        "retained_payload_bytes_peak": 0,
        "required_peak_payload_bytes_peak": 0,
        "appended_reused_columns": 0,
        "appended_rebuilt_columns": 0,
    }
    escalated = False
    stop_reason = "max_iter"
    obj = 0.5 * float(y64 @ y64)

    for it in range(max_iter):
        # ---------------- master ----------------
        t0 = time.time()
        beta, nu_sh, obj, S = master.solve(lam)
        # A newly priced atom must participate in at least one restricted
        # solve before a memory cap can rank it.  Appended coefficients start at
        # zero, so capping immediately after append would always discard new
        # columns once the cap was reached.
        if master.P > max_cols:
            order = np.argsort(np.abs(master.beta))[::-1]
            master.keep(sorted(order[:max_cols].tolist()))
            _, nu_sh, obj, S = master.solve(lam)
        timings["master"] += time.time() - t0

        # ------- gated consolidation: REPLACE working set if it wins --------
        # at the sketch-size cap with a losing proposal (typical on agnostic
        # real data), halve the gated frequency instead of growing further
        gated_skip = (
            (not gated_ok)
            and sketch_m
            and gated_m >= 4 * sketch_m
            and it % (2 * gated_every) != 0
        )
        if gated_every and len(S) >= 2 and it % gated_every == 0 and not gated_skip:
            t0 = time.time()
            Sg = S
            if len(Sg) > 96:  # consolidate the strongest blocks only
                ord_ = np.argsort(-np.abs(master.beta[np.asarray(S)]))[:96]
                Sg = [S[i] for i in ord_]
            Un = propose_consolidation(
                eng,
                master.U[:, Sg],
                master.beta[np.asarray(Sg)],
                lam,
                sketch_m=gated_m,
                rng=rng,
                iters=gated_iters,
                rng_seed=rng_seed + it,
                row_norms=row_norms,
            )
            if Un is not None and Un.shape[1] >= 1:
                trial = AtomLasso(eng)
                trial.set_atoms(Un)
                # warm start: the proposal's blocks are the expected active set
                # (positive coefficients by construction of the projection)
                trial.S = list(range(Un.shape[1]))
                trial.beta = np.full(Un.shape[1], 1e-12)
                bt, nut, objt, St = trial.solve(lam)
                if objt <= obj:
                    master = trial
                    beta, nu_sh, obj, S = bt, nut, objt, St
                    gated_ok = True
                    if gated_m:  # proposal won: decay sketch toward baseline
                        gated_m = max(sketch_m, gated_m // 2)
                else:
                    # proposal lost: usually the sketch was too noisy at this n
                    # -- the safeguard detects exactly that, so GROW the gated
                    # sketch (self-tuning); append only the strongest blocks
                    gated_ok = False
                    if gated_m and gated_m < min(n, 4 * sketch_m):
                        gated_m = min(n, 4 * sketch_m, 4 * gated_m)
                    fresh = _dedup_cols(list(Un.T[:8]), master.U)
                    if fresh:
                        master.append_atoms(np.stack(fresh, axis=1))
                        beta, nu_sh, obj, S = master.solve(lam)
            timings["gated"] += time.time() - t0

        # ---------------- prune zero-weight atoms; same-chamber merge -------
        t0 = time.time()
        bc = master.beta
        thr = 1e-10 * max(1.0, float(np.abs(bc).max()) if master.P else 1.0)
        keep = [j for j in range(master.P) if abs(bc[j]) > thr]
        if 0 < len(keep) < master.P:
            master.keep(keep)
            # keep() changes the represented model.  Re-solve so pricing never
            # uses a residual/objective left over from the pre-pruned master.
            _, nu_sh, obj, S = master.solve(lam)
        if it % 3 == 2 and len(master.S) >= 2 and _compress_chambers(master, eng):
            _, nu_sh, obj, S = master.solve(lam)
        n_active = len(S)
        timings["other"] += time.time() - t0

        # ---------------- pricing (joint +-nu) ----------------
        t0 = time.time()
        nu_host = eng.gather_vec(nu_sh)
        Ua = master.U[:, S] if n_active else None
        ba = master.beta[np.asarray(S)] if n_active else None
        # regimes with many active atoms need bigger batches, but
        # only while consolidation is keeping up (else batching feeds a runaway)
        eff_add = (
            max(add_per_round, min(32, n_active // 4)) if gated_ok else add_per_round
        )
        price_ok = dict(ok)
        # No append follows the last permitted round, so retaining finalists
        # there could only create an unused buffer.
        if it + 1 >= max_iter:
            price_ok["retain_finalist_features"] = False
        pr = search_signed_atoms(
            eng,
            nu_sh,
            nu_host,
            lam,
            k=eff_add,
            rng=rng,
            U_active=Ua,
            beta_active=ba,
            prev_dirs=prev_dirs,
            use_sdp=escalated,
            row_norms=row_norms,
            **price_ok,
        )
        cand = list(pr["plus"]) + list(pr["minus"])
        addition = pr.get("addition", [(v, v, u) for v, u in cand])
        addition_cache = pr.get("addition_cache")
        reuse = pr.get("feature_reuse", {})
        feature_reuse["pricing_calls"] += 1
        if reuse.get("requested"):
            feature_reuse["requested_calls"] += 1
            if reuse.get("retained"):
                feature_reuse["retained_calls"] += 1
            else:
                feature_reuse["fallback_calls"] += 1
        feature_reuse["retained_payload_bytes_peak"] = max(
            feature_reuse["retained_payload_bytes_peak"],
            int(reuse.get("retained_payload_bytes", 0)),
        )
        feature_reuse["required_peak_payload_bytes_peak"] = max(
            feature_reuse["required_peak_payload_bytes_peak"],
            int(reuse.get("required_peak_payload_bytes", 0)),
        )
        rho_lb = max([v for v, _ in cand], default=0.0)
        prev_dirs = [u for _, u in sorted(cand, key=lambda z: -z[0])[:6]]
        # Keep only the compact append payload.  In particular, do not retain
        # the complete oracle result while certificates or the next pricing
        # call allocate their own finalist buffers.
        pr = None
        timings["price"] += time.time() - t0

        # ---------------- certificate (periodic; honest reporting) ----------
        no_violator = rho_lb <= lam * (1.0 + price_tol)
        if no_violator:
            # This round cannot append, even when it triggers the one-time
            # escalated recheck, so release any unused retained columns before
            # certificate work allocates its own buffers.
            addition_cache = None
        ub = gap_cert = None
        if certify_every and (
            it % certify_every == certify_every - 1 or no_violator or it == max_iter - 1
        ):
            t0 = time.time()
            cert = estimate_pricing_bounds(
                eng,
                nu_host,
                n_steps=cert_steps,
                warm=cert_warm,
                sketch_m=sketch_m or 0,
                rng=rng,
                row_norms=row_norms,
            )
            cert_warm = cert["warm"]
            ub = max(cert["ub_plus"], cert["ub_minus"])
            gap_cert, _ = dual_gap(
                y64, nu_host, lam, cert["ub_plus"], cert["ub_minus"], obj
            )
            timings["cert"] += time.time() - t0
        gap_heur, _ = dual_gap(
            y64, nu_host, lam, max(rho_lb, lam), max(rho_lb, lam), obj
        )

        hist["iter"].append(it)
        hist["n_cols"].append(master.P)
        hist["n_active"].append(n_active)
        hist["obj"].append(obj)
        hist["rho_lb"].append(float(rho_lb))
        hist["ub"].append(None if ub is None else float(ub))
        hist["gap_cert"].append(None if gap_cert is None else float(gap_cert))
        hist["gap_heur"].append(float(gap_heur))
        hist["time"].append(time.time() - t_start)
        if verbose:
            ubs = f"{ub:.4f}" if ub is not None else "  --  "
            gcs = f"{gap_cert:.3e}" if gap_cert is not None else "   --   "
            print(
                f"[RCG {it:03d}] cols={master.P:4d} act={n_active:3d} "
                f"obj={obj:.6f} rho_lb={rho_lb:.4f} ub={ubs} "
                f"gapN={gcs} gapH={gap_heur:.3e}"
            )

        # ---------------- stopping ----------------
        if (
            allow_numerical_bound_stop
            and gap_cert is not None
            and gap_cert <= eps_rel * max(1.0, abs(obj))
        ):
            addition_cache = None
            stop_reason = "numerical_bound"
            break
        if eps_heur and gap_heur <= eps_heur * max(1.0, abs(obj)) and it > 2:
            addition_cache = None
            stop_reason = "heuristic_gap"
            break
        if no_violator:
            addition_cache = None
            if not escalated:
                escalated = True
                continue
            stop_reason = "no_violator"
            break
        if escalated and not no_violator:
            escalated = False
        if len(hist["obj"]) > patience:
            rec = hist["obj"][-(patience + 1) :]
            if (
                0 <= (rec[0] - rec[-1]) <= stall_rtol * max(1.0, abs(rec[-1]))
                and rho_lb <= lam * 1.05
            ):
                addition_cache = None
                stop_reason = "plateau"
                break
        if time_budget and time.time() - t_start > time_budget:
            addition_cache = None
            stop_reason = "time_budget"
            break

        # Appending after the final permitted round creates columns that are
        # never solved or scored and can make the returned model exceed the
        # declared column cap.
        if it + 1 >= max_iter:
            addition_cache = None
            break

        # ---------------- add violating atoms ----------------
        ranked_additions = sorted(enumerate(addition), key=lambda item: -item[1][0])
        viol = [
            (index, item[2])
            for index, item in ranked_additions
            if item[1] > lam * (1.0 + 1e-4)
        ][: 2 * eff_add]
        fresh, fresh_positions = _dedup_cols_indexed(
            # search_signed_atoms returns normalized directions.  Preserve those exact
            # bytes in both paths so enabling reuse changes only data movement.
            [u for _, u in viol],
            master.U,
            normalize=False,
        )
        if not fresh and not no_violator:
            addition_cache = None
            stop_reason = "no_new_column"
            break
        if fresh:
            Ufresh = np.stack(fresh, axis=1)
            cache_indices = [viol[pos][0] for pos in fresh_positions]
            if addition_cache is not None:
                Anew = eng.slice_cols(addition_cache["A_sh"], cache_indices)
                master.append_atoms(
                    Ufresh,
                    A_sh=Anew,
                    aty=np.asarray(addition_cache["aty"])[cache_indices],
                    sqnorms=np.asarray(addition_cache["sqnorm"])[cache_indices],
                )
                feature_reuse["appended_reused_columns"] += len(fresh)
                Anew = None
            else:
                master.append_atoms(Ufresh)
                feature_reuse["appended_rebuilt_columns"] += len(fresh)
        addition_cache = None

    # ---------------- final float64 diagnostic replay ----------------
    out = dict(
        history=hist,
        timings=timings,
        stop_reason=stop_reason,
        n_iter=len(hist["iter"]),
        feature_reuse=feature_reuse,
    )
    if final_verify and master.P:
        t0 = time.time()
        S = master.S
        Uact = master.U[:, S] if S else master.U
        ver = verify_solution(
            eng,
            lam,
            Uact,
            cert_steps=max(30, 2 * cert_steps),
            rng=rng,
            sketch_m=sketch_m or 0,
            row_norms=row_norms,
        )
        out.update(ver)
        timings["verify"] = time.time() - t0
    else:
        out.update(
            dict(
                obj=obj,
                nu=eng.gather_vec(nu_sh) if master.P else y64,
                beta=master.beta,
                U=master.U,
                n_active=len(master.S),
                n_cols=master.P,
                rho_lb=hist["rho_lb"][-1] if hist["rho_lb"] else 0.0,
                gap_cert=hist["gap_cert"][-1] if hist["gap_cert"] else None,
                gap_heur=hist["gap_heur"][-1] if hist["gap_heur"] else None,
            )
        )
    out["time"] = time.time() - t_start
    out["gap_cert_is_rigorous"] = False
    out["certificate_status"] = (
        "exact-arithmetic bound evaluated in round-to-nearest floating point; "
        "not outward-certified"
    )
    return out


def _compress_chambers(master, eng):
    """Same-chamber, same-sign merge over active finite-master atoms."""
    S = master.S
    if len(S) < 2:
        return False
    keys = eng.chamber_keys(master.U[:, S])
    beta = master.beta
    groups = {}
    for t, j in enumerate(S):
        sgn = 1.0 if beta[j] >= 0 else -1.0
        groups.setdefault((keys[t], sgn), []).append(j)
    merged = {k: v for k, v in groups.items() if len(v) > 1}
    if not merged:
        return False
    drop = set()
    new_dirs = []
    for (_, sgn), js in merged.items():
        vG = sum(abs(beta[j]) * master.U[:, j] for j in js)
        nrm = np.linalg.norm(vG)
        if nrm > 1e-30:
            new_dirs.append(vG / nrm)
        drop.update(js)
    keep = [j for j in range(master.P) if j not in drop]
    if keep:
        master.keep(keep)
        if new_dirs:
            master.append_atoms(np.stack(new_dirs, axis=1))
    elif new_dirs:
        master.set_atoms(np.stack(new_dirs, axis=1))
    return True


def verify_solution(
    eng, lam, U_dirs, cert_steps=30, rng=None, sketch_m=0, row_norms=None
):
    """Stricter float64 diagnostic replay of a returned finite working set.

    The replay tightens the numerical master tolerance, escalates heuristic
    pricing, rescores on all rows, and evaluates both bound forms.  It is not an
    exact solve, a global price oracle, or an outward-certified verification.
    """
    rng = rng or np.random.default_rng(0)
    U = np.asarray(U_dirs, float)
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-30
    master = AtomLasso(eng)
    master.set_atoms(U)
    beta, nu_sh, obj, S = master.solve(lam, tol=1e-9)
    nu_host = eng.gather_vec(nu_sh)
    pr = search_signed_atoms(
        eng,
        nu_sh,
        nu_host,
        lam,
        k=4,
        rng=rng,
        U_active=U[:, S] if S else None,
        sketch_m=sketch_m,
        n_random=32,
        iters_sketch=60,
        iters_full=15,
        top_polish=10,
        use_sdp=True,
        row_norms=row_norms,
    )
    rho_lb = pr["best"]
    cert = estimate_pricing_bounds(
        eng,
        nu_host,
        n_steps=cert_steps,
        forms=("joint", "split"),
        sketch_m=sketch_m,
        rng=rng,
        row_norms=row_norms,
    )
    y64 = eng._yhost
    gap_cert, theta = dual_gap(
        y64, nu_host, lam, cert["ub_plus"], cert["ub_minus"], obj
    )
    gap_heur, _ = dual_gap(y64, nu_host, lam, max(rho_lb, lam), max(rho_lb, lam), obj)
    return dict(
        obj=obj,
        nu=nu_host,
        beta=beta,
        U=U,
        n_active=len(S),
        n_cols=U.shape[1],
        rho_lb=float(rho_lb),
        ub_plus=float(cert["ub_plus"]),
        ub_minus=float(cert["ub_minus"]),
        gap_cert=float(gap_cert),
        gap_heur=float(gap_heur),
        theta=float(theta),
    )
