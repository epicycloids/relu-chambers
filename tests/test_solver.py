"""Focused correctness tests for RCG helpers.

Run with ``python -m tests.test_solver``.
"""

from __future__ import annotations

import os

import numpy as np

from relu_chambers.adaptive_pricing import (
    _ascent_pm,
    _ascent_pm_torch,
    _predicted_decrease,
    _select_distinct,
    search_signed_atoms,
)
from relu_chambers.restricted_master import AtomLasso
from relu_chambers.solver import (
    _dedup_cols,
    _dedup_cols_indexed,
    make_engine,
    solve_rcg,
)


def test_oriented_direction_dedup() -> None:
    u = np.array([1.0])
    existing = u[:, None]

    opposite = _dedup_cols([-u], existing)
    assert len(opposite) == 1, "u and -u are different ReLU atoms"

    near_same = _dedup_cols([np.array([1.0])], existing)
    assert not near_same, "same-oriented duplicate should still be removed"

    U = np.array([[1.0, -1.0]])
    selected = _select_distinct(np.array([2.0, 1.0]), U, k=2, dedup_cos=0.999)
    assert len(selected) == 2


def test_opposite_atom_improves_exact_master() -> None:
    # The two directions have absolute cosine one but orthogonal ReLU features.
    X = np.array([[1.0], [-1.0]])
    y = np.ones(2)
    lam = 0.25
    eng = make_engine(X, y)

    one = AtomLasso(eng)
    one.set_atoms(np.array([[1.0]]))
    _, _, obj_one, _ = one.solve(lam)

    two = AtomLasso(eng)
    two.set_atoms(np.array([[1.0, -1.0]]))
    beta, _, obj_two, _ = two.solve(lam)

    np.testing.assert_allclose(beta, np.array([0.75, 0.75]), atol=1e-8)
    np.testing.assert_allclose(obj_one, 0.71875, atol=1e-10)
    np.testing.assert_allclose(obj_two, 0.4375, atol=1e-10)
    assert obj_two < obj_one


def test_ascent_trial_modes() -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 5), dtype=np.float32)
    nu = rng.standard_normal(200, dtype=np.float32)
    U0 = rng.standard_normal((5, 12), dtype=np.float32)
    signs = np.r_[np.ones(6), -np.ones(6)]
    outputs = {}
    for mode in ("both", "damped", "fixed", "packed"):
        vals, U, sg = _ascent_pm(
            X, nu, U0, signs, n_iters=5, prune_at=0, trial_mode=mode
        )
        assert np.all(np.isfinite(vals))
        np.testing.assert_allclose(np.linalg.norm(U, axis=0), 1.0, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(sg, signs)
        outputs[mode] = vals
    np.testing.assert_allclose(outputs["packed"], outputs["both"], rtol=2e-5, atol=2e-5)

    try:
        _ascent_pm(X, nu, U0, signs, n_iters=1, trial_mode="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid ascent mode should fail loudly")


def test_torch_cpu_ascent_equivalence() -> None:
    """Optional: the one-device loop mirrors NumPy for every trial mode."""
    try:
        import torch  # type: ignore
    except Exception:
        return
    rng = np.random.default_rng(14)
    X = rng.standard_normal((200, 5), dtype=np.float32)
    nu = rng.standard_normal(200, dtype=np.float32)
    U0 = rng.standard_normal((5, 20), dtype=np.float32)
    signs = np.r_[np.ones(10), -np.ones(10)]
    for mode in ("both", "damped", "fixed", "packed"):
        host = _ascent_pm(
            X, nu, U0, signs, n_iters=5, prune_at=2, keep_min=3, trial_mode=mode
        )
        device = _ascent_pm_torch(
            torch.as_tensor(X),
            torch.as_tensor(nu),
            U0,
            signs,
            n_iters=5,
            prune_at=2,
            keep_min=3,
            trial_mode=mode,
            device="cpu",
        )
        np.testing.assert_allclose(device[0], host[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(device[1], host[1], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(device[2], host[2])


def test_torch_cpu_device_pricing_smoke() -> None:
    """Optional integration smoke including exact Torch-engine rescoring."""
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception:
        return
    rng = np.random.default_rng(8)
    X = rng.standard_normal((80, 4), dtype=np.float32)
    y = rng.standard_normal(80)
    nu = rng.standard_normal(80)
    engine = make_engine(X, y, device="torch:cpu", dtype="float32")
    nu_sh = engine.scatter_vec(nu)
    kwargs = dict(
        lam=0.1,
        k=3,
        n_random=4,
        n_subspace=2,
        n_perturb=0,
        sketch_m=0,
        iters_sketch=4,
        iters_full=2,
        top_polish=3,
        ascent_trial="packed",
    )
    host = search_signed_atoms(
        engine, nu_sh, nu, rng=np.random.default_rng(9), search_backend="host", **kwargs
    )
    device = search_signed_atoms(
        engine,
        nu_sh,
        nu,
        rng=np.random.default_rng(9),
        search_backend="device",
        **kwargs,
    )
    np.testing.assert_allclose(device["best"], host["best"], rtol=2e-6, atol=2e-6)
    assert len(device["plus"]) == len(host["plus"]) > 0
    assert len(device["minus"]) == len(host["minus"]) > 0
    assert np.all(np.isfinite([value for value, _ in device["plus"]]))


def test_device_pricing_requires_torch_engine() -> None:
    X = np.array([[1.0], [-1.0]])
    y = np.zeros(2)
    nu = np.ones(2)
    engine = make_engine(X, y)
    try:
        search_signed_atoms(
            engine,
            engine.scatter_vec(nu),
            nu,
            0.1,
            search_backend="device",
            n_random=0,
            n_subspace=0,
        )
    except ValueError as exc:
        assert "Torch engine" in str(exc)
    else:
        raise AssertionError("device search must reject a NumPy engine")


def test_predicted_coordinate_decrease() -> None:
    # min_z 1/2 ||r-z a||^2 + lam |z| from z=0.
    a = np.array([2.0, 0.0])
    r = np.array([3.0, 1.0])
    lam = 1.0
    corr = float(a @ r)
    sqnorm = float(a @ a)
    predicted = float(_predicted_decrease([corr], [sqnorm], lam)[0])
    z = (corr - lam) / sqnorm
    before = 0.5 * float(r @ r)
    after = 0.5 * float((r - z * a) @ (r - z * a)) + lam * abs(z)
    np.testing.assert_allclose(predicted, before - after, atol=1e-14)


def _host_features(engine, A_sh):
    return np.concatenate([engine._host64(A) for A in A_sh], axis=0)


def _check_retained_features(device, dtype, devices=None) -> None:
    rng = np.random.default_rng(31)
    X = rng.standard_normal((73, 5))
    y = rng.standard_normal(73)
    nu = rng.standard_normal(73)
    engine = make_engine(X, y, device=device, devices=devices, dtype=dtype)
    nu_sh = engine.scatter_vec(nu)
    U = rng.standard_normal((5, 5))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    stats = engine.rescore_finalists(
        U,
        nu_sh,
        retain_features=True,
        retained_feature_max_bytes=engine.feature_buffer_bytes(U.shape[1]),
    )
    assert stats["retained"]
    rebuilt = engine.build_atoms(U)
    for retained, independent in zip(stats["features"], rebuilt):
        np.testing.assert_array_equal(
            engine._host64(retained), engine._host64(independent)
        )
    corr, sqnorm = engine.rescore_stats(U, nu_sh)
    np.testing.assert_allclose(stats["corr"], corr, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(stats["sqnorm"], sqnorm, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        stats["aty"], engine.screen(rebuilt, engine.ys), rtol=2e-12, atol=2e-12
    )

    U0 = rng.standard_normal((5, 2))
    U0 /= np.linalg.norm(U0, axis=0, keepdims=True)
    ordinary = AtomLasso(engine)
    ordinary.set_atoms(U0)
    ordinary.append_atoms(U)
    reused = AtomLasso(engine)
    reused.set_atoms(U0)
    reused.append_atoms(
        U, A_sh=stats["features"], aty=stats["aty"], sqnorms=stats["sqnorm"]
    )
    b0, _, obj0, _ = ordinary.solve(0.25)
    b1, _, obj1, _ = reused.solve(0.25)
    np.testing.assert_allclose(obj1, obj0, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(b1, b0, rtol=2e-10, atol=2e-10)
    pred0 = _host_features(engine, ordinary.A) @ b0
    pred1 = _host_features(engine, reused.A) @ b1
    np.testing.assert_allclose(pred1, pred0, rtol=2e-10, atol=2e-10)


def test_retained_features_numpy_float64_float32() -> None:
    _check_retained_features("cpu", "float64")
    _check_retained_features("cpu", "float32")
    _check_retained_features("cpu", "float64", devices=["cpu"] * 3)
    _check_retained_features("cpu", "float32", devices=["cpu"] * 3)


def test_retained_features_torch_cpu_optional() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception:
        return
    _check_retained_features("torch:cpu", "float32")


def test_retained_features_torch_accelerators_optional() -> None:
    # Some ROCm installations report a device before the matching rocBLAS
    # kernels are usable, and that failure aborts the process rather than
    # raising Python.  Accelerator coverage is therefore explicit opt-in.
    if os.environ.get("RELU_CHAMBERS_TEST_ACCELERATOR") != "1":
        return
    try:
        import torch  # type: ignore
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    _check_retained_features("cuda:0", "float32")
    if torch.cuda.device_count() >= 2:
        _check_retained_features("cuda:0", "float32", devices=["cuda:0", "cuda:1"])


def test_oracle_reuse_alignment_and_cap_fallback() -> None:
    rng = np.random.default_rng(41)
    X = rng.standard_normal((96, 4))
    y = rng.standard_normal(96)
    nu = rng.standard_normal(96)
    engine = make_engine(X, y)
    nu_sh = engine.scatter_vec(nu)
    common = dict(
        lam=0.2,
        k=3,
        n_random=4,
        n_subspace=2,
        n_perturb=0,
        sketch_m=0,
        iters_sketch=3,
        iters_full=1,
        top_polish=3,
    )
    ordinary = search_signed_atoms(
        engine, nu_sh, nu, rng=np.random.default_rng(42), **common
    )
    reused = search_signed_atoms(
        engine,
        nu_sh,
        nu,
        rng=np.random.default_rng(42),
        retain_finalist_features=True,
        retained_feature_max_bytes=10**8,
        **common,
    )
    for sign in ("plus", "minus"):
        np.testing.assert_allclose(
            [value for value, _ in reused[sign]],
            [value for value, _ in ordinary[sign]],
            rtol=2e-12,
            atol=2e-12,
        )
    assert reused["feature_reuse"]["retained"]
    cache = reused["addition_cache"]
    Uadd = np.stack([entry[2] for entry in reused["addition"]], axis=1)
    independent = engine.build_atoms(Uadd)
    np.testing.assert_array_equal(
        _host_features(engine, cache["A_sh"]), _host_features(engine, independent)
    )
    np.testing.assert_allclose(
        cache["aty"], engine.screen(independent, engine.ys), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        cache["sqnorm"], engine.sqnorms(independent), rtol=2e-12, atol=2e-12
    )
    cache_dirs = [entry[2] for entry in reused["addition"]]
    fresh, positions = _dedup_cols_indexed(
        cache_dirs, np.zeros((4, 0)), normalize=False
    )
    cached_master = AtomLasso(engine)
    original_build_atoms = engine.build_atoms
    build_calls = 0

    def counted_build_atoms(U):
        nonlocal build_calls
        build_calls += 1
        return original_build_atoms(U)

    engine.build_atoms = counted_build_atoms
    cached_master.append_atoms(
        np.stack(fresh, axis=1),
        A_sh=engine.slice_cols(cache["A_sh"], positions),
        aty=np.asarray(cache["aty"])[positions],
        sqnorms=np.asarray(cache["sqnorm"])[positions],
    )
    engine.build_atoms = original_build_atoms
    assert build_calls == 0, "a retained append must not rebuild its columns"
    np.testing.assert_array_equal(
        _host_features(engine, cached_master.A),
        _host_features(engine, engine.build_atoms(cached_master.U)),
    )

    fallback = search_signed_atoms(
        engine,
        nu_sh,
        nu,
        rng=np.random.default_rng(42),
        retain_finalist_features=True,
        retained_feature_max_bytes=1,
        **common,
    )
    assert fallback["addition_cache"] is None
    assert fallback["feature_reuse"]["fallback_reason"] == "cap_exceeded"
    np.testing.assert_allclose(
        [entry[1] for entry in fallback["addition"]],
        [entry[1] for entry in ordinary["addition"]],
        rtol=2e-12,
        atol=2e-12,
    )


def test_reuse_preserves_oriented_dedup_and_appends_only_fresh() -> None:
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    y = np.ones(3)
    engine = make_engine(X, y)
    existing = np.array([[1.0], [0.0]])
    candidates = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0])]
    fresh, indices = _dedup_cols_indexed(candidates, existing)
    assert indices == [1, 2], "opposite orientation must survive dedup"
    Uall = np.stack(candidates, axis=1)
    stats = engine.rescore_finalists(
        Uall,
        engine.scatter_vec(y),
        retain_features=True,
        retained_feature_max_bytes=10**6,
    )
    master = AtomLasso(engine)
    master.set_atoms(existing)
    master.append_atoms(
        np.stack(fresh, axis=1),
        A_sh=engine.slice_cols(stats["features"], indices),
        aty=stats["aty"][indices],
        sqnorms=stats["sqnorm"][indices],
    )
    assert master.P == 3, "only the two fresh columns should be appended"
    np.testing.assert_array_equal(
        _host_features(engine, master.A),
        _host_features(engine, engine.build_atoms(master.U)),
    )


def test_end_to_end_reuse_and_tiny_cap_fallback() -> None:
    rng = np.random.default_rng(51)
    X = rng.standard_normal((120, 4))
    y = rng.standard_normal(120)
    solver = dict(
        lam=0.2,
        max_iter=3,
        add_per_round=2,
        certify_every=0,
        gated_every=0,
        final_verify=False,
        eps_heur=0,
        rng_seed=52,
    )
    oracle = dict(
        n_random=4,
        n_subspace=2,
        n_perturb=0,
        sketch_m=0,
        iters_sketch=3,
        iters_full=1,
        top_polish=3,
    )
    ordinary = solve_rcg(X, y, oracle_kwargs=oracle, **solver)
    reused = solve_rcg(
        X,
        y,
        oracle_kwargs={
            **oracle,
            "retain_finalist_features": True,
            "retained_feature_max_bytes": 10**8,
        },
        **solver,
    )
    fallback = solve_rcg(
        X,
        y,
        oracle_kwargs={
            **oracle,
            "retain_finalist_features": True,
            "retained_feature_max_bytes": 1,
        },
        **solver,
    )
    for candidate in (reused, fallback):
        np.testing.assert_allclose(
            candidate["obj"], ordinary["obj"], rtol=2e-10, atol=2e-10
        )
        np.testing.assert_array_equal(candidate["U"], ordinary["U"])
        pred = np.maximum(X @ candidate["U"], 0.0) @ candidate["beta"]
        reference = np.maximum(X @ ordinary["U"], 0.0) @ ordinary["beta"]
        np.testing.assert_allclose(pred, reference, rtol=2e-9, atol=2e-9)
        assert candidate["n_cols"] == ordinary["n_cols"]
    assert reused["feature_reuse"]["appended_reused_columns"] > 0
    assert reused["feature_reuse"]["appended_rebuilt_columns"] == 0
    assert fallback["feature_reuse"]["fallback_calls"] > 0
    assert fallback["feature_reuse"]["appended_rebuilt_columns"] > 0

    one_round = solve_rcg(
        X,
        y,
        lam=0.2,
        max_iter=1,
        certify_every=0,
        gated_every=0,
        final_verify=False,
        eps_heur=0,
        oracle_kwargs={
            **oracle,
            "retain_finalist_features": True,
            "retained_feature_max_bytes": 10**8,
        },
    )
    assert one_round["feature_reuse"]["retained_calls"] == 0
    assert one_round["feature_reuse"]["appended_reused_columns"] == 0
    assert one_round["n_cols"] == 0, "the last round must not append unused columns"


def run() -> bool:
    test_oriented_direction_dedup()
    test_opposite_atom_improves_exact_master()
    test_ascent_trial_modes()
    test_torch_cpu_ascent_equivalence()
    test_torch_cpu_device_pricing_smoke()
    test_device_pricing_requires_torch_engine()
    test_predicted_coordinate_decrease()
    test_retained_features_numpy_float64_float32()
    test_retained_features_torch_cpu_optional()
    test_retained_features_torch_accelerators_optional()
    test_oracle_reuse_alignment_and_cap_fallback()
    test_reuse_preserves_oriented_dedup_and_appends_only_fresh()
    test_end_to_end_reuse_and_tiny_cap_fallback()
    print("solver focused tests: OK")
    return True


if __name__ == "__main__":
    run()
