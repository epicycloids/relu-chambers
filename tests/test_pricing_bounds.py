"""Focused exact-form tests for the low-rank SDP bound machinery."""

from __future__ import annotations

import numpy as np

from relu_chambers.pricing_bounds import _dual_value_grad_host, _valid_bound_host


def test_joint_one_dimensional_positive_control() -> None:
    # X=[1], nu=[1] gives augmented rows B=[1;1].  With c=(1,1),
    # M=2, C=2, so the recovered SDP upper bound is exactly four.
    B = np.ones((2, 1))
    Z = np.ones((1, 1))
    bound = _valid_bound_host(B, Z)
    np.testing.assert_allclose(bound, 4.0, rtol=0.0, atol=1e-14)
    assert 0.5 * np.sqrt(bound) >= 1.0


def test_small_cone_gradient_finite_difference() -> None:
    rng = np.random.default_rng(2)
    B = rng.standard_normal((7, 3))
    A = rng.standard_normal((3, 3))
    Z = A @ A.T + np.eye(3)
    H = rng.standard_normal((3, 3))
    H = 0.5 * (H + H.T)
    value, grad = _dual_value_grad_host(B, Z)
    step = 1e-6
    plus, _ = _dual_value_grad_host(B, Z + step * H)
    minus, _ = _dual_value_grad_host(B, Z - step * H)
    directional = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        directional, float(np.sum(grad * H)), rtol=2e-6, atol=2e-6
    )
    assert np.isfinite(value)


def run() -> bool:
    test_joint_one_dimensional_positive_control()
    test_small_cone_gradient_finite_difference()
    print("pricing-bound focused tests: OK")
    return True


if __name__ == "__main__":
    run()
