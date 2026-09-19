"""Curvature preconditioning: the scales, the change of variables, the round trip.

These are deliberately synthetic. The claim under test is the linear algebra,
not the astrophysics: that optimizing in ``theta = scales * z`` is the SAME
problem with a better-conditioned Hessian, and that everything read off the
result afterwards (the point, the gradient, the inverse Hessian the standard
errors come from) is mapped back into theta space correctly.

The stretched quadratic below is built the way the real surface is stretched: a
well-conditioned intrinsic shape wrapped in badly scaled coordinates. That is
what the measured Hessians look like, twelve decades of curvature spread that
collapse to about six once rescaled, and it is the case preconditioning is
supposed to fix. A surface that is intrinsically ill conditioned is NOT fixable
this way, which is why the collapsed-Sigma fits stay ill conditioned and why
test_intrinsic_ill_conditioning_is_not_hidden pins that down.
"""
import numpy as np
import pytest
from scipy.optimize import minimize, OptimizeResult

from mcarma.fit import (PRECOND_CLAMP, curvature_scales,
                        precondition_objective, unscale_result)


def _stretched_quadratic(n=12, decades=6.0, seed=0):
    """0.5 * r.A.r with A badly SCALED but well SHAPED, r = x - x_star.

    A = S M S, where M is a random SPD matrix with condition number of order 10
    and S = diag(10^u) spreads the coordinates over 2*decades of curvature.
    cond(A) is then enormous while the problem underneath it is easy, which is
    exactly the situation the preconditioner targets.
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, n))
    M = B @ B.T / n + np.eye(n)
    u = np.linspace(-decades, decades, n)
    S = np.diag(10.0 ** u)
    A = S @ M @ S
    A = 0.5 * (A + A.T)
    x_star = rng.normal(size=n) / (10.0 ** u)

    def fun(x):
        r = np.asarray(x, float) - x_star
        return 0.5 * float(r @ A @ r), A @ r

    return fun, A, x_star


# ------------------------------------------------------------ scale building
def test_scales_are_inverse_sqrt_curvature_normalized_to_unit_median():
    h = np.array([1e-4, 1e-2, 1.0, 1e2, 1e4])   # inside the clamp
    s = curvature_scales(h)
    raw = 1.0 / np.sqrt(h)
    assert np.allclose(s, raw / np.median(raw))
    assert np.isclose(np.median(s), 1.0)


def test_sign_of_the_curvature_is_ignored():
    """A saddle direction still needs a step scale; only magnitude matters."""
    assert np.allclose(curvature_scales(np.array([-4.0, -1.0, -0.25])),
                       curvature_scales(np.array([4.0, 1.0, 0.25])))


def test_clamp_bounds_the_spread():
    s = curvature_scales(np.array([1e-40, 1.0, 1e40]), clamp=1e3)
    assert s.min() >= 1.0 / 1e3 - 1e-12
    assert s.max() <= 1e3 + 1e-12


def test_default_clamp_does_not_bite_on_a_realistic_spread():
    """The measured interior spread must survive the clamp uncorrected.

    Curvature spread on the interior fits is 11.86 decades, which is 5.93
    decades of step scale, i.e. +/- 2.97 about the median. The default clamp
    allows +/- 3. So it is not binding on a healthy fit, but only just: a
    near-floor fit (13.4 decades measured) DOES get clipped, and a collapsed one
    is clipped hard. That is the intended behaviour, not an accident, so the
    boundary is pinned here rather than left to drift.
    """
    interior = 10.0 ** np.linspace(-5.93, 5.93, 45)
    s = curvature_scales(interior, clamp=PRECOND_CLAMP)
    assert s.min() > 1.0 / PRECOND_CLAMP
    assert s.max() < PRECOND_CLAMP

    nearfloor = 10.0 ** np.linspace(-6.72, 6.72, 45)   # 13.4 decades
    s2 = curvature_scales(nearfloor, clamp=PRECOND_CLAMP)
    assert np.isclose(s2.max(), PRECOND_CLAMP)         # the guard engages


@pytest.mark.parametrize("bad", [np.zeros(5),
                                 np.full(5, np.nan),
                                 np.full(5, np.inf)])
def test_unusable_curvature_degrades_to_no_preconditioning(bad):
    assert np.allclose(curvature_scales(bad), 1.0)


def test_a_single_bad_entry_falls_back_to_the_median_not_a_huge_step():
    """The collapsing-Sigma guard.

    A coordinate whose curvature has gone to zero must not be handed an
    enormous step and allowed to dominate the search. It gets the median.
    """
    s = curvature_scales(np.array([1.0, 1.0, 0.0, 1.0, 1.0]))
    assert np.allclose(s, 1.0)
    assert np.isfinite(s).all()


# ------------------------------------------------------- change of variables
def test_preconditioned_objective_is_the_same_function_reparameterized():
    fun, _, _ = _stretched_quadratic()
    s = np.array([2.0 ** k for k in range(12)])
    g = precondition_objective(fun, s)
    z = np.linspace(-1.0, 1.0, 12)
    val_z, grad_z = g(z)
    val_t, grad_t = fun(s * z)
    assert np.isclose(val_z, val_t)                 # values identical
    assert np.allclose(grad_z, s * grad_t)          # chain rule


def test_preconditioned_gradient_matches_finite_differences():
    fun, _, _ = _stretched_quadratic(n=6, decades=2.0)
    s = curvature_scales(np.array([1e-3, 1e-1, 1.0, 10.0, 1e2, 1e4]))
    g = precondition_objective(fun, s)
    z = np.full(6, 0.1)
    _, grad = g(z)
    fd = np.empty(6)
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1e-6
        fd[i] = (g(z + e)[0] - g(z - e)[0]) / 2e-6
    assert np.allclose(grad, fd, rtol=1e-4, atol=1e-8)


# ------------------------------------------------------------- the round trip
def test_unscale_maps_point_gradient_and_inverse_hessian():
    s = np.array([0.5, 2.0, 4.0])
    hz = np.array([[2.0, 0.1, 0.0],
                   [0.1, 3.0, 0.2],
                   [0.0, 0.2, 5.0]])
    res = OptimizeResult(x=np.array([1.0, 1.0, 1.0]),
                         jac=np.array([1.0, 1.0, 1.0]),
                         hess_inv=hz.copy())
    out = unscale_result(res, s)
    assert np.allclose(out.x, s)                          # theta = s * z
    assert np.allclose(out.jac, 1.0 / s)                  # grad f = grad g / s
    assert np.allclose(out.hess_inv, s[:, None] * hz * s[None, :])


def test_inverse_hessian_round_trip_recovers_the_true_one():
    """The SE path, end to end.

    H_z = diag(s) H_theta diag(s), so unscaling the inverse of H_z must give
    back the inverse of H_theta. Getting this wrong leaves every fitted value
    right and every standard error silently rescaled, which is the quiet
    failure mode.
    """
    _, A, _ = _stretched_quadratic(n=8, decades=3.0, seed=3)
    s = curvature_scales(np.diag(A))
    H_z = (s[:, None] * A) * s[None, :]
    res = OptimizeResult(x=np.zeros(8), hess_inv=np.linalg.inv(H_z))
    out = unscale_result(res, s)
    assert np.allclose(out.hess_inv, np.linalg.inv(A), rtol=1e-6)


def test_unscale_tolerates_a_missing_or_odd_hess_inv():
    s = np.array([1.0, 2.0, 3.0])
    assert np.allclose(unscale_result(OptimizeResult(x=np.ones(3)), s).x, s)
    unscale_result(OptimizeResult(x=np.ones(3), hess_inv="nope"), s)  # no raise


# ------------------------------------------------------------------ the point
def test_preconditioning_rescues_a_search_that_otherwise_hits_the_cap():
    """The whole reason this exists.

    Same objective, same start, same iteration cap. Raw BFGS runs out of
    iterations a long way from the optimum; preconditioned BFGS converges well
    inside the budget and lands on the true minimizer.
    """
    # n=45 is the real thing: d=6, (p,q)=(2,1) gives 12 AR + 6 MA + 21 chol
    # + 6 mu = 45 free parameters.
    fun, A, x_star = _stretched_quadratic(n=45, decades=6.0, seed=1)
    x0 = np.zeros(45)
    cap = 300

    raw = minimize(fun, x0, jac=True, method="BFGS",
                   options={"maxiter": cap, "gtol": 1e-5})

    s = curvature_scales(np.diag(A))
    pre = minimize(precondition_objective(fun, s), x0 / s, jac=True,
                   method="BFGS", options={"maxiter": cap, "gtol": 1e-5})
    pre = unscale_result(pre, s)

    assert raw.nit >= cap, "fixture is not actually hard for raw BFGS"
    assert pre.nit < cap
    assert pre.fun < raw.fun
    # and it is the right point, measured in the metric the problem lives in
    w = np.sqrt(np.abs(np.diag(A)))
    assert np.allclose(pre.x * w, x_star * w, atol=1e-3)


def test_raw_bfgs_can_report_success_at_a_point_that_is_not_the_optimum():
    """Why "converged" is not the metric to trust on a stretched surface.

    On a smaller version of the same fixture raw BFGS stops with success=True
    while still far from the minimum: the gradient is tiny in the badly scaled
    directions, so the gtol test is satisfied at a point that is not the answer.
    This is the same mechanism that lets a collapsed Sigma fit pass the
    stationarity check, and it is the reason the acceptance test is kept in
    theta coordinates on the plain likelihood rather than read off this flag.
    """
    fun, A, _ = _stretched_quadratic(n=12, decades=6.0, seed=1)
    raw = minimize(fun, np.zeros(12), jac=True, method="BFGS",
                   options={"maxiter": 300, "gtol": 1e-5})
    assert raw.success
    assert raw.fun > 1e-3            # "successful" and nowhere near zero

    s = curvature_scales(np.diag(A))
    pre = minimize(precondition_objective(fun, s), np.zeros(12) / s, jac=True,
                   method="BFGS", options={"maxiter": 300, "gtol": 1e-5})
    assert pre.fun < 1e-8


def test_conditioning_actually_improves():
    """The measured claim, on a controlled surface: cond(H) falls by decades."""
    _, A, _ = _stretched_quadratic(n=12, decades=6.0, seed=2)
    s = curvature_scales(np.diag(A))
    A_pre = (s[:, None] * A) * s[None, :]
    assert np.log10(np.linalg.cond(A_pre)) < np.log10(np.linalg.cond(A)) - 6.0


def test_intrinsic_ill_conditioning_is_not_hidden():
    """The red-flag test, kept as a test.

    Diagonal rescaling can only remove ill conditioning that lives in the
    SCALING. A surface that is intrinsically flat in some direction, which is
    what a collapsing Sigma produces, must stay ill conditioned afterwards. If
    this ever started passing, the preconditioner would be making degenerate
    fits look converged, which is worse than not fixing them at all.
    """
    n = 8
    rng = np.random.default_rng(7)
    Qm, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Qm @ np.diag(np.concatenate([np.ones(n - 1), [1e-14]])) @ Qm.T
    A = 0.5 * (A + A.T)
    s = curvature_scales(np.diag(A))
    A_pre = (s[:, None] * A) * s[None, :]
    assert np.log10(np.linalg.cond(A_pre)) > 10.0
