"""`mcarma.fit` is the two-stage estimator, wired the way the papers ran it.

No optimization runs here: both stages are replaced by recorders, so the test
checks only what each stage is called with and how the result is assembled.
"""
import numpy as np

import mcarma
from mcarma import estimate
from mcarma.observation import ObservationData


def _data():
    rng = np.random.default_rng(0)
    n = 40
    t = np.sort(rng.uniform(0, 1000, n))
    band = np.arange(n) % 2
    y = np.where(band == 0, 19.0, 20.5) + 0.1 * rng.standard_normal(n)
    return ObservationData(t, y, band, np.full(n, 0.01), 2)


def test_two_stage_wiring(monkeypatch):
    d, p, q = 2, 1, 0
    calls = {}
    theta1 = np.zeros(d * p + d * (d + 1) // 2 + d)

    def fake_warm(t, y, band, R, **kw):
        calls["warm_y"] = np.asarray(y)
        return theta1.copy()

    def fake_stage1(data, p_, q_, **kw):
        calls["stage1"] = kw
        calls["stage1_y"] = np.asarray(data.y_obs)
        return {"theta": theta1.copy()}

    def fake_finish(data, p_, q_, theta_start, **kw):
        calls["finish"] = kw
        return {"theta": theta_start + 0.1, "ll": -12.5, "ll_start": -13.0,
                "converged": True, "grad_norm": 1e-6, "attempt": "jax_multi",
                "trail": [], "fit_result": {"hess_inv": np.eye(theta1.size)}}

    monkeypatch.setattr(estimate, "build_perband_warm_theta", fake_warm)
    monkeypatch.setattr(estimate, "fit_penalized", fake_stage1)
    monkeypatch.setattr(estimate, "finish_mle", fake_finish)

    data = _data()
    res = mcarma.fit(data, p, q)

    # Stage 1 carries the production penalties and the loading.
    s1 = calls["stage1"]
    assert s1["ar_band_lambda"] == 1e3 and s1["ma_band_lambda"] == 1e3
    assert s1["ar_damping_lambda"] == 1e3
    assert s1["diag_load_lambda"] == 0.05
    assert s1["chol_ridge_lambda"] == 10.0 and s1["chol_ridge_onesided"]
    assert s1["warm_theta"] is not None

    # Stage 2 is the ordinary likelihood: explicit zeros, not the None hinge.
    s2 = calls["finish"]
    assert s2["fit_kwargs"] == {"ar_band_lambda": 0.0, "ma_band_lambda": 0.0}
    assert s2["precondition"] == "curvature"

    # Both stages see centered bands; the reported means are shifted back.
    y, band = np.asarray(data.y_obs), np.asarray(data.band)
    for b in range(d):
        assert abs(calls["stage1_y"][band == b].mean()) < 1e-12
        assert np.isclose(res["mu"][b, 0], y[band == b].mean() + 0.1)

    assert res["loglik"] == res["loglik_pure"] == -12.5
    assert res["loglik_stage1"] == -13.0
    assert res["converged"] is True
    assert res["k"] == theta1.size
    assert np.isfinite(res["aicc"])
    assert res["F"].shape == (d * p, d * p)


def test_fit_penalized_is_the_single_stage_optimizer():
    from mcarma.fit import fit as single_stage
    assert mcarma.fit_penalized is single_stage
    assert mcarma.fit is estimate.fit
