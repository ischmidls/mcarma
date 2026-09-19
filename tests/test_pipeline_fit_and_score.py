"""Golden regression for mcarma.pipeline.fit_and_score -- the shared
fit -> pure-ll AICc -> SE composer. The two things under test are the scoring
(AICc must be computed on loglik_pure, NOT the penalized loglik) and the SE
dispatch (se="hess_inv_diag" reproduces the historical LSST raw BFGS
inverse-Hessian diagonal exactly, None when the fit exposed no hess_inv; any
other se value routes through standard_errors and reports sqrt(diag(cov))).

``fit`` is stubbed with a canned result so no optimizer/scipy/jax runs -- the
composition arithmetic is what's asserted, against ``_legacy_lsst_score`` lifted
verbatim from fit_lsst_downsample.fit_one (pre-refactor).

Run: PYTHONUTF8=1 python tests/test_pipeline_fit_and_score.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcarma.pipeline as PL  # noqa: E402
from mcarma.model_utils import compute_aicc, count_params  # noqa: E402


class _Data:
    def __init__(self, d, n):
        self.d = d
        self.n = n


# ---- frozen legacy per-order scoring from fit_lsst_downsample.fit_one --------
def _legacy_lsst_score(result, d, p, q, n_obs):
    k = count_params(d, p, q)
    ll_pen = result["loglik"]
    ll = result.get("loglik_pure", ll_pen)
    aicc_val = compute_aicc(ll, k, n_obs)
    hess_inv = result.get("hess_inv")
    se_theta = (np.sqrt(np.maximum(np.diag(np.asarray(hess_inv)), 0)).tolist()
                if hess_inv is not None else None)
    return {
        "ll": float(ll), "ll_pen": float(ll_pen), "k": int(k),
        "aicc": float(aicc_val), "se_theta": se_theta,
    }


def _canned_result(with_hess=True, with_pure=True):
    d, p, q = 5, 2, 1
    ntheta = 2 + 2 + 1 + d + (d * (d + 1)) // 2  # arbitrary consistent length
    theta = np.linspace(-2.0, 1.0, ntheta)
    res = {
        "theta": theta,
        "loglik": -1234.56,             # penalized MAP objective
        "mu": np.zeros(d),
        "Sigma": np.eye(d),
        "success": True,
    }
    if with_pure:
        res["loglik_pure"] = -1230.10   # pure ll (higher: penalty removed)
    if with_hess:
        H = np.diag(np.linspace(0.01, 0.09, ntheta))
        H[0, 1] = H[1, 0] = 0.001       # off-diag ignored by the diagonal SE
        res["hess_inv"] = H
    return d, p, q, res


def _with_stub_fit(res, fn):
    old = PL.fit
    PL.fit = lambda *a, **k: res
    try:
        return fn()
    finally:
        PL.fit = old


def test_hess_inv_diag_matches_legacy():
    for with_hess in (True, False):
        for with_pure in (True, False):
            d, p, q, res = _canned_result(with_hess, with_pure)
            data = _Data(d, n=400)
            fs = _with_stub_fit(res, lambda: PL.fit_and_score(
                data, p, q, se="hess_inv_diag"))
            leg = _legacy_lsst_score(res, d, p, q, data.n)
            assert fs["ll"] == leg["ll"], (fs["ll"], leg["ll"])
            assert fs["ll_pen"] == leg["ll_pen"]
            assert fs["k"] == leg["k"]
            assert fs["aicc"] == leg["aicc"]
            assert fs["se_theta"] == leg["se_theta"], (with_hess, with_pure)
            # AICc must be on the PURE ll, so it must differ from AICc-on-penalized
            # whenever loglik_pure is present.
            if with_pure:
                assert fs["aicc"] != compute_aicc(res["loglik"], leg["k"], data.n)
            exp_backend = "bfgs_hess_inv_diag" if with_hess else None
            assert fs["se_backend"] == exp_backend
            print(f"OK  hess_inv_diag  hess={with_hess!s:5s} pure={with_pure!s:5s} "
                  f"aicc={fs['aicc']:.4f}")


def test_standard_errors_routing():
    """Non-diag se values route through standard_errors; se_theta = sqrt(diag(cov))."""
    d, p, q, res = _canned_result()
    data = _Data(d, n=400)
    cov = np.diag(np.linspace(0.04, 0.16, len(res["theta"])))
    theta_se = np.asarray(res["theta"]) + 0.001

    def _stub_se(theta, dat, pp, qq, **kw):
        assert kw["source"] == "polish"
        return {"theta_se": theta_se, "cov": cov, "pd": True,
                "se_backend": "jax_repolish+jax_hvp_hess", "grad_norm_map": 1e-3}

    old_se = PL.standard_errors
    PL.standard_errors = _stub_se
    try:
        fs = _with_stub_fit(res, lambda: PL.fit_and_score(
            data, p, q, se="polish"))
    finally:
        PL.standard_errors = old_se

    assert fs["se_backend"] == "jax_repolish+jax_hvp_hess"
    assert fs["pd"] is True
    assert fs["grad_norm_map"] == 1e-3
    assert np.array_equal(np.asarray(fs["theta_se"]), theta_se)
    exp = np.sqrt(np.maximum(np.diag(cov), 0)).tolist()
    assert fs["se_theta"] == exp
    # AICc scored on pure ll regardless of SE source.
    assert fs["aicc"] == compute_aicc(res["loglik_pure"], count_params(d, p, q),
                                      data.n)
    print("OK  polish routing -> standard_errors, se_theta=sqrt(diag(cov))")


if __name__ == "__main__":
    test_hess_inv_diag_matches_legacy()
    test_standard_errors_routing()
    print("OK: fit_and_score composes fit -> pure-ll AICc -> SE correctly")
