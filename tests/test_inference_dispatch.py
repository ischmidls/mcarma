"""Golden regression for mcarma.inference.standard_errors -- the ONE new piece of
logic in the SE unification (the estimator bodies polished_cov/mle_cov/_jax_se_objs
were relocated verbatim from sim_study, so they are byte-identical by construction and
are not re-tested here). This asserts the dispatcher reproduces sim_study's historical
``--se-source`` branch (the frozen ``_legacy_dispatch`` below, lifted verbatim from
sim_study.py run_task) exactly: same theta_se, cov, pd, se_backend, grad_norm_map, and
the same underlying polished_cov/mle_cov calls with the same arguments.

We stub polished_cov and mle_cov with recording sentinels so the branch logic + arg
forwarding is what's under test (fast, deterministic, no fit/scipy/jax). The real
hess_inv arithmetic (symmetrize + PD check) is exercised numerically.

Run: PYTHONUTF8=1 python tests/test_inference_dispatch.py"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcarma.inference as I  # noqa: E402


# ---- frozen legacy inline block from sim_study.run_task (pre-refactor) -------
# Verbatim, but with polished_cov/mle_cov passed in so the SAME stubs drive both.
def _legacy_dispatch(theta, data, p, q, res, prior_kwargs, se_source, use_jax,
                     polished_cov, mle_cov):
    out = {}
    cov, pd = None, False
    theta_se = theta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if se_source == "polish":
            theta_se, cov, pd, gn, se_backend = polished_cov(
                theta, data, p, q, prior_kwargs, use_jax=use_jax,
                hess_method="rich")
            out["se_backend"] = se_backend
            out["grad_norm_map"] = float(gn)
            out["theta"] = np.asarray(theta_se).tolist()
        elif se_source == "hess_inv":
            hi = res.get("hess_inv")
            if hi is not None:
                cov = np.asarray(hi, dtype=float)
                cov = 0.5 * (cov + cov.T)
                pd = bool(np.all(np.linalg.eigvalsh(cov) > 0))
                out["se_backend"] = "bfgs_hess_inv"
        if cov is None:
            cov, pd = mle_cov(theta_se, data, p, q, prior_kwargs, use_jax=use_jax)
            out["se_backend"] = "fd_approx_mle_cov"
    out["theta_se"] = np.asarray(theta_se)
    out["cov"] = cov
    out["pd"] = bool(pd)
    return out


# ---- recording stubs --------------------------------------------------------
class _Rec:
    def __init__(self, ret):
        self.ret = ret
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.ret


def _run_new(theta, data, p, q, res, pk, src, use_jax, pc, mc):
    """Drive standard_errors with the same stubs, mapping its dict to the legacy
    out-shape for comparison."""
    old_pc, old_mc = I.polished_cov, I.mle_cov
    I.polished_cov, I.mle_cov = pc, mc
    try:
        se = I.standard_errors(theta, data, p, q, result=res, prior_kwargs=pk,
                               source=src, use_jax=use_jax, hess_method="rich")
    finally:
        I.polished_cov, I.mle_cov = old_pc, old_mc
    out = {"theta_se": np.asarray(se["theta_se"]), "cov": se["cov"],
           "pd": bool(se["pd"]), "se_backend": se["se_backend"]}
    if se["grad_norm_map"] is not None:
        out["grad_norm_map"] = float(se["grad_norm_map"])
        out["theta"] = np.asarray(se["theta_se"]).tolist()
    return out


def _assert_same(a, b):
    assert set(a.keys()) == set(b.keys()), (sorted(a), sorted(b))
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), (k, va, vb)
        else:
            assert va == vb, (k, va, vb)


def _assert_calls_same(r_new, r_leg):
    assert len(r_new.calls) == len(r_leg.calls), (r_new.calls, r_leg.calls)
    for (an, kn), (al, kl) in zip(r_new.calls, r_leg.calls):
        # positional args: theta,data,p,q,prior_kwargs (compare elementwise)
        assert len(an) == len(al)
        for x, y in zip(an, al):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                assert np.array_equal(np.asarray(x), np.asarray(y))
            else:
                assert x == y, (x, y)
        assert kn == kl, (kn, kl)


THETA = np.array([-0.7, -0.5, -2.3, 0.02, -2.3, 0.1, -0.1])
DATA = object()          # never dereferenced by stubs
PK = {"chol_ridge_lambda": 1.0, "ar_band_lambda": 1.0, "ma_band_lambda": 1.0}
P, Q = 1, 0


def _cases():
    # (source, res, use_jax)
    M = np.array([[2.0, 0.3], [0.3, 1.5]])       # PD -> hess_inv usable
    Mnpd = np.array([[1.0, 2.0], [2.0, 1.0]])    # indefinite
    return [
        ("polish", {}, False),
        ("polish", {}, True),
        ("auto", {}, True),                       # auto -> polish
        ("hess_inv", {"hess_inv": M}, False),     # PD hess_inv path
        ("hess_inv", {"hess_inv": Mnpd}, False),  # non-PD hess_inv (pd False)
        ("hess_inv", {}, False),                  # missing -> mle_cov fallthrough
        ("exact", {}, False),
    ]


def test_dispatch_matches_legacy():
    for src, res, uj in _cases():
        # fresh stubs per case; sentinel returns are distinguishable
        pc_ret = (np.array([-0.71, -0.49, -2.3, 0.02, -2.3, 0.1, -0.1]),
                  np.eye(7) * 0.5, True, 1.23e-3, "jax_repolish+jax_hvp_hess")
        mc_ret = (np.eye(7) * 0.9, True)
        pc_new, pc_leg = _Rec(pc_ret), _Rec(pc_ret)
        mc_new, mc_leg = _Rec(mc_ret), _Rec(mc_ret)

        # NEW path uses "auto" as-is; legacy has no "auto" -> map auto->polish for
        # the frozen reference (that IS the documented meaning of auto).
        leg_src = "polish" if src == "auto" else src
        new = _run_new(THETA, DATA, P, Q, res, PK, src, uj, pc_new, mc_new)
        leg = _legacy_dispatch(THETA, DATA, P, Q, res, PK, leg_src, uj,
                               pc_leg, mc_leg)
        _assert_same(new, leg)
        _assert_calls_same(pc_new, pc_leg)
        _assert_calls_same(mc_new, mc_leg)
        print(f"OK  source={src:9s} use_jax={uj!s:5s} "
              f"backend={new['se_backend']}")


if __name__ == "__main__":
    test_dispatch_matches_legacy()
    print("OK: standard_errors reproduces the legacy --se-source dispatch")
