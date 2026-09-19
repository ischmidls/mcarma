"""Guards on mcarma.rts.reconstruct_band -- the shared refit-free RTS-smoother
reconstruction used by the bigindep and LSST figure scripts. Checks the
properties a reconstruction must have, so a regression in the smoother wiring is
caught before it silently reshapes a figure:

  * output arrays are finite and aligned to the dense evaluation grid;
  * the smoother posterior std is SMALLER near observed times than in an
    unobserved gap (the smoother pins to data and widens without it);
  * the reconstruction is deterministic (same inputs -> identical output).

Imports mcarma.rts, which loads JAX via the smoother, so this test needs a JAX
environment (a compute node or a laptop with JAX). Pure synthetic data, no fit.
Run: PYTHONUTF8=1 python tests/test_rts_reconstruction.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcarma.optimizer_utils import build_state_space  # noqa: E402
from mcarma.rts import reconstruct_band               # noqa: E402


def _state_space(d=2, zeta=0.4, wn=0.08, rho=0.6):
    ar = [[(2.0 * zeta * wn, wn ** 2)] for _ in range(d)]
    ma = [[(1.0 / (2.5 * wn),)] for _ in range(d)]
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return build_state_space(ar, ma, Sigma, d, 2, 1)


def _obs_with_gap(d=2, ref=1, seed=0):
    """Two observed windows [0,80] and [160,240] with an empty gap (80,160)."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.sort(rng.uniform(0, 80, 60)),
                        np.sort(rng.uniform(160, 240, 60))])
    band = np.full(t.size, ref, dtype=int)          # single reference band
    R = np.full(t.size, 0.02 ** 2)
    y = 0.3 * np.sin(2 * np.pi * t / 90.0) + rng.normal(0, 0.02, t.size)
    y -= y.mean()
    return t, y, band, R


def test_reconstruct_shape_and_finite():
    F, G, H, V = _state_space()
    t, y, band, R = _obs_with_gap(ref=1)
    t_eval = np.linspace(0, 240, 300)
    out = reconstruct_band(t, y, band, R, F, G, H, V, ref_band=1, t_eval=t_eval)
    assert out["yhat"].shape == t_eval.shape
    assert out["std"].shape == t_eval.shape
    assert np.isfinite(out["yhat"]).all() and np.isfinite(out["std"]).all()
    assert (out["std"] > 0).all()
    assert out["t_obs"].size == t.size          # all reference-band obs returned
    print("OK  reconstruct_band shapes finite, std positive")


def test_std_widens_in_gap():
    """Posterior std is larger mid-gap than near the observed windows."""
    F, G, H, V = _state_space()
    t, y, band, R = _obs_with_gap(ref=1)
    t_eval = np.linspace(0, 240, 300)
    out = reconstruct_band(t, y, band, R, F, G, H, V, ref_band=1, t_eval=t_eval)
    near_obs = out["std"][(t_eval > 20) & (t_eval < 60)].mean()
    in_gap = out["std"][(t_eval > 100) & (t_eval < 140)].mean()
    assert in_gap > near_obs, (in_gap, near_obs)
    print(f"OK  std widens in gap ({in_gap:.4f} > {near_obs:.4f})")


def test_reconstruct_deterministic():
    F, G, H, V = _state_space()
    t, y, band, R = _obs_with_gap(ref=0)
    t_eval = np.linspace(0, 240, 200)
    a = reconstruct_band(t, y, band, R, F, G, H, V, ref_band=0, t_eval=t_eval)
    b = reconstruct_band(t, y, band, R, F, G, H, V, ref_band=0, t_eval=t_eval)
    assert np.array_equal(a["yhat"], b["yhat"])
    assert np.array_equal(a["std"], b["std"])
    print("OK  reconstruct_band deterministic")


if __name__ == "__main__":
    test_reconstruct_shape_and_finite()
    test_std_widens_in_gap()
    test_reconstruct_deterministic()
    print("OK: rts reconstruction invariants hold")
