"""
examples/sdss_stripe82_demo.py
==============================
A self-contained worked example: fit a multivariate CARMA (mCARMA) model to a
real SDSS Stripe 82 quasar light curve with the `mcarma` package, select the
model order by AICc, and plot the light curve alongside the fitted power
spectral density (PSD) of each band.

This is the "SDSS demo / appendix" for the shipped package — it exercises the
public API end-to-end on the bundled example object (examples/data/1000679, a
5-band ugriz light curve) with no dependency on the research tree
(fits/, HPC, JAX). It uses the same MAP prior stack as the production pipeline
(fits/process_quasars/process_quasars.py).

Timing: this runs finite-difference BFGS from random starts, which is the slow
way to do it. `--quick` (the (1,0) fit alone) takes a few minutes on the bundled
object, and the full (1,0)/(2,0)/(2,1) ladder several times that. Lower
`--n-restarts` for a rougher, faster look. For the fast version of the same
fit, see cross_band_coherence.py, which warm starts from per-band fits and uses
the analytic JAX gradient.

Run:
    python examples/sdss_stripe82_demo.py                # full (1,0)/(2,0)/(2,1) ladder
    python examples/sdss_stripe82_demo.py --quick        # (1,0) only, fastest
    python examples/sdss_stripe82_demo.py --obj PATH     # a different Stripe 82 file

Output:
    examples/sdss_stripe82_demo.png   (light curve + selected-model PSD)
    an AICc order-selection table printed to stdout

Note on scale: production fits use analytic JAX gradients on a compute node
(`--use-jax-grad`, see the run-real-data workflow) for speed and accuracy; this
demo uses finite-difference BFGS so it is dependency-free but slower. The
science is identical.
"""
import argparse
import os
import sys

# Allow running in-repo without `pip install -e .` (adds the repo root, the
# parent of this examples/ dir, so `import mcarma` resolves).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcarma import ObservationData, fit, compute_aicc, mcarma_psd

from stripe82 import (N_BANDS, BAND_LABELS, BAND_COLORS, HERE, DEFAULT_OBJ,
                      load_stripe82, production_prior_kwargs, n_params)

OUT_PNG = os.path.join(HERE, "sdss_stripe82_demo.png")


def fit_order(data, y, band, p, q, n_restarts, seed=0):
    """Fit one mCARMA(p, q) order with the production priors; return (result, aicc)."""
    priors = production_prior_kwargs(y, band, data.d)
    res = fit(data, p, q, n_restarts=n_restarts, seed=seed, **priors)
    # AICc is scored on the PURE (unpenalized) maximized log-likelihood.
    aicc = compute_aicc(res["loglik_pure"], n_params(data.d, p, q), data.n)
    return res, aicc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="path to a Stripe 82 file")
    ap.add_argument("--quick", action="store_true", help="fit only (1,0) (fastest)")
    ap.add_argument("--n-restarts", type=int, default=8,
                    help="random restarts per order (default 8; production uses ~20)")
    ap.add_argument("--out", default=OUT_PNG, help="output figure path")
    args = ap.parse_args()

    t, y, band, R = load_stripe82(args.obj)
    d = N_BANDS
    data = ObservationData(t, y, band, R, d=d)
    obj_name = os.path.basename(args.obj)
    print(f"Loaded {obj_name}: {data.n} epochs over {t.max()-t.min():.0f} days, "
          f"per-band counts = {[int((band==b).sum()) for b in range(d)]}")

    orders = [(1, 0)] if args.quick else [(1, 0), (2, 0), (2, 1)]
    results = {}
    print("\nFitting the order ladder (this runs finite-difference BFGS locally)...")
    for (p, q) in orders:
        res, aicc = fit_order(data, y, band, p, q, args.n_restarts)
        results[(p, q)] = (res, aicc)
        print(f"  mCARMA({p},{q}): loglik={res['loglik_pure']:.2f}  "
              f"k={n_params(d, p, q)}  AICc={aicc:.2f}")

    best = min(results, key=lambda k: results[k][1])
    best_res = results[best][0]
    print(f"\nAICc-selected order: mCARMA{best}")

    # --- figure: light curve (left) + selected-model per-band PSD (right) ----
    fig, (ax_lc, ax_psd) = plt.subplots(1, 2, figsize=(14, 5))

    for b in range(d):
        m = band == b
        ax_lc.errorbar(t[m], y[m], yerr=np.sqrt(R[m]), fmt="o", ms=3, lw=0.5,
                       alpha=0.7, color=BAND_COLORS[BAND_LABELS[b]],
                       label=BAND_LABELS[b])
    ax_lc.invert_yaxis()  # magnitudes: brighter is up
    ax_lc.set_xlabel("MJD (days)")
    ax_lc.set_ylabel("magnitude")
    ax_lc.set_title(f"SDSS Stripe 82 quasar {obj_name}")
    ax_lc.legend(ncol=5, fontsize=8, loc="upper right")

    freqs = np.logspace(-3.5, -0.7, 400)  # cyc/day
    psd = mcarma_psd(best_res["F"], best_res["G"], best_res["H"],
                     best_res["Sigma"], freqs)  # (m, d, d)
    for b in range(d):
        ax_psd.loglog(freqs, np.abs(psd[:, b, b]), lw=1.8,
                      color=BAND_COLORS[BAND_LABELS[b]], label=BAND_LABELS[b])
    ax_psd.set_xlabel("frequency (cyc/day)")
    ax_psd.set_ylabel("power spectral density")
    ax_psd.set_title(f"Fitted mCARMA{best} auto-PSD by band")
    ax_psd.legend(ncol=5, fontsize=8)

    fig.suptitle(f"mcarma demo — {obj_name}: AICc selects mCARMA{best}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
