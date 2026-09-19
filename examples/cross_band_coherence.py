"""
examples/cross_band_coherence.py
================================
The thing this package is for: fit all five ugriz bands of one quasar as one
process, then read off how strongly the bands vary together and what each
band's spectrum looks like.

Fitting the bands jointly rather than one at a time buys two things. The
cross-band correlation of the driving noise is estimated rather than assumed,
and every band's spectrum is constrained by all the epochs instead of only its
own, which matters here because SDSS puts about a fifth of the measurements in
each filter.

Run:
    python examples/cross_band_coherence.py
    python examples/cross_band_coherence.py --obj PATH --order 2,0

Prints the driving correlation matrix and the squared coherence, and writes
examples/cross_band_coherence.png with the five marginal spectra.

For a run that also compares orders by AICc, see sdss_stripe82_demo.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcarma import (ObservationData, fit, compute_aicc, mcarma_psd,
                    sigma_to_var_corr)

from mcarma.model_utils import build_perband_warm_theta

from stripe82 import (N_BANDS, BAND_LABELS, BAND_COLORS, HERE, DEFAULT_OBJ,
                      load_stripe82, production_prior_kwargs, n_params)

OUT_PNG = os.path.join(HERE, "cross_band_coherence.png")


def print_matrix(M, title, note=""):
    print("\n" + title)
    if note:
        print(note)
    print("      " + "".join("%8s" % b for b in BAND_LABELS))
    for i, b in enumerate(BAND_LABELS):
        print("  %s   " % b + "".join("%8.3f" % M[i, j] for j in range(N_BANDS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="path to a Stripe 82 file")
    ap.add_argument("--order", default="1,0",
                    help="p,q to fit (default 1,0; the ladder is 1,0 / 2,0 / 2,1)")
    ap.add_argument("--n-restarts-perband", type=int, default=2,
                    help="restarts inside each single-band warm-start fit (default 2)")
    ap.add_argument("--out", default=OUT_PNG, help="output figure path")
    args = ap.parse_args()

    p, q = (int(v) for v in args.order.split(","))
    t, y, band, R = load_stripe82(args.obj)
    d = N_BANDS
    data = ObservationData(t, y, band, R, d=d)
    obj_name = os.path.basename(args.obj)
    print(f"{obj_name}: {data.n} epochs over {t.max() - t.min():.0f} days, "
          f"per-band counts = {[int((band == b).sum()) for b in range(d)]}")

    # Warm start the joint fit from five independent single-band fits, which is
    # what the production pipeline does. A 23-parameter search started at random
    # takes minutes on finite-difference gradients; started here it takes
    # seconds, and it is the same optimum.
    priors = production_prior_kwargs(y, band, d)
    warm = build_perband_warm_theta(t, y, band, R, d, p, q,
                                    n_restarts=args.n_restarts_perband, seed=0)
    res = fit(data, p, q, n_restarts=1, warm_theta=warm, seed=0,
              use_jax_grad=True, **priors)
    k = n_params(d, p, q)
    print(f"mCARMA({p},{q}) fit: loglik={res['loglik_pure']:.2f}  k={k}  "
          f"AICc={compute_aicc(res['loglik_pure'], k, data.n):.2f}")

    # Sigma is the covariance of the driving noise. Its correlation matrix says
    # how much of each band's variability is shared, and squaring entry (i, j)
    # gives the fraction of band i's variance that band j accounts for.
    var, corr = sigma_to_var_corr(np.asarray(res["Sigma"], float))
    print_matrix(corr, "driving correlation",
                 "  off-diagonal near 1 means the bands move together")
    print_matrix(corr ** 2, "squared coherence",
                 "  the shared fraction of variance, band pair by band pair")

    # The diagonal of the spectral matrix is each band's own spectrum.
    freqs = np.logspace(-3.5, -0.7, 400)
    psd = mcarma_psd(res["F"], res["G"], res["H"], res["Sigma"], freqs)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for b in range(d):
        ax.loglog(freqs, np.abs(psd[:, b, b]), lw=1.8,
                  color=BAND_COLORS[BAND_LABELS[b]], label=BAND_LABELS[b])
    ax.set_xlabel("frequency (cycles/day)")
    ax.set_ylabel(r"marginal PSD (mag$^2$ day)")
    ax.set_title(f"{obj_name}: joint mCARMA({p},{q}) marginal spectra")
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
