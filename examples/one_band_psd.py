"""
examples/one_band_psd.py
========================
First contact with the API: fit one band of one quasar and read its power
spectrum off the fit.

The model is a single-band CARMA(1,0), the continuous-time analogue of a damped
random walk and the standard description of quasar optical variability. It has
one dynamical parameter, so the fit takes a few seconds and the answer is a
single number you can check against the light curve by eye: a variability
timescale in days.

Run:
    python examples/one_band_psd.py                # r band of the bundled object
    python examples/one_band_psd.py --band g
    python examples/one_band_psd.py --obj PATH --band i

Everything printed comes out of the fitted state-space matrices, so the same
three lines work for any (p, q) and any number of bands.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mcarma import ObservationData, fit, compute_aicc, mcarma_psd

from stripe82 import (BAND_LABELS, DEFAULT_OBJ, load_stripe82,
                      production_prior_kwargs, n_params)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", default=DEFAULT_OBJ, help="path to a Stripe 82 file")
    ap.add_argument("--band", default="r", choices=BAND_LABELS,
                    help="which filter to fit (default r)")
    ap.add_argument("--n-restarts", type=int, default=4,
                    help="random restarts (default 4)")
    args = ap.parse_args()

    t_all, y_all, band_all, R_all = load_stripe82(args.obj)
    b = BAND_LABELS.index(args.band)
    keep = band_all == b
    if keep.sum() < 10:
        raise SystemExit(f"only {int(keep.sum())} epochs in {args.band}; pick another band")

    # One band is d=1: every observation belongs to band 0 of a one-band model.
    t, y, R = t_all[keep], y_all[keep], R_all[keep]
    band = np.zeros(t.size, dtype=int)
    data = ObservationData(t, y, band, R, d=1)

    print(f"{os.path.basename(args.obj)}, {args.band} band: {data.n} epochs over "
          f"{t.max() - t.min():.0f} days, median gap {np.median(np.diff(t)):.1f} days")

    p, q = 1, 0
    res = fit(data, p, q, n_restarts=args.n_restarts, seed=0,
              use_jax_grad=True, **production_prior_kwargs(y, band, 1))
    k = n_params(1, p, q)
    print(f"CARMA({p},{q}) fit: loglik={res['loglik_pure']:.2f}  k={k}  "
          f"AICc={compute_aicc(res['loglik_pure'], k, data.n):.2f}")

    # The AR root is the one dynamical parameter of a (1,0) model. F is 1x1 here
    # and holds minus that root, so its reciprocal is the decay timescale and
    # the spectrum turns over at the matching frequency.
    a = -float(np.asarray(res["F"], float)[0, 0])
    tau = 1.0 / a
    f_break = a / (2.0 * np.pi)
    print(f"\ndecay timescale  tau = 1/a = {tau:.1f} days")
    print(f"break frequency  f = a/2pi = {f_break:.2e} cycles/day "
          f"(period {1.0 / f_break:.1f} days)")

    # mcarma_psd takes the fitted state-space matrices and returns the (d, d)
    # spectral matrix at each frequency. At d=1 that is one number per
    # frequency: the band's own power spectrum, in mag^2 per cycle/day.
    freqs = np.array([f_break / 10.0, f_break / 3.0, f_break,
                      f_break * 3.0, f_break * 10.0])
    psd = mcarma_psd(res["F"], res["G"], res["H"], res["Sigma"], freqs)[:, 0, 0]
    print("\n  f (cyc/day)   period (d)      PSD      slope from previous row")
    for i, (f, s) in enumerate(zip(freqs, psd.real)):
        if i == 0:
            slope = ""
        else:
            slope = "%+.2f" % (np.log(psd.real[i] / psd.real[i - 1])
                               / np.log(freqs[i] / freqs[i - 1]))
        print("  %10.3e   %8.1f   %.4e   %s" % (f, 1.0 / f, s, slope))
    print("\nA (1,0) spectrum is flat well below the break and falls as f^-2 well\n"
          "above it, so the last slope should be near -2 and the first near 0.")


if __name__ == "__main__":
    main()
