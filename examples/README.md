# mcarma examples

Three runnable examples on a real SDSS Stripe 82 quasar light curve that ships
with the package. Nothing here needs the research tree, a cluster, or any data
you have to download.

| script | what it does | roughly how long |
| --- | --- | --- |
| `one_band_psd.py` | fits one filter, prints the variability timescale and the spectrum around its break | 10 seconds |
| `cross_band_coherence.py` | fits all five filters as one process, prints the cross-band coherence, plots the five spectra | 1 minute |
| `sdss_stripe82_demo.py` | the full walkthrough: the (1,0)/(2,0)/(2,1) order ladder, AICc selection, light curve and PSD figure | several minutes |

```bash
pip install mcarma[plots]

python examples/one_band_psd.py                     # r band, prints only
python examples/one_band_psd.py --band g
python examples/cross_band_coherence.py             # writes cross_band_coherence.png
python examples/sdss_stripe82_demo.py --quick       # (1,0) only
python examples/sdss_stripe82_demo.py               # the whole ladder
```

`one_band_psd.py` prints its results and needs no plotting library. The other
two write a PNG, so they need matplotlib, which is the `plots` extra above.
Every script takes `--obj PATH` to point at a different Stripe 82 file.

## Start here: one band

`one_band_psd.py` is the shortest path through the API. It loads one filter,
fits a CARMA(1,0) (the continuous-time damped random walk that is the standard
description of quasar optical variability), and then reads two things off the
fitted state-space matrices: the decay timescale, and the frequency where the
spectrum turns over. It also prints the local slope of the spectrum on either
side of that break, which should come out near 0 below it and near -2 above it.

## The one that matters: five bands at once

`cross_band_coherence.py` is what this package does that a single-band CARMA
code cannot. It fits all five ugriz bands as one process, so the correlation
between the bands' driving noise is estimated rather than assumed, and each
band's spectrum is constrained by every epoch instead of only its own. SDSS
puts about a fifth of the measurements in each filter, so that matters.

It prints the driving correlation matrix and its square, the shared fraction of
variance for each band pair, and plots the five marginal spectra together.

The joint fit is warm started from five independent single-band fits, which is
what the production pipeline does: a 25-parameter search started at random takes
minutes, started from the per-band fits it takes seconds.

## The full walkthrough

`sdss_stripe82_demo.py` runs the whole ladder and is the closest thing here to
the pipeline in the papers. It shows:

- **Loading** a real Stripe 82 file (the 17-column ugriz format) into the
  package's `ObservationData` container.
- **Fitting** `mcarma.fit.fit` for each order with the same penalty stack the
  production pipeline uses: the one-sided Sigma soft box (floor and ceiling on
  the latent driving variance) plus the AR/MA band and AR-damping guardrails.
  `examples/stripe82.py::production_prior_kwargs` mirrors
  `fits/process_quasars/process_quasars.py`.
- **Selecting** the order with `mcarma.model_utils.compute_aicc`, scored on the
  *pure* (unpenalized) maximized log-likelihood.
- **Interpreting** the fit through `mcarma.simulate.mcarma_psd`, the closed-form
  multivariate PSD `P(w) = H(iwI-F)^-1 G Sigma G^T (iwI-F)^-H H^T / 2pi`.

It runs on finite-difference gradients, which is why it is the slow one.
`--quick` fits (1,0) alone; lowering `--n-restarts` gives a rougher, faster
look.

## Bundled data

`examples/data/1000679` is one SDSS Stripe 82 quasar, ugriz, 307 measurements
across five filters over 3337 days (64 of them in r). It was copied from
`fits/QSO_S82/`, where the other 9257 objects of the sample live.

`examples/stripe82.py` holds the pieces the three scripts share: the file
format, the production penalty stack, and the AICc parameter count. It is
example scaffolding, not part of the public API.

## Demo versus production

These examples fit one object. The real pipeline (`fits/process_quasars/`) fits
hundreds as a SLURM array, always with the analytic JAX gradient, and adds a
penalized-Hessian and parametric-bootstrap standard-error pass. The modeling is
the same; the scale is not.

## Cadence note (SDSS versus LSST)

SDSS Stripe 82 is **co-observed**: all five ugriz are taken within about five
minutes each night, so every epoch carries all bands. The Rubin/LSST regime is
**staggered**, one band per visit. The package handles both, since each
`(t, band)` observation is a scalar measurement in `ObservationData`, but the
information per epoch differs, which changes order selection at low
signal-to-noise.
