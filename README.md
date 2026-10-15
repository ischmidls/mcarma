# mcarma

Fit multivariate continuous-time autoregressive moving average (MCARMA) models
jointly to irregularly sampled multi-band astronomical time series.

The package extends multivariate damped random walk modeling to higher-order
stochastic dynamics. Each band can have its own temporal dynamics, while
dependence across bands is modeled through correlated stochastic drivers. The
resulting matrix-valued power spectral density characterizes both marginal
variability within individual bands and dependence across bands.

mcarma supports irregular sampling, heteroscedastic measurement errors, and
partially observed bands, and fits the joint model by maximum likelihood with
state-space computation.

```bash
pip install mcarma
```

## Fit one multi-band object

```python
import numpy as np
from mcarma import ObservationData, fit, mcarma_psd

# One scalar measurement per row: time, magnitude, which band, and the
# measurement variance. Bands need not be observed together.
data = ObservationData(t_obs=t, y_obs=y, band=band, R=err**2, d=5)

res = fit(data, p=2, q=1)                     # the two-stage estimator
res["aicc"]                                    # model selection score

freqs = np.logspace(-3.5, -0.7, 400)           # cycles per day
psd   = mcarma_psd(res["F"], res["G"], res["H"], res["Sigma"], freqs)
psd[:, 2, 2]                                   # the r band's own spectrum
```

`fit` is the estimator used in the papers, and it runs in two stages:

1. **Stage 1** maximizes a penalized likelihood from a warm start built out of
   independent single-band fits. The penalties are a soft box on each band's
   driving variance, priors that keep the AR and MA timescales inside the
   observed range, a floor on the AR decay rate, and diagonal loading of the
   driving covariance. They exist only to find a good starting point.
2. **Stage 2** starts from the stage-1 point and maximizes the ordinary
   likelihood, with no penalty.

Every reported quantity comes from stage 2. Each band is centered before
fitting and the means are shifted back, so pass raw magnitudes. `res["theta"]`
is the fitted parameter vector, `res["loglik"]` the ordinary log-likelihood,
`res["aicc"]` the AICc built from it, and `res["converged"]` says whether stage
2 reached a stationary point above its start. `res["hess_inv"]` carries the
inverse Hessian from stage 2 when the optimizer produced one.

The single-stage optimizer is still available as `fit_penalized`
(`mcarma.fit.fit`) for anyone who wants to set the penalties by hand. Its
defaults are **not** the papers' estimator: with `ar_band_lambda` and
`ma_band_lambda` left as `None` it applies a legacy hinge on the AR/MA
timescales, kept so older runs still reproduce. Pass
`ar_band_lambda=0.0, ma_band_lambda=0.0` for the ordinary likelihood, or call
`mcarma.stage2.finish_mle` on its result.

## Examples

Three runnable examples are included using an SDSS Stripe 82 quasar light curve
distributed with the package. The examples require no external data downloads
and can be run on a standard computer.

| script | what it does | roughly how long |
| --- | --- | --- |
| `one_band_psd.py` | fits one filter, prints the variability timescale and the spectrum around its break | 10 seconds |
| `cross_band_coherence.py` | jointly fits all five filters and estimates cross-band coherence and marginal PSDs | 1 minute |
| `sdss_stripe82_demo.py` | the full walkthrough: the (1,0)/(2,0)/(2,1) order ladder, AICc selection, light curve and PSD figure | several minutes |

```bash
pip install mcarma[plots]

python examples/one_band_psd.py                  # r band, prints only
python examples/one_band_psd.py --band g
python examples/cross_band_coherence.py          # writes cross_band_coherence.png
python examples/sdss_stripe82_demo.py --quick    # (1,0) only
python examples/sdss_stripe82_demo.py            # the whole ladder
```

`one_band_psd.py` prints its results and needs no plotting library. The other
two write a PNG, so they need matplotlib, which is the `plots` extra above.
Every script takes `--obj PATH` to point at a different Stripe 82 file, and the
examples directory carries a longer walkthrough of what each one shows.

## Model

mcarma represents a $d$-band astronomical time series as a joint
continuous-time stochastic process. Band-specific ARMA dynamics determine the
marginal temporal behavior, while correlated Wiener drivers introduce
dependence across bands.

$$dZ(t)=F Z(t)dt + G\,dB(t),\ \mathrm{Cov}(dB)=\Sigma dt;\quad
Y_k = C_k(\mu + H Z_k) + \varepsilon_k,\ \varepsilon_k\sim N(0,R_k).$$

$F$ is built from AR Jones factors, $H$ from MA Jones factors. Under the
structured MCARMA formulation implemented here, cross-band dependence is
introduced through the covariance matrix $\Sigma$ of the Wiener drivers.

Parameter vector:

```
theta = [ AR | MA | chol(Sigma) | mu ],   dim = d*p + d*q + d(d+1)/2 + d
```

AR/MA factor coefficients are stored in log space (positive roots). An MA linear
factor `b` places a zero at frequency `1/b`, so to put a zero in the observable
band you set `b = 1/omega_z`. That convention matters.

The likelihood is the Kalman prediction-error decomposition
(`mcarma/kalman.py`) with exact irregular-gap transitions
(`mcarma/statespace.py`, Lyapunov `Qd`). The analytic
gradient and Hessian-vector products are in
`mcarma/jax_loglik.py` and are what `use_jax_grad=True`
selects. The penalties in stage 1 improve numerical stability and steer the
search away from poorly identified configurations. The estimate, the
log-likelihood, model comparison and standard errors all use the ordinary,
unpenalized likelihood at the stage-2 point.

## Dependencies

numpy, scipy and jax. matplotlib, numdifftools, and pandas with statsmodels are
extras (`plots`, `numdiff`, `analysis`) used by the figures, the fallback
Hessian, and the analysis scripts.

Importing `mcarma` does not import jax. The analytic-gradient objective
(`mcarma.jax_loglik`), the transition helpers (`mcarma.jax_transitions`), the
RTS smoother (`mcarma.smoother`) and the reconstruction helpers (`mcarma.rts`)
import it at module load, so import those explicitly when you need them. On a
shared cluster that means importing them inside a job rather than on a login
node, where jax cannot start its thread pool.

The library logs its fit progress through the standard `logging` module and
attaches no handler, so it is silent until an application configures one:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

## References

- Kelly, B. C. et al. (2014), CARMA for stochastic variability.
- Hu, Z. et al. (2020), multiband damped random walk.
- Jones & Ackerson (1990); Cox & Snell (1968); Cordeiro & Klein (1994);
  Politis, Romano & Wolf (1999).

---

## Development

```bash
git clone https://github.com/ischmidls/mcarma
cd mcarma
pip install -e ".[plots,numdiff,analysis,dev]"
pytest tests/
```

The research behind the package, the simulation study and the SDSS Stripe 82
pipeline the papers report, lives in a separate repository.
