# mcarma

Fit continuous-time ARMA models to several bands of an astronomical light curve
at once, by penalized maximum likelihood, on data sampled as irregularly as real
observations are.

Single-band CARMA fits are standard for quasar variability (Kelly et al. 2014).
The multiband case is what this package is for: the bands are modeled as one
process driven by correlated noise, so the cross-band spectrum is estimated
rather than assumed, and each band's spectrum is constrained by every epoch
instead of only its own. Builds on the multiband damped random walk of Hu et al.
(2020).

```bash
pip install mcarma
```

Not on PyPI until the papers are submitted; until then install from a
checkout (see Development at the bottom).

## Fit one object

```python
import numpy as np
from mcarma import ObservationData, fit, compute_aicc, mcarma_psd

# One scalar measurement per row: time, magnitude, which band, and the
# measurement variance. Bands need not be observed together.
data = ObservationData(t_obs=t, y_obs=y, band=band, R=err**2, d=5)

res = fit(data, p=2, q=1, n_restarts=4, use_jax_grad=True)

k    = 5 * 2 + 5 * 1 + 5 * 6 // 2 + 5          # AR + MA + chol(Sigma) + mu
aicc = compute_aicc(res["loglik_pure"], k, data.n)

freqs = np.logspace(-3.5, -0.7, 400)           # cycles per day
psd   = mcarma_psd(res["F"], res["G"], res["H"], res["Sigma"], freqs)
psd[:, 2, 2]                                   # the r band's own spectrum
```

Each band is centered internally, so pass raw magnitudes. `res["theta"]` is the
fitted parameter vector, `res["loglik"]` the penalized value and
`res["loglik_pure"]` the plain one, which is what AICc must be scored on.
`res["hess_inv"]` carries the inverse Hessian when the optimizer produced one.

## Examples

Three runnable examples ship with the package, on a real SDSS Stripe 82 quasar
light curve that ships with them. Nothing here needs a cluster or a download.

| script | what it does | roughly how long |
| --- | --- | --- |
| `one_band_psd.py` | fits one filter, prints the variability timescale and the spectrum around its break | 10 seconds |
| `cross_band_coherence.py` | fits all five filters as one process, prints the cross-band coherence, plots the five spectra | 1 minute |
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

$$dZ(t)=F Z(t)dt + G\,dB(t),\ \mathrm{Cov}(dB)=\Sigma dt;\quad
Y_k = C_k(\mu + H Z_k) + \varepsilon_k,\ \varepsilon_k\sim N(0,R_k).$$

$F$ is built from AR Jones factors, $H$ from MA Jones factors; cross-band
dependence enters only through $\Sigma$. Parameter vector:

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
selects. Production fits add penalties (observable band, AR damping, one-sided
Sigma box) and take standard errors from the penalized Hessian.

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
