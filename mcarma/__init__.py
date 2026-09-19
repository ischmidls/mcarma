"""mcarma — multivariate CARMA modeling of irregularly sampled light curves.

Penalized maximum-likelihood fitting of multiband continuous-time ARMA models
to astronomical light curves, plus the closed-form multivariate PSD
and a data simulator. Builds on single-band CARMA (Kelly et al. 2014) and the
multiband damped random walk (Hu et al. 2020).

Quick start
-----------
    from mcarma import ObservationData, fit, compute_aicc, mcarma_psd

    data = ObservationData(t, y, band, R, d=5)     # scalar (t, band) observations
    res  = fit(data, p=2, q=1)                      # MLE + optional penalties
    aicc = compute_aicc(res["loglik_pure"], k, data.n)
    psd  = mcarma_psd(res["F"], res["G"], res["H"], res["Sigma"], freqs)

See `examples/sdss_stripe82_demo.py` for a full worked example on real SDSS
Stripe 82 data.

Note on JAX: the analytic-gradient objective (`mcarma.jax_loglik`), the JAX
transition helpers (`mcarma.jax_transitions`), the RTS smoother
(`mcarma.smoother`, which uses them), and the smoother-based reconstruction
helpers (`mcarma.rts`) import JAX at module load. They are **not** imported here,
so `import mcarma` stays JAX-free and safe on login nodes; import those
submodules explicitly on a compute node when you need them.
"""
import logging as _logging

# The library logs progress through the standard logging module and attaches no
# handler of its own, so an application that has not configured logging sees
# nothing. The research drivers in fits/ call logging.basicConfig, which is what
# puts these lines back in the SLURM logs.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .observation import ObservationData
from .fit import fit
from .model_utils import compute_aicc
from .simulate import simulate, mcarma_psd
from .reporting import sigma_to_var_corr

__version__ = "0.1"

__all__ = [
    "ObservationData",
    "fit",
    "compute_aicc",
    "simulate",
    "mcarma_psd",
    "sigma_to_var_corr",
    "__version__",
]
