"""
mcarma/simulate.py
==================
Exact discrete-time simulation and power spectral density computation for MCARMA processes.

See docs/METHODS.Rmd sec 5 (shares unpack_params_jones / build_state_space / statespace
with the likelihood, so simulate and fit are convention-consistent by construction).

Returns
-------
simulate() now returns a SimResult dataclass that bundles the simulated
ObservationData together with the true parameter vector (theta_true), p, and q.
This makes true parameters available to any downstream script without
reconstruction.  The ObservationData is accessible via SimResult.data.

Usage
-----
    from mcarma.simulate import simulate, SimResult, mcarma_psd

    sim = simulate(theta, data, p, q)
    sim.data        # ObservationData with simulated y_obs
    sim.theta_true  # exact theta vector used to generate the data
    sim.p, sim.q    # model orders
"""

import numpy as np
from dataclasses import dataclass
from .optimizer_utils import unpack_params_jones, build_state_space
from .statespace import stationary_cov, transition_and_noise
from .observation import ObservationData


@dataclass
class SimResult:
    """
    Container returned by simulate().

    Attributes
    ----------
    data       : ObservationData
        Simulated observations (y_obs replaced by draws from the model).
        t_obs, band, and R are identical to the input template.
    theta_true : np.ndarray
        The exact flat parameter vector [AR as | MA bs | Chol | mu]
        used to generate data.  Storing this here avoids any reconstruction
        ambiguity (e.g. sign flips in Cholesky L).
    p : int   AR order used for generation.
    q : int   MA order used for generation.
    """
    data:        ObservationData
    theta_true:  np.ndarray
    p:           int
    q:           int


def simulate(theta, data, p, q, initial_state="stationary", seed=None):
    """
    Simulate an MCARMA(p, q) process at the observation times and band
    assignments given in data, using the parameters in theta.

    The simulation uses exact discrete-time transitions via the Van Loan
    matrix exponential, so it handles irregular observation times correctly.
    The initial state is drawn from the stationary distribution by default.

    Model
    -----
        dZ(t) = F Z(t) dt + G dB(t),   Cov(dB) = Sigma dt
        Y_k   = C_k (mu + H Z_k) + eps_k,   eps_k ~ N(0, R_k)

    Parameters
    ----------
    theta : np.ndarray
        Flat parameter vector [AR as | MA bs | Cholesky | mu],
        as produced by fit() or pack_params_jones() + mu concatenation.
    data  : ObservationData
        Provides t_obs (observation times), band (band indices), R_list
        (observation noise variances), d and n. y_obs is ignored — the
        simulated values replace it.
    p     : int, AR order
    q     : int, MA order

    Returns
    -------
    SimResult
        .data        : ObservationData with simulated y_obs
        .theta_true  : copy of theta (the exact generating parameters)
        .p, .q       : model orders

    Notes
    -----
    Every random draw goes through the local Generator built from `seed`: the
    stationary initial state, the per-epoch process-noise increment, and the
    measurement noise. Passing the same seed reproduces the light curve exactly.
    Passing seed=None draws a fresh trajectory.

    Before 2026-08-24 the process-noise increment came from the global legacy
    stream instead, so the seed governed only the endpoints and repeated calls
    returned different trajectories. Data generated before that date is not
    reproducible from its recorded seed.
    """
    d = data.d
    n = data.n

    # --- Unpack parameters ---
    mu_vec      = theta[-d:].reshape(d, 1)
    theta_carma = theta[:-d]
    ar_factors, ma_factors, Sigma = unpack_params_jones(theta_carma, d, p, q)
    F, G, H, Sigma = build_state_space(ar_factors, ma_factors, Sigma, d, p, q)

    state_dim = F.shape[0]

    rng = np.random.default_rng(seed)

    # --- Initialise state ---
    # initial_state: "stationary" (default) draws x ~ N(0, P0);
    #                "zero" sets x = 0 (so initial observation equals mu plus obs noise)
    P0 = stationary_cov(F, G, Sigma)
    if initial_state == "stationary":
        x = rng.multivariate_normal(np.zeros(state_dim), P0)
    elif initial_state == "zero":
        x = np.zeros(state_dim)
    else:
        # allow user to pass an explicit initial state vector
        try:
            x = np.asarray(initial_state, dtype=float).reshape(state_dim)
        except Exception:
            raise ValueError("initial_state must be 'stationary', 'zero', or a state vector of length state_dim")

    # --- Simulate state trajectory ---
    t_obs = data.t_obs
    y_sim = np.zeros(n)

    for k in range(n):
        if k > 0:
            dt       = t_obs[k] - t_obs[k - 1]
            Phi, Qd  = transition_and_noise(F, G, Sigma, dt)
            noise    = rng.multivariate_normal(np.zeros(state_dim), Qd)
            x        = Phi @ x + noise

        Ck     = data.C_list[k]
        Rk     = data.R_list[k][0, 0]
        mu_k   = (Ck @ mu_vec)[0, 0]
        y_hat  = mu_k + (Ck @ H @ x.reshape(-1, 1))[0, 0]
        y_sim[k] = y_hat + rng.normal(scale=np.sqrt(Rk))

    R_flat = np.array([data.R_list[k][0, 0] for k in range(n)])
    sim_data = ObservationData(
        t_obs = data.t_obs,
        y_obs = y_sim,
        band  = data.band,
        R     = R_flat,
        d     = d,
    )

    return SimResult(
        data       = sim_data,
        theta_true = theta.copy(),
        p          = p,
        q          = q,
    )


# ---------------------------------------------------------------------------
# Pole-zero distance helpers (Euclidean distance in complex plane)
# ---------------------------------------------------------------------------

def _factors_to_roots(band_factors):
    """
    Convert a list of Jones factor tuples for one band to complex roots.

    Jones factor conventions (matching optimizer_utils.unpack_params_jones):
      Linear  : (a,)       → root at -a  (real)
      Quadratic: (a1, a2) → roots at -a1/2 ± i*sqrt(a2 - (a1/2)^2)
    """
    roots = []
    for f in band_factors:
        if len(f) == 1:
            roots.append(complex(-f[0], 0.0))
        else:
            a1, a2 = f[0], f[1]
            re_root = -a1 / 2.0
            disc    = a2 - (a1 / 2.0) ** 2
            im_root = np.sqrt(max(disc, 0.0))
            roots.append(complex(re_root,  im_root))
            roots.append(complex(re_root, -im_root))
    return roots



# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

def mcarma_psd(F, G, H, Sigma, freqs):
    """
    Matrix-valued power spectral density of an MCARMA(p, q) process.

    Implements the state-space form of the paper's Eq. (PSD):

        P(ω) = H (iωI − F)⁻¹ G Σ Gᵀ [(iωI − F)⁻¹]ᴴ Hᵀ / (2π)

    Parameters
    ----------
    F      : (kp, kp)   state transition matrix
    G      : (kp, k)    noise input matrix
    H      : (k,  kp)   observation matrix
    Sigma  : (k,  k)    innovation covariance  (= V in the paper)
    freqs  : (m,)       frequencies in cycles/unit-time (NOT angular)

    Returns
    -------
    psd : ndarray, shape (m, k, k)
        Real part of the PSD matrix at each frequency.
    """
    state_dim = F.shape[0]
    k         = H.shape[0]
    I         = np.eye(state_dim)
    GSGt      = G @ Sigma @ G.T

    psd = np.zeros((len(freqs), k, k))
    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        M     = np.linalg.solve((1j * omega) * I - F, np.eye(state_dim))
        S     = H @ M @ GSGt @ M.conj().T @ H.T
        psd[i] = np.real(S) / (2.0 * np.pi)

    return psd