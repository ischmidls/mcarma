"""The observation container every other module takes as its input.

`ObservationData` holds one multiband light curve in the scalar-observation
form the Kalman filter wants: each epoch carries a single measurement of a
single band, not a d-vector. Bands are therefore observed one at a time and may
be sampled on completely different grids, which is what real surveys give you.

Building it once precomputes the per-epoch band selector and noise matrices, so
the filter does not rebuild them on every likelihood evaluation.

    data = ObservationData(t_obs, y_obs, band, R, d=5)

The class docstring below documents the state layout and the attributes.
"""
import numpy as np


class ObservationData:
    """
    Precomputed per-observation matrices for Kalman filtering.

    Stores the raw time series alongside the band selector matrices C_list
    and observation noise matrices R_list, so that t_obs, y_obs, and d do
    not need to be passed separately throughout the fitting pipeline.

    State layout is lag-major: the state vector is
        Z = [Z_0(d), Z_1(d), ..., Z_{p-1}(d)]
    where d is the number of bands. Band b at lag 0 occupies state index b.

    The band selector C_k is a (1, d) matrix with C_k[0, b_k] = 1, picking
    band b_k from the lag-0 block. The full effective observation matrix
        C_eff = C_k @ H   shape (1, d*p)
    is formed in kalman.py by multiplying with the MA matrix H.

    Attributes
    ----------
    t_obs  : (n,) float array, observation times
    y_obs  : (n,) float array, scalar observations
    d      : int, number of bands
    n      : int, number of observations
    band   : (n,) int array, band index at each observation
    C_list : list of n (1, d) band selector matrices
    R_list : list of n (1, 1) observation noise variance matrices
    """

    def __init__(self, t_obs, y_obs, band, R, d):
        """
        Parameters
        ----------
        t_obs : (n,) float array
            Sorted observation times.
        y_obs : (n,) float array
            Scalar observations (raw, not centred).
        band  : (n,) int array
            Band index at each observation, values in {0, ..., d-1}.
        R     : (n,) float array
            Observation noise variance at each observation.
        d     : int
            Number of bands.
        """
        self.t_obs = np.asarray(t_obs, dtype=float)
        self.y_obs = np.asarray(y_obs, dtype=float)
        self.band  = np.asarray(band,  dtype=int)
        self.d     = d
        self.n     = len(t_obs)

        self.R_list = [np.array([[r]]) for r in R]

        self.C_list = []
        for b_obs in band:
            C = np.zeros((1, d))
            C[0, b_obs] = 1.0
            self.C_list.append(C)