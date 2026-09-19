"""Lightweight preprocessing shared by the fit pipeline and the bootstrap.

Kept in the mcarma package (rather than the analysis layer) so that core
routines -- e.g. the parametric bootstrap -- can center simulated data the same
way the fit does, without importing upward from fits/.
"""
import numpy as np


def center_bands(t: np.ndarray, y: np.ndarray, band: np.ndarray) -> np.ndarray:
    """
    Subtract per-band empirical mean (no polynomial trend removal).

    Iterates over the bands ACTUALLY PRESENT in ``band`` (np.unique) rather than a
    hardcoded ``range(N_BANDS)``: N_BANDS=5 (real-data ugriz) silently skipped band
    5 of the 6-band (ugrizy, D=6) simulation study, leaving the 'y' band at its raw
    mean (~19) while the others were centred to ~0. That broke the cross-band
    coherence the (2,1) MA detectability relies on (1/6 bands carrying a ~19 offset
    into build_perband_warm_theta) and tanked any mu=0 truth evaluation. np.unique
    is correct for both 5- and 6-band data.
    """
    y_out = y.copy()
    for b in np.unique(band):
        mask = band == b
        if np.any(mask):
            y_out[mask] -= np.mean(y[mask])
    return y_out
