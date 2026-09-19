"""
model_utils.py
==============
Model fitting helpers: warm‑start construction, AICc, Hessian analysis,
innovation computation, diagnostic plots.

See docs/METHODS.Rmd sec 10-11 (AICc, per-band warm start).
"""

import logging
import warnings
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# mcarma imports
from mcarma.observation import ObservationData
from mcarma.kalman import loglik

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AICc and parameter count
# ---------------------------------------------------------------------------
def count_params(d: int, p: int, q: int) -> int:
    """Number of free parameters in a d‑dimensional MCARMA(p,q) model."""
    return d * p + d * q + d * (d + 1) // 2 + d


def compute_aicc(log_lik: float, k: int, n_obs: int) -> float:
    """
    Akaike Information Criterion with correction for small samples.
    Returns inf if n_obs - k - 1 <= 0.
    """
    denom = n_obs - k - 1
    if denom <= 0:
        return np.inf
    return -2.0 * log_lik + 2.0 * k + 2.0 * k * (k + 1) / denom


# ---------------------------------------------------------------------------
# Warm‑start from per‑band CARMA fits
# ---------------------------------------------------------------------------
def build_perband_warm_theta(
    t: np.ndarray, y: np.ndarray, band: np.ndarray, R: np.ndarray,
    d: int, p: int, q: int,
    n_restarts: int = 4,
    seed: Optional[int] = None,
    slopes: Optional[np.ndarray] = None,
    ma_above_ar: bool = False,
) -> Optional[np.ndarray]:
    """
    Fit CARMA(p,q) independently per band (d=1) and assemble a joint warm‑start theta.

    Returns None if more than d//2 bands fail.

    slopes : ndarray (d,) or None, default None
        Per-band slopes of a known/fixed deterministic linear trend in
        observation time. When given, each per-band fit models its own slope
        ``slopes[b]`` inside the likelihood (forwarded to `fit`), keeping the
        warm start consistent with a joint fit that also models the trend.
        When None, no trend is modelled (default).
    """
    from mcarma.fit import fit

    rng = np.random.default_rng(seed)
    n_chol = d * (d + 1) // 2
    rows, cols = np.tril_indices(d)
    diag_pos = [k for k, (i, j) in enumerate(zip(rows, cols)) if i == j]

    ar_parts, ma_parts, mu_parts = [], [], []
    chol_warm = np.zeros(n_chol)
    n_failed = 0

    for b in range(d):
        mask = band == b
        if not np.any(mask):
            ar_parts.append(np.zeros(p))
            if q > 0:
                ma_parts.append(np.zeros(q))
            chol_warm[diag_pos[b]] = np.log(1e-6)
            mu_parts.append(0.0)
            n_failed += 1
            continue

        t_b = t[mask]
        y_b = y[mask]
        R_b = R[mask]
        band_b = np.zeros(len(t_b), dtype=int)
        data_b = ObservationData(t_b, y_b, band_b, R_b, d=1)

        slopes_b = (np.array([slopes[b]], dtype=float)
                    if slopes is not None else None)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit(data_b, p, q,
                          n_restarts=n_restarts,
                          seed=int(rng.integers(1 << 31)),
                          slopes=slopes_b,
                          ma_above_ar=ma_above_ar)
            theta_b = res["theta"]
            ar_parts.append(theta_b[:p].copy())
            if q > 0:
                ma_parts.append(theta_b[p:p + q].copy())
            chol_warm[diag_pos[b]] = float(theta_b[p + q])
            mu_parts.append(float(theta_b[p + q + 1]))
        except Exception:
            ar_parts.append(np.zeros(p))
            if q > 0:
                ma_parts.append(np.zeros(q))
            chol_warm[diag_pos[b]] = np.log(1e-6)
            mu_parts.append(0.0)
            n_failed += 1

    if n_failed > d // 2:
        return None

    parts = [np.concatenate(ar_parts)]
    if q > 0:
        parts.append(np.concatenate(ma_parts))
    parts.append(chol_warm)
    parts.append(np.array(mu_parts))
    return np.concatenate(parts)


def build_higher_order_warm_theta(
    theta_lower: np.ndarray,
    data: ObservationData,
    p: int,
    q_lower: int,
    q_higher: int,
    rng: Optional[np.random.Generator] = None,
    ma_above_ar: bool = False,
) -> np.ndarray:
    """
    Construct a warm‑start theta for MCARMA(p, q_higher) from a fitted
    MCARMA(p, q_lower) parameter vector (q_higher > q_lower, same p).

    The AR block (d*p), Cholesky block (d(d+1)/2) and mu block (d) are copied
    verbatim from the lower‑order fit. For the MA block, each band keeps its
    ``q_lower`` fitted coefficients and gains ``q_higher - q_lower`` fresh
    coefficients drawn from a random initialisation (whose MA roots are placed
    near the AR roots via the eta parameterisation in ``prepare_initial_params``).

    This is what enables, e.g., warm‑starting the (2,1) fit from a converged
    (2,0) fit: the AR/covariance structure is inherited and only the new MA
    factor is initialised afresh.
    """
    from mcarma.fit import prepare_initial_params_safe

    if q_higher <= q_lower:
        raise ValueError(
            f"q_higher ({q_higher}) must exceed q_lower ({q_lower})."
        )

    d = data.d
    n_ar = d * p
    n_chol = d * (d + 1) // 2
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    ar_lower = theta_lower[:n_ar]
    ma_lower = theta_lower[n_ar:n_ar + d * q_lower]
    chol_lower = theta_lower[n_ar + d * q_lower:n_ar + d * q_lower + n_chol]
    mu_lower = theta_lower[-d:]

    # Fresh higher‑order draw supplies the new MA coefficients (band‑major).
    theta_seed = prepare_initial_params_safe(data, p, q_higher, rng=rng,
                                             ma_above_ar=ma_above_ar)
    ma_seed = theta_seed[n_ar:n_ar + d * q_higher]

    # Merge per band: keep the q_lower fitted MA coeffs, append fresh ones.
    ma_high = np.empty(d * q_higher)
    for b in range(d):
        merged = ma_seed[b * q_higher:(b + 1) * q_higher].copy()
        if q_lower > 0:
            merged[:q_lower] = ma_lower[b * q_lower:(b + 1) * q_lower]
        ma_high[b * q_higher:(b + 1) * q_higher] = merged

    return np.concatenate([ar_lower, ma_high, chol_lower, mu_lower])


# ---------------------------------------------------------------------------
# Hessian analysis: positive definiteness & eigenvalues
# ---------------------------------------------------------------------------
def _neg_loglik_wrapper(data: ObservationData):
    """Return a callable that computes the negative log‑likelihood for a theta."""
    def neg_ll(theta: np.ndarray) -> float:
        try:
            return -loglik(data, theta)
        except Exception:
            return np.inf
    return neg_ll


def _hessian_from_bfgs_inv(result: Dict) -> Optional[np.ndarray]:
    """Tier 1: Invert the stored BFGS inverse Hessian."""
    hess_inv = result.get("hess_inv")
    if hess_inv is not None and isinstance(hess_inv, np.ndarray):
        try:
            return np.linalg.inv(hess_inv)
        except Exception:
            pass
    return None


def _hessian_from_numdifftools(neg_ll, theta: np.ndarray) -> Optional[np.ndarray]:
    """Tier 2: Richardson‑extrapolated Hessian via numdifftools."""
    try:
        import numdifftools as nd
        hess_fn = nd.Hessian(neg_ll, method="central", step_ratio=2.0)
        H = hess_fn(theta)
        if np.all(np.isfinite(H)):
            return H
    except Exception:
        pass
    return None


def _hessian_from_manual_fd(neg_ll, theta: np.ndarray, eps: float = 1e-4) -> Optional[np.ndarray]:
    """Tier 3: Central finite differences with adaptive step."""
    n = len(theta)
    h = np.where(np.abs(theta) > 1.0, eps * np.abs(theta), eps)
    try:
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n); ei[i] = h[i]
                ej = np.zeros(n); ej[j] = h[j]
                fpp = neg_ll(theta + ei + ej)
                fpm = neg_ll(theta + ei - ej)
                fmp = neg_ll(theta - ei + ej)
                fmm = neg_ll(theta - ei - ej)
                if not np.all(np.isfinite([fpp, fpm, fmp, fmm])):
                    return None
                H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * h[i] * h[j])
                H[j, i] = H[i, j]
        return H
    except Exception:
        return None


def compute_hessian_eigenvalues(
    data: ObservationData,
    result: Dict[str, Any],
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Compute eigenvalues of the Hessian of the negative log‑likelihood at MLE.

    Uses the same three‑tier strategy as check_hessian_pd.
    Returns sorted eigenvalues (ascending) or None if all tiers fail.
    """
    theta = result.get("theta")
    if theta is None:
        return None

    neg_ll = _neg_loglik_wrapper(data)

    # Tier 1
    H = _hessian_from_bfgs_inv(result)
    if H is not None:
        if verbose:
            _log.info("    [eigenvals] Tier 1 (BFGS inverse) succeeded.")
        return np.linalg.eigvalsh(0.5 * (H + H.T))

    # Tier 2
    H = _hessian_from_numdifftools(neg_ll, theta)
    if H is not None:
        if verbose:
            _log.info("    [eigenvals] Tier 2 (numdifftools) succeeded.")
        return np.linalg.eigvalsh(0.5 * (H + H.T))

    # Tier 3
    H = _hessian_from_manual_fd(neg_ll, theta)
    if H is not None:
        if verbose:
            _log.info("    [eigenvals] Tier 3 (manual FD) succeeded.")
        return np.linalg.eigvalsh(0.5 * (H + H.T))

    if verbose:
        _log.warning("    [eigenvals] All Hessian tiers failed.")
    return None


def check_hessian_pd(data: ObservationData, result: Dict[str, Any]) -> Optional[bool]:
    """
    Assess positive‑definiteness of the Hessian at the MLE using three tiers.

    Returns
    -------
    True  : Hessian is numerically positive definite
    False : at least one non‑positive eigenvalue
    None  : Hessian could not be computed
    """
    eigvals = compute_hessian_eigenvalues(data, result, verbose=False)
    if eigvals is None:
        return None
    return bool(np.all(eigvals > 0.0))


# ---------------------------------------------------------------------------
# Diagnostic plots (if matplotlib is available)
# ---------------------------------------------------------------------------
def kalman_innovations(data: ObservationData, result: Dict) -> np.ndarray:
    """Compute standardised Kalman innovations for the fitted model."""
    from mcarma.statespace import stationary_cov, transition_and_noise

    F, G, H_mat = result["F"], result["G"], result["H"]
    Sigma = result["Sigma"]
    mu = result["mu"]
    state_dim = F.shape[0]
    n = data.n

    C_eff = np.array([data.C_list[k] @ H_mat for k in range(n)])[:, 0, :]
    mu_sc = np.array([(data.C_list[k] @ mu)[0, 0] for k in range(n)])
    R_sc = np.array([data.R_list[k][0, 0] for k in range(n)])

    x = np.zeros(state_dim)
    P = stationary_cov(F, G, Sigma)
    innovations = np.zeros(n)
    S_vals = np.zeros(n)

    for k in range(n):
        if k > 0:
            dt = data.t_obs[k] - data.t_obs[k - 1]
            Phi, Qd = transition_and_noise(F, G, Sigma, dt)
            x = Phi @ x
            P = Phi @ P @ Phi.T + Qd
            P = 0.5 * (P + P.T)
        c = C_eff[k]
        Pc = P @ c
        S = float(c @ Pc) + R_sc[k]
        v = data.y_obs[k] - (mu_sc[k] + c @ x)
        K = Pc / max(S, 1e-30)
        x = x + K * v
        IKC = np.eye(state_dim) - np.outer(K, c)
        P = IKC @ P @ IKC.T + R_sc[k] * np.outer(K, K)
        P = 0.5 * (P + P.T)
        innovations[k] = v
        S_vals[k] = S

    return innovations / np.sqrt(np.maximum(S_vals, 1e-30))


def make_diagnostic_plots(
    data: ObservationData,
    result: Dict,
    p: int, q: int,
    out_prefix: str,
    y_raw: Optional[np.ndarray] = None,
    trend_info: Optional[List[Dict]] = None
) -> None:
    """Save light curve + innovation scatter plot and ACF plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from statsmodels.tsa.stattools import acf as sm_acf
    except ImportError as e:
        _log.warning(f"    [warn] Cannot produce plots ({e})")
        return

    try:
        std_innov = kalman_innovations(data, result)
    except Exception as e:
        _log.warning(f"    [warn] Innovation computation failed: {e}")
        return

    band = data.band
    t_obs = data.t_obs
    y_det = data.y_obs
    cmap = plt.cm.tab10
    from fits.process_quasars.data_utils import BAND_LABELS, eval_trend  # avoid circular import

    has_trend = (y_raw is not None) and (trend_info is not None)
    n_rows = 3 if has_trend else 2
    fig_height = 3.5 * n_rows

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height),
                             sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    row_idx = 0
    if has_trend:
        ax = axes[row_idx]; row_idx += 1
        for b in range(len(BAND_LABELS)):
            mask = band == b
            if not np.any(mask):
                continue
            color = cmap(b / len(BAND_LABELS))
            t_b = t_obs[mask]
            y_b = y_raw[mask]
            ax.scatter(t_b, y_b, s=4, alpha=0.5, color=color,
                       label=BAND_LABELS[b])
            t_fine = np.linspace(t_b.min(), t_b.max(), 400)
            tr_fine = eval_trend(t_fine, trend_info[b])
            ax.plot(t_fine, tr_fine, color=color, lw=1.2, alpha=0.85, linestyle="--")
        ax.invert_yaxis()
        ax.set_ylabel("Magnitude (raw)")
        ax.set_title(f"Raw light curve + polynomial trend — MCARMA({p},{q})")
        ax.legend(fontsize=8, ncol=len(BAND_LABELS), loc="upper right")
        ax.grid(True, alpha=0.25)

    ax = axes[row_idx]; row_idx += 1
    for b in range(len(BAND_LABELS)):
        mask = band == b
        if not np.any(mask):
            continue
        ax.scatter(t_obs[mask], y_det[mask], s=4, alpha=0.5,
                   color=cmap(b / len(BAND_LABELS)), label=BAND_LABELS[b])
    ax.axhline(0.0, color="k", lw=0.7, linestyle="--")
    ax.set_ylabel("Magnitude (centered)")
    ax.set_title("Mean-centered observations")
    ax.legend(fontsize=8, ncol=len(BAND_LABELS), loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[row_idx]; row_idx += 1
    for b in range(len(BAND_LABELS)):
        mask = band == b
        if not np.any(mask):
            continue
        ax.scatter(t_obs[mask], std_innov[mask], s=5, alpha=0.6,
                   color=cmap(b / len(BAND_LABELS)), label=BAND_LABELS[b])
    ax.axhline(0.0, color="k", lw=0.8, linestyle="--")
    ax.axhline(2.0, color="grey", lw=0.6, linestyle=":")
    ax.axhline(-2.0, color="grey", lw=0.6, linestyle=":")
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$v_k / \sqrt{S_k}$")
    ax.set_title("Standardised Kalman innovations")
    ax.legend(fontsize=8, ncol=len(BAND_LABELS), loc="upper right")
    ax.grid(True, alpha=0.25)

    fig.savefig(f"{out_prefix}_innovations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ACF plot
    n_lags = 30
    fig2, axes2 = plt.subplots(1, len(BAND_LABELS), figsize=(16, 3), sharey=True)
    for b in range(len(BAND_LABELS)):
        ax = axes2[b]
        mask = band == b
        resid_b = std_innov[mask]
        if len(resid_b) < n_lags + 2:
            ax.set_title(f"{BAND_LABELS[b]}\n(too few obs)")
            continue
        try:
            acf_vals, ci = sm_acf(resid_b, nlags=n_lags, alpha=0.05, fft=True)
            lags = np.arange(len(acf_vals))
            ax.bar(lags, acf_vals, color=cmap(b / len(BAND_LABELS)), alpha=0.7)
            ax.fill_between(lags, ci[:, 0] - acf_vals, ci[:, 1] - acf_vals,
                            alpha=0.2, color=cmap(b / len(BAND_LABELS)))
            ax.axhline(0, color="k", lw=0.8)
        except Exception as exc:
            ax.set_title(f"{BAND_LABELS[b]}\nACF failed: {exc}")
            continue
        ax.set_title(BAND_LABELS[b])
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Lag")
        ax.grid(True, alpha=0.2)
    axes2[0].set_ylabel("ACF")
    plt.suptitle(f"Innovations ACF — MCARMA({p},{q})", fontsize=11)
    plt.tight_layout()
    fig2.savefig(f"{out_prefix}_acf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"    Saved {out_prefix}_innovations.png and {out_prefix}_acf.png")