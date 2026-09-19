"""
mcarma/jax_loglik.py
====================
Analytic (autodiff) value-and-gradient for the MCARMA negative log-likelihood.

See docs/METHODS.Rmd sec 13 (JAX analytic gradient; same Lyapunov Qd as numpy).

Motivation
----------
``fit.fit`` calls ``scipy.optimize.minimize`` with NO ``jac=``, so BFGS
finite-differences the gradient: ~ (n_theta + 1) full Kalman passes PER line
search step. For d=6, p=2, q=1 that is 46 loglik evals/step, and with the tight
``gtol=1e-9`` every fit burns the full ``maxiter`` without converging. This
module reimplements the EXACT same forward objective as ``fit.neg_loglik`` in
JAX so ``jax.value_and_grad`` returns the analytic gradient in ONE pass
(reverse-mode), and exposes a ``(value, grad)`` callable that scipy accepts via
``jac=True``. Combined with a looser ``gtol`` this removes the FD blow-up.

What is mirrored (and what is not)
----------------------------------
The forward map ``theta -> -loglik + penalties`` is reproduced exactly:
  unpack_params_jones -> build_state_space -> stationary_cov (Lyapunov) ->
  Phi=expm(F dt), Q=P0-Phi P0 Phi^T -> scalar Kalman innovations filter.
Supported priors (the production constraint set): the log-Cholesky Sigma prior
(symmetric / one-sided floor / two-sided box), the AR & MA observable-band
log-frequency walls, and the AR damping floor. The partial-pooling and
pole-zero priors are NOT implemented here; callers requesting them must fall
back to the finite-difference path (``supports_prior`` returns False).

Design
------
Everything that does not depend on ``theta`` (observation times, band indices,
R, the unique time-gap structure, the prior band edges) is precomputed ONCE in
numpy by ``build_objective`` and closed over as static constants. Only ``theta``
flows through JAX. The continuous-time eigen-decomposition is deliberately
avoided (``jnp.linalg.eig`` has no general autodiff rule); the large-dt regime
is handled implicitly because ``expm(F dt) -> 0`` for stable F, giving
``Q -> P0`` with no eigenvalue gate needed. Restricted to p in {1,2}, q in
{0,1} (the orders the study uses), so the Jones factor -> coefficient map is
written out directly rather than via convolution.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm

jax.config.update("jax_enable_x64", True)

_LOG2PI = float(np.log(2.0 * np.pi))


def supports(p, q, ar_pool_lambda=None, ma_pool_lambda=None,
             pole_zero_lambda=None):
    """Whether the JAX objective can represent this (p, q) + prior config.

    Restricted to p in {1,2}, q in {0,1}; the pooling and pole-zero priors are
    not implemented (caller must use the finite-difference path for those).
    """
    if p not in (1, 2) or q not in (0, 1):
        return False
    if ar_pool_lambda or ma_pool_lambda or pole_zero_lambda:
        return False
    return True


def _precompute_static(data, p, q, slopes):
    """Numpy precompute of every theta-independent quantity used in the loop.

    Returns a dict of plain numpy arrays / python scalars.
    """
    t_obs = np.asarray(data.t_obs, dtype=float)
    y_obs = np.asarray(data.y_obs, dtype=float)
    n_obs = len(t_obs)
    d = int(data.d)

    # Band observed at each epoch (one scalar obs per epoch): argmax of the
    # (1, d) selector, matching kalman.loglik's band_idx.
    band_idx = np.argmax(np.asarray(data.C_list)[:, 0, :], axis=1).astype(int)
    R_arr = np.array([data.R_list[k][0, 0] for k in range(n_obs)], dtype=float)

    # Fixed deterministic per-band linear trend (folded into the mean offset).
    if slopes is not None:
        slopes = np.asarray(slopes, dtype=float).ravel()
        slope_term = slopes[band_idx] * t_obs
    else:
        slope_term = np.zeros(n_obs, dtype=float)

    # Unique time gaps. Each epoch k>=1 maps to either the identity slot (dt==0,
    # same timestamp across bands) or a unique-dt slot; epoch 0 uses identity.
    dts = np.array([t_obs[k] - t_obs[k - 1] for k in range(1, n_obs)])
    nonzero = dts > 0.0
    unique_dts = np.unique(np.round(dts[nonzero], 4)) if nonzero.any() \
        else np.array([], dtype=float)
    dt_to_slot = {round(float(u), 4): j for j, u in enumerate(unique_dts)}

    sidx = np.zeros(n_obs, dtype=int)  # slot 0 == identity/dt0; 1+j == unique j
    for k in range(1, n_obs):
        dt = float(t_obs[k] - t_obs[k - 1])
        sidx[k] = 0 if dt == 0.0 else (1 + dt_to_slot[round(dt, 4)])

    # Cholesky tril index bookkeeping (lower-triangular, row-major like numpy).
    rows, cols = np.tril_indices(d)
    diag_pos = np.where(rows == cols)[0]   # positions of L_ii inside the block

    # ---- prior band edges (theta-independent) ----------------------------
    band = np.asarray(data.band) if getattr(data, "band", None) is not None \
        else band_idx
    perband_dt = [np.median(np.diff(np.sort(t_obs[band == b])))
                  for b in np.unique(band) if int(np.sum(band == b)) > 1]
    dt_med_band = float(max(perband_dt)) if perband_dt \
        else float(np.median(np.diff(t_obs)))
    dt_med_glob = float(np.median(np.diff(t_obs))) if n_obs > 1 else 1.0
    T_span = float(t_obs.max() - t_obs.min())

    c_cadence_band, c_span = 0.7, 0.2          # neg_loglik's band-prior call
    log_min = np.log(1.0 / (c_span * T_span))
    log_max = np.log(1.0 / (c_cadence_band * dt_med_band))
    log_floor = np.log(1.0 / (c_span * T_span))   # damping floor (c_span only)

    return dict(
        n_obs=n_obs, d=d, p=p, q=q,
        y_obs=y_obs, R_arr=R_arr, band_idx=band_idx, slope_term=slope_term,
        unique_dts=unique_dts, sidx=sidx,
        rows=rows, cols=cols, diag_pos=diag_pos,
        log_min=log_min, log_max=log_max, log_floor=log_floor,
        dt_med_band=dt_med_band, dt_med_glob=dt_med_glob, T_span=T_span,
    )


def _quad_root_mag_re(a1, a2):
    """|root| and |Re root| for both roots of monic s^2 + a1 s + a2.

    Complex arithmetic so the real (overdamped, disc>0) and complex
    (underdamped, disc<0) branches are handled uniformly, matching
    optimizer_utils._ar_roots_from_factor exactly.
    """
    disc = (a1 * a1 - 4.0 * a2).astype(jnp.complex128)
    sq = jnp.sqrt(disc)
    r1 = 0.5 * (-a1 + sq)
    r2 = 0.5 * (-a1 - sq)
    mags = jnp.stack([jnp.abs(r1), jnp.abs(r2)])
    res = jnp.stack([jnp.abs(jnp.real(r1)), jnp.abs(jnp.real(r2))])
    return mags, res


def _wall_log(mags, log_min, log_max):
    """sum_k h(log_min - log m_k)^2 + h(log m_k - log_max)^2 (soft band walls)."""
    lm = jnp.log(jnp.maximum(mags, 1e-300))
    lo = jnp.maximum(log_min - lm, 0.0)
    hi = jnp.maximum(lm - log_max, 0.0)
    return jnp.sum(lo * lo + hi * hi)


def build_objective(data, p, q, slopes=None,
                    chol_ridge_lambda=0.0, chol_ridge_center=0.0,
                    ar_band_lambda=None, ma_band_lambda=None,
                    chol_ridge_onesided=False, chol_ridge_center_hi=None,
                    ar_damping_lambda=None, corr_ridge_lambda=0.0,
                    diag_load_lambda=0.0,
                    with_scores=False):
    """Build a jitted JAX ``theta -> (neg_loglik+penalties)`` and its grad.

    Returns ``(value_fn, value_and_grad_fn, hessian_fn)`` where each takes a
    single ``theta`` array (length n_ar+n_ma+n_chol+d) and returns jax scalars /
    arrays. Use ``scipy_objective`` for a numpy ``(float, ndarray)`` callable.

    With ``with_scores=True`` a fourth element is appended, a jitted
    ``ll_vec_fn(theta) -> (n_obs,)`` returning the per-observation
    log-likelihood contributions (no penalty), for the GIC score matrix.
    """
    if not supports(p, q):
        raise NotImplementedError(f"jax_loglik supports p in (1,2), q in (0,1); got ({p},{q})")

    s = _precompute_static(data, p, q, slopes)
    d, n_obs = s["d"], s["n_obs"]
    dp = d * p
    n_ar, n_ma = d * p, d * q
    n_chol = d * (d + 1) // 2

    # Static jnp constants.
    y_obs = jnp.asarray(s["y_obs"])
    R_arr = jnp.asarray(s["R_arr"])
    band_idx = jnp.asarray(s["band_idx"])
    slope_term = jnp.asarray(s["slope_term"])
    unique_dts = jnp.asarray(s["unique_dts"]) if len(s["unique_dts"]) else None
    sidx = jnp.asarray(s["sidx"])
    rows_np, cols_np, diag_pos_np = s["rows"], s["cols"], s["diag_pos"]
    rows = jnp.asarray(rows_np)
    cols = jnp.asarray(cols_np)
    diag_pos = jnp.asarray(diag_pos_np)
    log_min, log_max, log_floor = s["log_min"], s["log_max"], s["log_floor"]

    eye_dp = jnp.eye(dp)
    eye_d = jnp.eye(d)
    # Kron operator pieces for the Lyapunov solve are theta-dependent (F), so
    # built inside; only the identities are constant.
    eye_n2_kron_left = jnp.eye(dp)

    # Resolve prior config to concrete floats (None -> 0 / off).
    cr_lambda = float(chol_ridge_lambda or 0.0)
    cr_center = jnp.asarray(np.asarray(chol_ridge_center, dtype=float))
    cr_onesided = bool(chol_ridge_onesided)
    cr_center_hi = (jnp.asarray(np.asarray(chol_ridge_center_hi, dtype=float))
                    if chol_ridge_center_hi is not None else None)
    arb = float(ar_band_lambda or 0.0)
    mab = float(ma_band_lambda or 0.0)
    ard = float(ar_damping_lambda or 0.0)
    corr_lambda = float(corr_ridge_lambda or 0.0)
    # Diagonal loading (Zhirui 2026-08-13): a MODEL regularizer, not a penalty.
    # Replaces the driving covariance V with V + dl_lambda*mean(diag(V))*I inside
    # the likelihood, lifting the smallest eigenvalue off the near-singular cross-
    # band boundary. Relative (mean-diagonal) scaling makes dl_lambda dimensionless
    # and cell-scale-free. Default 0 = off = byte-identical forward map.
    dl_lambda = float(diag_load_lambda or 0.0)
    # Static strict-upper-triangle indices of the d x d correlation matrix, used
    # by the optional cross-band correlation ridge below.
    _iu = np.triu_indices(d, 1)
    corr_iu_r = jnp.asarray(_iu[0])
    corr_iu_c = jnp.asarray(_iu[1])
    use_band = (ar_band_lambda is not None) or (ma_band_lambda is not None)

    def _unpack(theta):
        ar_log = theta[:n_ar]
        ma_log = theta[n_ar:n_ar + n_ma]
        chol = theta[n_ar + n_ma:n_ar + n_ma + n_chol]
        mu = theta[n_ar + n_ma + n_chol:]
        return ar_log, ma_log, chol, mu

    def _state_space(ar_log, ma_log, chol):
        # AR coefficient columns (d, p): build_state_space uses coeffs[1:][::-1].
        if p == 2:
            a1 = jnp.exp(ar_log[0::2])           # (d,)
            a2 = jnp.exp(ar_log[1::2])           # (d,)
            ar_cols = [a2, a1]                   # lag 0, lag 1
        else:  # p == 1
            a0 = jnp.exp(ar_log)
            ar_cols = [a0]
        # MA coefficient columns (d, q).
        ma_cols = []
        if q == 1:
            ma_cols = [jnp.exp(ma_log)]          # b0

        # F (dp, dp): super-diagonal identity blocks + bottom -diag(ar) blocks.
        F = jnp.zeros((dp, dp))
        for i in range(p - 1):
            F = F.at[i * d:(i + 1) * d, (i + 1) * d:(i + 2) * d].set(eye_d)
        for lag in range(p):
            F = F.at[(p - 1) * d:p * d, lag * d:(lag + 1) * d].set(
                -jnp.diag(ar_cols[lag]))
        G = jnp.zeros((dp, d)).at[(p - 1) * d:p * d, :].set(eye_d)
        H = jnp.zeros((d, dp)).at[:, 0:d].set(eye_d)
        for lag in range(q):
            H = H.at[:, (lag + 1) * d:(lag + 2) * d].set(jnp.diag(ma_cols[lag]))

        # V = L L^T from log-Cholesky block (diag stores log(L_ii^2)).
        entry = jnp.where(rows == cols, jnp.exp(0.5 * chol), chol)
        L = jnp.zeros((d, d)).at[rows, cols].set(entry)
        V = L @ L.T
        # Optional diagonal loading (model regularizer; see dl_lambda note above).
        if dl_lambda > 0.0:
            V = V + dl_lambda * (jnp.trace(V) / d) * eye_d
        return F, G, H, V

    def _stationary_cov(F, G, V):
        Qc = G @ V @ G.T                                   # (dp, dp)
        # Solve F P + P F^T = -Qc via column-stack vec + Kronecker operator.
        # A tiny ridge on A conditions the BACKWARD pass: reverse-mode through
        # jnp.linalg.solve re-solves the transposed system with the same A, which
        # is near-singular when F approaches marginal stability (eigval -> 0).
        # 1e-10 is negligible forward (Qc entries are O(1), so P0 shifts ~1e-11,
        # below the 1e-12 P0 jitter) but bounds the gradient there. This is the
        # lesson from the failed torch port, where an un-ridged solve NaN'd the
        # gradient and a 1e-6 ridge over-regularized and biased the landscape.
        A = jnp.kron(eye_n2_kron_left, F) + jnp.kron(F, eye_dp)
        A = A + 1e-10 * jnp.eye(dp * dp)
        rhs = -jnp.reshape(Qc, (dp * dp,), order="F")
        vecP = jnp.linalg.solve(A, rhs)
        P0 = jnp.reshape(vecP, (dp, dp), order="F")
        P0 = 0.5 * (P0 + P0.T) + 1e-12 * eye_dp
        return P0

    def _transitions(F, P0):
        if unique_dts is None:
            Phi_all = eye_dp[None]
            Q_all = jnp.zeros((1, dp, dp))
        else:
            Phi_u = jax.vmap(lambda dt: jax_expm(F * dt))(unique_dts)  # (U,n,n)
            PhiP = jnp.einsum("uij,jk->uik", Phi_u, P0)
            Q_u = P0[None] - jnp.einsum("uik,ulk->uil", PhiP, Phi_u)
            Q_u = 0.5 * (Q_u + jnp.transpose(Q_u, (0, 2, 1)))
            Phi_all = jnp.concatenate([eye_dp[None], Phi_u], axis=0)
            Q_all = jnp.concatenate([jnp.zeros((1, dp, dp)), Q_u], axis=0)
        return Phi_all[sidx], Q_all[sidx]    # (n_obs, n, n) each

    def _kalman(Phi_seq, Q_seq, H, mu, P0):
        c_seq = H[band_idx]                              # (n_obs, dp)
        mu_seq = mu[band_idx] + slope_term               # (n_obs,)

        def step(carry, inp):
            x, P = carry
            Phi, Q, c, m, R, y = inp
            x_pred = Phi @ x
            P_pred = Phi @ P @ Phi.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)
            y_pred = m + c @ x_pred
            v = y - y_pred
            Pc = P_pred @ c
            S = c @ Pc + R
            ll_k = -0.5 * (jnp.log(S) + v * v / S + _LOG2PI)
            K = Pc / S
            x_new = x_pred + K * v
            IKC = eye_dp - jnp.outer(K, c)
            P_new = IKC @ P_pred @ IKC.T + R * jnp.outer(K, K)
            P_new = 0.5 * (P_new + P_new.T)
            return (x_new, P_new), ll_k

        x0 = jnp.zeros(dp)
        xs = (Phi_seq, Q_seq, c_seq, mu_seq, R_arr, y_obs)
        (_, _), ll_seq = jax.lax.scan(step, (x0, P0), xs)
        # Return the PER-OBSERVATION log-likelihood contributions (one scalar
        # per band-epoch innovation). value_fn sums them; the GIC score matrix
        # (mcarma/gic.py) needs the vector before the sum.
        return ll_seq

    def _priors(ar_log, ma_log, chol):
        pen = 0.0
        # ---- AR/MA observable-band walls + AR damping floor ----
        if use_band or ard > 0.0:
            ar_mags_all = []
            ar_re_all = []
            ma_mags_all = []
            for b in range(d):
                if p == 2:
                    a1 = jnp.exp(ar_log[2 * b]); a2 = jnp.exp(ar_log[2 * b + 1])
                    mags, res = _quad_root_mag_re(a1, a2)
                    ar_mags_all.append(mags); ar_re_all.append(res)
                else:
                    a0 = jnp.exp(ar_log[b])
                    ar_mags_all.append(jnp.reshape(a0, (1,)))
                    ar_re_all.append(jnp.reshape(a0, (1,)))
                if q == 1:
                    b0 = jnp.exp(ma_log[b])
                    ma_mags_all.append(jnp.reshape(1.0 / b0, (1,)))
            if use_band:
                if arb > 0.0:
                    for mags in ar_mags_all:
                        pen = pen + 0.5 * arb * _wall_log(mags, log_min, log_max)
                if mab > 0.0:
                    for mags in ma_mags_all:
                        pen = pen + 0.5 * mab * _wall_log(mags, log_min, log_max)
            if ard > 0.0:
                for res in ar_re_all:
                    lr = jnp.log(jnp.maximum(res, 1e-12))
                    dev = jnp.maximum(log_floor - lr, 0.0)
                    pen = pen + 0.5 * ard * jnp.sum(dev * dev)
        # ---- log-Cholesky Sigma prior ----
        if cr_lambda > 0.0:
            diag = chol[diag_pos]
            logvar = diag   # 2026-07 fix: param is 2*log L_bb (already log-variance); was 2.0*diag (see fit.py:987 note)
            if cr_onesided:
                dev = jnp.maximum(cr_center - logvar, 0.0)
                pen = pen + 0.5 * cr_lambda * jnp.sum(dev * dev)
                if cr_center_hi is not None:
                    dev_hi = jnp.maximum(logvar - cr_center_hi, 0.0)
                    pen = pen + 0.5 * cr_lambda * jnp.sum(dev_hi * dev_hi)
            else:
                dev = logvar - cr_center
                pen = pen + 0.5 * cr_lambda * jnp.sum(dev * dev)
        # ---- cross-band correlation ridge (Zhirui 2026-08-11) ----
        # Shrinks the OFF-diagonal correlations of Sigma toward zero, pulling the
        # fit away from the near-singular cross-band corner that carries the
        # +rho over-estimation. Mirror of fit.py neg_loglik; the only correlation-
        # touching penalty. Default off (lambda 0) so the forward map is
        # byte-identical to production. Penalty = 0.5*lambda*sum_{b<c} R_bc^2.
        if corr_lambda > 0.0:
            entry = jnp.where(rows == cols, jnp.exp(0.5 * chol), chol)
            L = jnp.zeros((d, d)).at[rows, cols].set(entry)
            V = L @ L.T
            sd = jnp.sqrt(jnp.clip(jnp.diag(V), 1e-300, None))
            R = V / jnp.outer(sd, sd)
            off = R[corr_iu_r, corr_iu_c]
            pen = pen + 0.5 * corr_lambda * jnp.sum(off * off)
        return pen

    def _ll_vec(theta):
        """Per-observation log-likelihood contributions (n_obs,), no penalty.

        This is the model score seam for the GIC bias correction: its Jacobian
        in theta is the per-observation score matrix whose outer product is the
        Konishi-Kitagawa 'meat' matrix K.
        """
        ar_log, ma_log, chol, mu = _unpack(theta)
        F, G, H, V = _state_space(ar_log, ma_log, chol)
        P0 = _stationary_cov(F, G, V)
        Phi_seq, Q_seq = _transitions(F, P0)
        return _kalman(Phi_seq, Q_seq, H, mu, P0)

    def value_fn(theta):
        ar_log, ma_log, chol, mu = _unpack(theta)
        F, G, H, V = _state_space(ar_log, ma_log, chol)
        P0 = _stationary_cov(F, G, V)
        Phi_seq, Q_seq = _transitions(F, P0)
        ll = jnp.sum(_kalman(Phi_seq, Q_seq, H, mu, P0))
        pen = _priors(ar_log, ma_log, chol)
        return -ll + pen

    value_fn_j = jax.jit(value_fn)
    vg_j = jax.jit(jax.value_and_grad(value_fn))
    # Analytic Hessian via forward-over-reverse (jacfwd of jacrev): the exact
    # curvature of the SAME penalized objective, in one autodiff pass. Replaces
    # the study's FD/Richardson Hessian (approx_hessian) -- no step-size or
    # truncation error (a documented source of the optimistic SEs -> coverage),
    # and far cheaper than ~n_theta^2 likelihood evals. The hinge-squared priors
    # have a well-defined Hessian away from their kinks (zero curvature below the
    # wall, 2*lambda above), so this is exact at an interior optimum. At an
    # optimum resting ON a wall the objective is still twice differentiable, but
    # the full Hessian is no longer the matrix to test for definiteness -- see
    # inference.polished_cov.
    hess_j = jax.jit(jax.jacfwd(jax.jacrev(value_fn)))
    if with_scores:
        ll_vec_j = jax.jit(_ll_vec)
        return value_fn_j, vg_j, hess_j, ll_vec_j
    return value_fn_j, vg_j, hess_j


def make_scipy_objective(data, p, q, slopes=None, sanitize_value=1e10, **prior):
    """Return a numpy ``f(theta, *ignored) -> (float, ndarray)`` for scipy.

    Compatible with ``scipy.optimize.minimize(..., jac=True)``. Non-finite
    values/gradients (theta wandered into an unstable / degenerate region) are
    mapped to ``(sanitize_value, zeros)`` so the optimiser rejects the step,
    mirroring ``fit.neg_loglik``'s 1e10 sentinel. Extra positional ``*ignored``
    lets scipy pass its usual ``args`` tuple harmlessly.
    """
    _, vg, _ = build_objective(data, p, q, slopes=slopes, **prior)
    n_theta = p * data.d + q * data.d + data.d * (data.d + 1) // 2 + data.d

    def f(theta, *ignored):
        val, grad = vg(jnp.asarray(theta, dtype=jnp.float64))
        val = float(val)
        grad = np.asarray(grad, dtype=float)
        if not np.isfinite(val) or not np.isfinite(grad).all():
            return float(sanitize_value), np.zeros(n_theta)
        return val, grad

    return f


def make_scipy_hessian(data, p, q, slopes=None, **prior):
    """Return a numpy ``f(theta, *ignored) -> (n_theta, n_theta)`` analytic Hessian.

    Drop-in replacement for the study's ``approx_hessian`` (central-FD) at the
    optimum: exact second derivatives of the penalized objective via autodiff,
    symmetrised. Invert (or pinv) for the parameter covariance, then apply the
    same change-of-variables the FD path uses.
    """
    _, _, hess = build_objective(data, p, q, slopes=slopes, **prior)

    def f(theta, *ignored):
        H = np.asarray(hess(jnp.asarray(theta, dtype=jnp.float64)), dtype=float)
        return 0.5 * (H + H.T)

    return f


def make_scipy_hessian_hvp(data, p, q, slopes=None, **prior):
    """Memory-safe analytic Hessian via a Hessian-vector-product loop.

    Same exact penalized-objective Hessian as ``make_scipy_hessian``, but built
    column-by-column with ``jax.jvp`` of the gradient (one tangent at a time)
    instead of the vectorized ``jacfwd(jacrev)``, which materializes all n_theta
    tangents through the long Kalman scan at once and OOMs at study scale. This
    is O(n_theta) forward-over-reverse passes, each as memory-light as one
    gradient, and replaces the O(n_theta^2) central-FD ``approx_hessian`` (~660s
    for a 35-param fit) with a handful of seconds -- exact, no step-size error.
    Returns a numpy ``(n, n)`` symmetrised Hessian; non-finite -> zeros so the
    caller's pinv/PD guard handles it.
    """
    value_fn_j, _, _ = build_objective(data, p, q, slopes=slopes, **prior)
    grad_fn = jax.grad(value_fn_j)
    hvp = jax.jit(lambda x, v: jax.jvp(grad_fn, (x,), (v,))[1])
    n_theta = p * data.d + q * data.d + data.d * (data.d + 1) // 2 + data.d
    eye = jnp.eye(n_theta, dtype=jnp.float64)

    def f(theta, *ignored):
        x = jnp.asarray(theta, dtype=jnp.float64)
        cols = [np.asarray(hvp(x, eye[i]), dtype=float) for i in range(n_theta)]
        H = np.array(cols)
        if not np.isfinite(H).all():
            return np.zeros((n_theta, n_theta))
        return 0.5 * (H + H.T)

    return f
