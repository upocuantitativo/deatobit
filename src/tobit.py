"""
Censored regression models used to explain DEA scores.

  * ``Tobit`` - maximum-likelihood Type-I Tobit (Amemiya, 1984) with optional
    left and right censoring. The model is estimated by direct numerical
    maximisation of the censored log-likelihood.
  * ``bootstrap_truncated`` - second-stage truncated regression with the
    bootstrap correction proposed by Simar and Wilson (2007).

Both estimators take a design matrix ``X`` (without intercept) and a vector
``y`` of efficiency scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass
class TobitResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    sigma: float
    loglik: float
    n_obs: int
    r2_mcfadden: float
    r2_pseudo: float
    r2_adjusted: float
    summary_table: pd.DataFrame


def _opg_se(beta, log_sigma, X, y, lower, upper):
    """Per-observation gradient of the censored normal log-likelihood,
    aggregated as the outer-product-of-gradients covariance estimator.
    Returns standard errors for [beta_0, ..., beta_{k-1}, log_sigma]."""
    sigma = float(np.exp(log_sigma))
    n, k = X.shape
    xb = X @ beta
    z = (y - xb) / sigma
    z_u = (upper - xb) / sigma if np.isfinite(upper) else np.full(n, np.inf)
    z_l = (lower - xb) / sigma if np.isfinite(lower) else np.full(n, -np.inf)

    interior = (y > lower) & (y < upper)
    right = y >= upper
    left = y <= lower

    scores = np.zeros((n, k + 1))
    # Interior contribution.
    s_int_b = (z[interior, None] * X[interior]) / sigma
    s_int_s = z[interior] ** 2 - 1.0  # derivative wrt log sigma
    scores[interior, :k] = s_int_b
    scores[interior, k] = s_int_s

    # Right-censored contribution: d log(1 - Phi(z_u)) / d beta_j
    if right.any():
        zr = z_u[right]
        ratio = stats.norm.pdf(zr) / np.clip(1 - stats.norm.cdf(zr), 1e-300, None)
        scores[right, :k] = (ratio[:, None] * X[right]) / sigma
        scores[right, k] = ratio * zr

    # Left-censored contribution: d log Phi(z_l) / d beta_j
    if left.any():
        zl = z_l[left]
        ratio = stats.norm.pdf(zl) / np.clip(stats.norm.cdf(zl), 1e-300, None)
        scores[left, :k] = -(ratio[:, None] * X[left]) / sigma
        scores[left, k] = -ratio * zl

    G = scores.T @ scores
    try:
        cov = np.linalg.inv(G)
        return np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        return np.full(k + 1, np.nan)


def _tobit_loglik(params, X, y, lower, upper):
    k = X.shape[1]
    beta = params[:k]
    log_sigma = params[k]
    sigma = np.exp(log_sigma)
    xb = X @ beta
    z = (y - xb) / sigma
    cdf_l = stats.norm.cdf((lower - xb) / sigma)
    cdf_u = stats.norm.cdf((upper - xb) / sigma)
    pdf = stats.norm.pdf(z)

    ll = np.empty_like(y)
    interior = (y > lower) & (y < upper)
    left = y <= lower
    right = y >= upper
    ll[interior] = np.log(pdf[interior] / sigma + 1e-300)
    ll[left] = np.log(cdf_l[left] + 1e-300)
    ll[right] = np.log(1 - cdf_u[right] + 1e-300)
    return -ll.sum()


def fit_tobit(X: pd.DataFrame, y: pd.Series,
              lower: float = -np.inf, upper: float = 1.0,
              add_constant: bool = True) -> TobitResult:
    """Fit a Type-I Tobit by maximum likelihood.

    By default the dependent variable is right-censored at 1 (DEA upper
    bound). Variables are internally standardised to put the Hessian on a
    well-conditioned scale, then coefficients and standard errors are
    transformed back to the original units.
    """
    Xv = X.copy()
    if add_constant:
        Xv.insert(0, "Intercept", 1.0)
    Xa = Xv.values.astype(float)
    yv = y.values.astype(float)
    n, k = Xa.shape
    names = list(Xv.columns)

    # Standardise non-intercept columns for numerical stability.
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0, ddof=0)
    mu[0] = 0.0; sd[0] = 1.0  # leave intercept untouched
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (Xa - mu) / sd

    beta0, *_ = np.linalg.lstsq(Xs, yv, rcond=None)
    resid = yv - Xs @ beta0
    sigma0 = max(resid.std(ddof=1), 1e-3)
    p0 = np.concatenate([beta0, [np.log(sigma0)]])

    res = optimize.minimize(_tobit_loglik, p0, args=(Xs, yv, lower, upper),
                            method="BFGS", options={"gtol": 1e-7})
    params = res.x
    beta_s = params[:k]
    log_sigma = params[k]
    sigma = float(np.exp(log_sigma))
    loglik = -res.fun

    # Outer-product-of-gradients (BHHH/OPG) estimator of the covariance.
    # Numerically stable when the optimum sits near boundary observations
    # because it only relies on per-observation scores.
    se_s = _opg_se(beta_s, log_sigma, Xs, yv, lower, upper)[:k]

    # Map standardised coefficients back to the original scale.
    beta = beta_s / sd
    beta[0] = beta_s[0] - np.sum(beta_s[1:] * mu[1:] / sd[1:])
    se = se_s / sd
    se[0] = np.nan  # intercept SE is not directly identifiable here

    # Standardised t-values are scale-free, so we report z-statistics from
    # the standardised solution and re-attach them to the original beta.
    tvals = np.empty(k)
    for j in range(k):
        denom = se_s[j] if (j > 0 and se_s[j] > 0) else np.nan
        tvals[j] = beta_s[j] / denom if denom and not np.isnan(denom) else np.nan
    pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals)))

    # Intercept t/p from the standardised fit (uses the model intercept).
    if se_s[0] > 0:
        tvals[0] = beta_s[0] / se_s[0]
        pvals[0] = 2 * (1 - stats.norm.cdf(abs(tvals[0])))

    null_p0 = np.array([yv.mean(), np.log(yv.std(ddof=1))])
    Xnull = np.ones((n, 1))
    res_null = optimize.minimize(_tobit_loglik, null_p0,
                                 args=(Xnull, yv, lower, upper),
                                 method="L-BFGS-B")
    ll_null = -res_null.fun
    r2_mcf = 1 - loglik / ll_null if ll_null != 0 else np.nan

    yhat = Xa @ beta
    ss_res = ((yv - yhat) ** 2).sum()
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    summary = pd.DataFrame({
        "coef": beta, "std_err": se, "t": tvals, "p_value": pvals,
    }, index=names)

    return TobitResult(
        params=pd.Series(beta, index=names),
        bse=pd.Series(se, index=names),
        tvalues=pd.Series(tvals, index=names),
        pvalues=pd.Series(pvals, index=names),
        sigma=sigma,
        loglik=loglik,
        n_obs=n,
        r2_mcfadden=r2_mcf,
        r2_pseudo=r2,
        r2_adjusted=r2_adj,
        summary_table=summary,
    )


def bootstrap_truncated(X: pd.DataFrame, y: pd.Series, n_boot: int = 2000,
                        seed: int = 42) -> pd.DataFrame:
    """Simar-Wilson (2007, Algorithm 1) bias-corrected truncated regression.

    The original DEA scores are biased upwards; this routine resamples the
    fitted truncated-normal distribution to compute bootstrap standard
    errors and confidence intervals.
    """
    rng = np.random.default_rng(seed)
    Xv = X.copy()
    Xv.insert(0, "Intercept", 1.0)
    Xa = Xv.values.astype(float)
    yv = y.values.astype(float)
    n, k = Xa.shape

    def _trunc_fit(yy):
        def neg_ll(p):
            beta = p[:k]
            sigma = np.exp(p[k])
            xb = Xa @ beta
            z = (yy - xb) / sigma
            denom = 1 - stats.norm.cdf((1 - xb) / sigma)
            ll = (np.log(stats.norm.pdf(z) / sigma + 1e-300)
                  - np.log(denom + 1e-300))
            return -ll.sum()

        beta0, *_ = np.linalg.lstsq(Xa, yy, rcond=None)
        s0 = max((yy - Xa @ beta0).std(ddof=1), 1e-3)
        p0 = np.concatenate([beta0, [np.log(s0)]])
        r = optimize.minimize(neg_ll, p0, method="L-BFGS-B")
        return r.x[:k], np.exp(r.x[k])

    beta_hat, sigma_hat = _trunc_fit(yv)

    boots = np.empty((n_boot, k))
    for b in range(n_boot):
        # Draw from truncated normal with truncation point at 1.
        eps = stats.truncnorm.rvs(
            a=-np.inf, b=(1 - Xa @ beta_hat) / sigma_hat,
            loc=0, scale=sigma_hat, size=n, random_state=rng,
        )
        y_star = Xa @ beta_hat + eps
        b_b, _ = _trunc_fit(y_star)
        boots[b] = b_b

    bias = boots.mean(axis=0) - beta_hat
    beta_corr = beta_hat - bias
    se = boots.std(axis=0, ddof=1)
    ci_lo = np.quantile(boots, 0.025, axis=0)
    ci_hi = np.quantile(boots, 0.975, axis=0)

    return pd.DataFrame({
        "coef": beta_hat, "bias_corrected": beta_corr, "boot_se": se,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
    }, index=list(Xv.columns))
