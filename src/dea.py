"""
Data Envelopment Analysis (DEA) models.

Implements the four formulations used in the empirical study:

  * CCR (Charnes-Cooper-Rhodes, 1978) - constant returns to scale
  * BCC (Banker-Charnes-Cooper, 1984) - variable returns to scale
  * Super-efficiency (Andersen and Petersen, 1993)
  * Slack-Based Measure - SBM (Tone, 2001)

All models are solved as linear programs through ``scipy.optimize.linprog``
(HiGHS dual-simplex). The interface accepts numpy arrays of shape
``(n_dmus, n_inputs)`` and ``(n_dmus, n_outputs)`` and returns efficiency
scores as a one-dimensional numpy array of length ``n_dmus``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

_LP_OPTS = {"presolve": True, "disp": False}
_METHOD = "highs"


def _ccr_input_oriented(theta_idx: int, X: np.ndarray, Y: np.ndarray,
                        exclude: int | None = None) -> float:
    """Solve the input-oriented CCR envelopment LP for a single DMU.

    min  theta
    s.t. sum_j lambda_j * x_ij  <= theta * x_i0    (inputs)
         sum_j lambda_j * y_rj  >= y_r0            (outputs)
         lambda_j >= 0
    If ``exclude`` is given, that DMU is removed from the reference set
    (super-efficiency formulation).
    """
    n, m = X.shape
    _, s = Y.shape
    idx = np.arange(n)
    if exclude is not None:
        idx = idx[idx != exclude]

    n_ref = len(idx)
    # variables: [theta, lambda_1, ..., lambda_n_ref]
    c = np.zeros(1 + n_ref)
    c[0] = 1.0

    # Input constraints: sum lambda_j x_ij - theta * x_i0 <= 0
    A_in = np.zeros((m, 1 + n_ref))
    A_in[:, 0] = -X[theta_idx]
    A_in[:, 1:] = X[idx].T
    b_in = np.zeros(m)

    # Output constraints: -sum lambda_j y_rj <= -y_r0
    A_out = np.zeros((s, 1 + n_ref))
    A_out[:, 1:] = -Y[idx].T
    b_out = -Y[theta_idx]

    A_ub = np.vstack([A_in, A_out])
    b_ub = np.concatenate([b_in, b_out])

    bounds = [(None, None)] + [(0.0, None)] * n_ref
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method=_METHOD, options=_LP_OPTS)
    if not res.success:
        return np.nan
    return float(res.x[0])


def _bcc_input_oriented(theta_idx: int, X: np.ndarray, Y: np.ndarray) -> float:
    """Input-oriented BCC adds the convexity constraint sum lambda_j = 1."""
    n, m = X.shape
    _, s = Y.shape
    c = np.zeros(1 + n)
    c[0] = 1.0

    A_in = np.zeros((m, 1 + n))
    A_in[:, 0] = -X[theta_idx]
    A_in[:, 1:] = X.T
    b_in = np.zeros(m)

    A_out = np.zeros((s, 1 + n))
    A_out[:, 1:] = -Y.T
    b_out = -Y[theta_idx]

    A_ub = np.vstack([A_in, A_out])
    b_ub = np.concatenate([b_in, b_out])

    A_eq = np.zeros((1, 1 + n))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    bounds = [(None, None)] + [(0.0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method=_METHOD, options=_LP_OPTS)
    if not res.success:
        return np.nan
    return float(res.x[0])


def _sbm(theta_idx: int, X: np.ndarray, Y: np.ndarray) -> float:
    """Tone (2001) Slack-Based Measure under constant returns to scale.

    The fractional program is linearised using the Charnes-Cooper transform.
    Variables: [t, Lambda_1..Lambda_n, S_minus_1..S_minus_m, S_plus_1..S_plus_s]
    """
    n, m = X.shape
    _, s = Y.shape
    nv = 1 + n + m + s

    # Objective: min t - (1/m) sum (S_minus_i / x_i0)
    c = np.zeros(nv)
    c[0] = 1.0
    x0 = X[theta_idx]
    y0 = Y[theta_idx]
    eps = 1e-12
    for i in range(m):
        c[1 + n + i] = -(1.0 / m) / max(x0[i], eps)

    # Equality constraint: t + (1/s) sum (S_plus_r / y_r0) = 1
    A_eq_norm = np.zeros((1, nv))
    A_eq_norm[0, 0] = 1.0
    for r in range(s):
        A_eq_norm[0, 1 + n + m + r] = (1.0 / s) / max(y0[r], eps)

    # Input equalities: sum Lambda_j x_ij + S_minus_i - t * x_i0 = 0
    A_eq_x = np.zeros((m, nv))
    for i in range(m):
        A_eq_x[i, 0] = -x0[i]
        A_eq_x[i, 1:1 + n] = X[:, i]
        A_eq_x[i, 1 + n + i] = 1.0

    # Output equalities: sum Lambda_j y_rj - S_plus_r - t * y_r0 = 0
    A_eq_y = np.zeros((s, nv))
    for r in range(s):
        A_eq_y[r, 0] = -y0[r]
        A_eq_y[r, 1:1 + n] = Y[:, r]
        A_eq_y[r, 1 + n + m + r] = -1.0

    A_eq = np.vstack([A_eq_norm, A_eq_x, A_eq_y])
    b_eq = np.concatenate([[1.0], np.zeros(m), np.zeros(s)])

    bounds = [(eps, None)] + [(0.0, None)] * (n + m + s)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method=_METHOD, options=_LP_OPTS)
    if not res.success:
        return np.nan
    return float(res.fun)


def ccr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([_ccr_input_oriented(j, X, Y) for j in range(X.shape[0])])


def bcc(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([_bcc_input_oriented(j, X, Y) for j in range(X.shape[0])])


def super_efficiency(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Andersen-Petersen (1993): allows scores >= 1 for efficient DMUs."""
    out = np.empty(X.shape[0])
    for j in range(X.shape[0]):
        out[j] = _ccr_input_oriented(j, X, Y, exclude=j)
    return out


def sbm(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([_sbm(j, X, Y) for j in range(X.shape[0])])


def descriptive(scores: np.ndarray) -> dict:
    """Summary statistics used in the article: mean, median, std, n_efficient."""
    s = np.asarray(scores, dtype=float)
    return {
        "n": int(s.size),
        "mean": float(np.nanmean(s)),
        "median": float(np.nanmedian(s)),
        "std": float(np.nanstd(s, ddof=1)),
        "min": float(np.nanmin(s)),
        "max": float(np.nanmax(s)),
        "n_efficient": int(np.sum(np.isclose(s, 1.0, atol=1e-4))),
    }
