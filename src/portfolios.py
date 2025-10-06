from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Optional
from .rmt_analysis import cov_from_corr

def _ensure_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w[w < eps] = eps
    return (V @ np.diag(w) @ V.T)

# ---------- GMV ----------

def gmv_weights(cov: pd.DataFrame, long_only: bool = True, leverage_cap: float = 1.0) -> pd.Series:
    n = cov.shape[0]
    S = _ensure_psd(cov.values)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, S))
    constraints = [cp.sum(w) == 1.0]
    if long_only:
        constraints.append(w >= 0.0)
    if leverage_cap is not None:
        constraints.append(cp.norm1(w) <= leverage_cap)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    sol = w.value
    if sol is None:
        # fallback closed-form unconstrained GMV
        invS = np.linalg.pinv(S)
        ones = np.ones(n)
        sol = invS @ ones / (ones @ invS @ ones)
        if long_only:
            # simple projection to simplex
            sol = _project_to_simplex(sol)
    return pd.Series(sol, index=cov.index)

def _project_to_simplex(y: np.ndarray) -> np.ndarray:
    # Euclidean projection onto {w >= 0, sum w = 1}
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(y - theta, 0)
    return w

# ---------- Risk Parity (equal risk contribution) ----------

def risk_parity_weights(cov: pd.DataFrame, long_only: bool = True, max_iter: int = 10_000, tol: float = 1e-8) -> pd.Series:
    n = cov.shape[0]
    S = _ensure_psd(cov.values)
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma = np.sqrt(np.maximum(w @ S @ w, 1e-12))
        mrc = (S @ w) / max(sigma, 1e-12)  # marginal risk contributions
        rc = w * mrc
        target = sigma / n
        grad = rc - target
        step = 0.1
        w_new = w - step * grad
        if long_only:
            w_new = np.maximum(w_new, 0.0)
        w_new = w_new / w_new.sum() if w_new.sum() > 0 else np.ones(n)/n
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cov.index)

# ---------- Interface: use raw vs cleaned correlations ----------

def build_covariance(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    return cov_from_corr(corr, vols)

def portfolio_from_corr(corr: pd.DataFrame, vols: pd.Series, method: str = "gmv",
                        long_only: bool = True, leverage_cap: float = 1.0) -> pd.Series:
    cov = build_covariance(corr, vols)
    if method.lower() == "gmv":
        return gmv_weights(cov, long_only=long_only, leverage_cap=leverage_cap)
    elif method.lower() in ("riskparity", "risk_parity", "erc"):
        return risk_parity_weights(cov, long_only=long_only)
    else:
        raise ValueError(f"Unknown method {method}")
