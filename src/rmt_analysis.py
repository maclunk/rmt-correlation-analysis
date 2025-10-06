"""
Random Matrix Theory analysis utilities following Plerou et al. (2001).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression

# ---------- Returns & correlations ----------

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    if method == "log":
        rets = np.log(prices).diff()
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")

def standardize_returns(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # z-score each column (zero mean, unit variance)
    mu = returns.mean()
    sig = returns.std(ddof=0).replace(0.0, np.nan)
    Z = (returns - mu) / sig
    Z = Z.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any").dropna()
    # return per-asset std (useful for cov reconstruction later)
    std = returns.loc[Z.index, Z.columns].std(ddof=0)
    return Z, std

def corr_matrix(Z: pd.DataFrame) -> pd.DataFrame:
    # assumes Z standardized (per column var ~1)
    return Z.corr()

# ---------- Marchenko–Pastur ----------

def mp_bounds(Q: float, sigma2: float = 1.0) -> Tuple[float, float]:
    """
    MP support for eigenvalues of correlation when T observations, N assets; Q=T/N.
    λ± = σ^2 (1 ± sqrt(1/Q))^2
    """
    if Q <= 1:
        # when T < N, correlation is singular; MP support still defined but lower edge 0
        lower = sigma2 * (1 - np.sqrt(1 / Q)) ** 2 if Q > 0 else 0.0
        return (max(0.0, lower), sigma2 * (1 + np.sqrt(1 / Q)) ** 2)
    return (sigma2 * (1 - np.sqrt(1 / Q)) ** 2, sigma2 * (1 + np.sqrt(1 / Q)) ** 2)

def eigendecompose(C: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # returns eigenvalues (descending) and eigenvectors as columns
    w, V = np.linalg.eigh(C.values)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    return w, V

# ---------- Inverse Participation Ratio (IPR) ----------

def ipr(V: np.ndarray) -> np.ndarray:
    return np.sum(V ** 4, axis=0)

# ---------- Market mode regression ----------

def market_mode_regression(Z: pd.DataFrame, k: int = 1) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Regress each asset's standardized return on top-k PCs of correlation (market + sectors).
    Returns residuals (same index/columns), eigenvalues, eigenvectors used.
    """
    C = corr_matrix(Z)
    w, V = eigendecompose(C)
    PCs = Z.values @ V[:, :k]  # T x k scores
    X = PCs  # explanatory
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    X = np.nan_to_num(X)

    resid = np.zeros_like(Z.values)
    for j, col in enumerate(Z.columns):
        y = Z[col].values
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, y)
        y_hat = lr.predict(X)
        resid[:, j] = y - y_hat
    R = pd.DataFrame(resid, index=Z.index, columns=Z.columns)
    return R, w, V

# ---------- RMT Cleaning (Eigenvalue Clipping) ----------

def rmt_clean_correlation(C: pd.DataFrame, T: int, method: str = "clip", keep_top: int = 1) -> pd.DataFrame:
    """
    Eigenvalue clipping:
    - Compute MP upper bound λ+ from Q=T/N
    - Replace all eigenvalues inside the bulk (<= λ+) except the largest 'keep_top' with their average within the bulk
    """
    N = C.shape[0]
    Q = T / N
    lam_minus, lam_plus = mp_bounds(Q, sigma2=1.0)

    w, V = eigendecompose(C)
    # Identify bulk indices excluding the top 'keep_top' eigenvalues
    bulk_mask = np.ones_like(w, dtype=bool)
    bulk_mask[:keep_top] = False
    # bulk defined by eigenvalues <= λ+
    bulk_mask = bulk_mask & (w <= lam_plus + 1e-12)

    if method == "clip":
        if bulk_mask.any():
            bulk_mean = w[bulk_mask].mean()
            w_clean = w.copy()
            w_clean[bulk_mask] = bulk_mean
        else:
            w_clean = w
    else:
        w_clean = w

    C_clean = (V @ np.diag(w_clean) @ V.T)
    # enforce diag to 1, symmetrize
    C_clean = 0.5 * (C_clean + C_clean.T)
    np.fill_diagonal(C_clean, 1.0)
    return pd.DataFrame(C_clean, index=C.index, columns=C.columns)

# ---------- Overlap matrices ----------

def overlap_matrix(V1: np.ndarray, V2: np.ndarray, k: int) -> np.ndarray:
    """
    |U1^T U2|^2 for the top-k eigenvectors (N x k each).
    """
    U1 = V1[:, :k]
    U2 = V2[:, :k]
    M = np.abs(U1.T @ U2) ** 2
    return M

# ---------- Helpers to move between corr / cov ----------

def cov_from_corr(C: pd.DataFrame, std: pd.Series) -> pd.DataFrame:
    s = std.reindex(C.index).values
    S = np.outer(s, s)
    return pd.DataFrame(C.values * S, index=C.index, columns=C.columns)

def correlation_pipeline(returns: pd.DataFrame, T_est: Optional[int] = None, use_market_reg=False, keep_top:int=1) -> Dict[str, object]:
    """
    Convenience wrapper:
      - Standardize returns
      - Correlation
      - Optional market-mode regression and residual correlation
      - RMT clean correlation using T_est (or len(Z))
    """
    Z, std = standardize_returns(returns)
    C_raw = corr_matrix(Z)
    if use_market_reg:
        R, w, V = market_mode_regression(Z, k=1)
        C_resid = corr_matrix(R)
    else:
        R, w, V = Z, *eigendecompose(C_raw)
        C_resid = None
    T_eff = T_est if T_est is not None else len(Z)
    C_clean = rmt_clean_correlation(C_raw, T=T_eff, keep_top=keep_top)
    return dict(Z=Z, std=std, C_raw=C_raw, C_clean=C_clean, C_resid=C_resid, evals=w, evecs=V)
