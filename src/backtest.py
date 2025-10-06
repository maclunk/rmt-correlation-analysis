from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .rmt_analysis import standardize_returns, corr_matrix, rmt_clean_correlation, cov_from_corr
from .portfolios import portfolio_from_corr
from .utils import realized_portfolio_returns, turnover, apply_tc

def rolling_backtest(returns: pd.DataFrame,
                     window: int = 504,
                     rebalance: int = 21,
                     use_cleaned: bool = True,
                     keep_top: int = 1,
                     method: str = "gmv",
                     long_only: bool = True,
                     leverage_cap: float = 1.0,
                     tc_bps: float = 0.0) -> Dict[str, pd.Series]:

    dates = returns.index
    weights = []
    rebal_dates = []

    for t in range(window, len(dates), rebalance):
        train_slice = returns.iloc[t-window:t]
        test_slice = returns.iloc[t:min(t+rebalance, len(dates))]
        # standardize for corr; keep realized vols from train
        Z, vols = standardize_returns(train_slice)
        if Z.shape[1] < 2:
            continue
        C_raw = corr_matrix(Z)
        if use_cleaned:
            C_use = rmt_clean_correlation(C_raw, T=len(Z), keep_top=keep_top)
        else:
            C_use = C_raw
        w = portfolio_from_corr(C_use, vols.reindex(C_use.index), method=method,
                                long_only=long_only, leverage_cap=leverage_cap)
        # align to universe in test period
        w = w.reindex(test_slice.columns).fillna(0.0)
        W_df = pd.DataFrame([w.values], index=[dates[t]], columns=test_slice.columns)
        weights.append(W_df)
        rebal_dates.append(dates[t])

    if not weights:
        raise ValueError("Backtest produced no weights. Check window/rebalance/data.")

    weights_df = pd.concat(weights).sort_index()
    port_ret = realized_portfolio_returns(weights_df, returns)
    to = turnover(weights_df)
    net_ret = apply_tc(port_ret, to, tc_bps=tc_bps)
    eq_curve = (1 + net_ret).cumprod()
    return dict(gross=port_ret, net=net_ret, equity_curve=eq_curve, turnover=to, weights=weights_df)

def compare_strategies(returns: pd.DataFrame, window: int = 504, rebalance: int = 21,
                       methods=("gmv","riskparity"), tc_bps: float = 1.0) -> Dict[str, Dict[str, pd.Series]]:
    out = {}
    for use_cleaned in (False, True):
        tag = "cleaned" if use_cleaned else "raw"
        for m in methods:
            key = f"{m}_{tag}"
            out[key] = rolling_backtest(returns, window=window, rebalance=rebalance,
                                        use_cleaned=use_cleaned, method=m, tc_bps=tc_bps)
    return out
