import numpy as np
import pandas as pd

def annualize_return(daily_ret: pd.Series, trading_days: int = 252) -> float:
    mu = (1 + daily_ret).prod() ** (trading_days / len(daily_ret)) - 1
    return float(mu)

def annualize_vol(daily_ret: pd.Series, trading_days: int = 252) -> float:
    return float(daily_ret.std(ddof=0) * np.sqrt(trading_days))

def sharpe(daily_ret: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    er = daily_ret.mean() * trading_days - rf
    vol = daily_ret.std(ddof=0) * np.sqrt(trading_days)
    return float(er / vol) if vol > 0 else np.nan

def max_drawdown(cumret: pd.Series) -> float:
    roll_max = cumret.cummax()
    dd = cumret / roll_max - 1.0
    return float(dd.min())

def turnover(weights_df: pd.DataFrame) -> pd.Series:
    return weights_df.diff().abs().sum(axis=1).fillna(0.0)

def realized_portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    aligned_w = weights.reindex(returns.index).ffill().fillna(0.0)
    aligned_w = aligned_w.div(aligned_w.abs().sum(axis=1), axis=0).fillna(0.0)  # normalize to 1 leverage
    pr = (aligned_w * returns).sum(axis=1)
    return pr

def apply_tc(daily_ret: pd.Series, to: pd.Series, tc_bps: float = 0.0) -> pd.Series:
    # transaction cost at rebal dates; multiply turnover by tc in decimals
    tc = to * (tc_bps / 1e4)
    net = daily_ret.copy()
    net.loc[tc.index] = net.loc[tc.index] - tc
    return net
