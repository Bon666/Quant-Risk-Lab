from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import BacktestRequest
from .market_data import get_prices_cached, returns_from_prices
from .optimize_engine import min_variance, max_sharpe, risk_parity
from .metrics import max_drawdown, cagr, ann_vol, sharpe, sortino, calmar

def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    period = index.to_period("M" if freq == "monthly" else "Q")
    first = {}
    for dt, p in zip(index, period):
        if p not in first:
            first[p] = dt
    return set(first.values())

def _weights_for_strategy(rets_window: pd.DataFrame, strategy: str, rf_annual: float) -> np.ndarray:
    n = rets_window.shape[1]
    if strategy == "min_variance":
        return min_variance(rets_window)
    if strategy == "max_sharpe":
        return max_sharpe(rets_window, rf_annual)
    if strategy == "risk_parity":
        return risk_parity(rets_window)
    return np.ones(n) / n

def run_backtest(req: BacktestRequest) -> dict:
    tickers = [t.upper() for t in req.tickers]
    px = get_prices_cached(tickers, req.start, req.end)
    rets = returns_from_prices(px)
    dates = rets.index

    rb = _rebalance_dates(dates, req.rebalance)
    tc_rate = req.tc_bps / 10000.0

    n = rets.shape[1]
    w = np.ones(n) / n
    nav = 1.0

    nav_list = []
    turnover_list = []
    w_list = []
    port_rets = []

    for i, dt in enumerate(dates):
        if dt in rb:
            w_old = w.copy()
            window = rets.iloc[max(0, i-126):i]  # ~6 months
            if len(window) < 20:
                window = rets.iloc[:i+1]
            w = _weights_for_strategy(window, req.strategy, req.rf_annual)

            turnover = float(np.abs(w - w_old).sum())
            cost = turnover * tc_rate
        else:
            turnover = 0.0
            cost = 0.0

        r = float(rets.iloc[i].values @ w)
        nav = nav * (1 + r - cost)

        port_rets.append(r - cost)
        nav_list.append(nav)
        w_list.append(w.tolist())
        turnover_list.append(turnover)

    nav_arr = np.array(nav_list, dtype=float)
    peak = np.maximum.accumulate(nav_arr)
    dd_arr = (nav_arr / peak) - 1.0
    pret = np.array(port_rets, dtype=float)

    stats = {
        "cagr": cagr(nav_arr),
        "vol_annual": ann_vol(pret),
        "sharpe": sharpe(pret, rf_annual=req.rf_annual),
        "sortino": sortino(pret, rf_annual=req.rf_annual),
        "max_drawdown": max_drawdown(nav_arr),
        "calmar": calmar(nav_arr),
        "avg_turnover": float(np.mean(turnover_list)),
    }

    return {
        "tickers": tickers,
        "start": req.start,
        "end": req.end,
        "rebalance": req.rebalance,
        "tc_bps": req.tc_bps,
        "strategy": req.strategy,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "nav": nav_arr.tolist(),
        "drawdown": dd_arr.tolist(),
        "weights": w_list,
        "turnover": turnover_list,
        "stats": stats,
    }
