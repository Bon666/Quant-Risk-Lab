from __future__ import annotations
import numpy as np

def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = (nav / peak) - 1.0
    return float(dd.min())

def cagr(nav: np.ndarray, periods_per_year: int = 252) -> float:
    if len(nav) < 2:
        return 0.0
    years = (len(nav) - 1) / periods_per_year
    return float(nav[-1] ** (1 / max(years, 1e-12)) - 1)

def ann_vol(rets: np.ndarray, periods_per_year: int = 252) -> float:
    return float(np.std(rets, ddof=1) * np.sqrt(periods_per_year))

def sharpe(rets: np.ndarray, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = rets - rf_daily
    vol = np.std(ex, ddof=1) + 1e-12
    return float(np.mean(ex) / vol * np.sqrt(periods_per_year))

def sortino(rets: np.ndarray, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = rets - rf_daily
    downside = ex[ex < 0]
    dd = np.std(downside, ddof=1) + 1e-12
    return float(np.mean(ex) / dd * np.sqrt(periods_per_year))

def calmar(nav: np.ndarray, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(nav)) + 1e-12
    return cagr(nav, periods_per_year) / mdd
