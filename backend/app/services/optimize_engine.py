from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..schemas import OptimizeRequest
from .market_data import get_prices_cached, returns_from_prices

def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, 1)
    s = w.sum()
    return w / (s if s else 1)

def _cov_ann(rets: pd.DataFrame) -> np.ndarray:
    return rets.cov().values * 252.0

def _mu_ann(rets: pd.DataFrame) -> np.ndarray:
    return rets.mean().values * 252.0

def min_variance(rets: pd.DataFrame) -> np.ndarray:
    cov = _cov_ann(rets)
    n = cov.shape[0]
    def obj(w): return float(w.T @ cov @ w)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type":"eq","fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return _normalize(res.x if res.success else x0)

def max_sharpe(rets: pd.DataFrame, rf_annual: float) -> np.ndarray:
    cov = _cov_ann(rets)
    mu = _mu_ann(rets)
    n = len(mu)

    def neg_sharpe(w):
        w = _normalize(w)
        ret = float(mu @ w)
        vol = float(np.sqrt(w.T @ cov @ w) + 1e-12)
        return -((ret - rf_annual) / vol)

    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type":"eq","fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return _normalize(res.x if res.success else x0)

def risk_parity(rets: pd.DataFrame) -> np.ndarray:
    cov = _cov_ann(rets)
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def risk_contrib(w):
        w = _normalize(w)
        port_var = float(w.T @ cov @ w) + 1e-12
        mrc = cov @ w
        rc = w * mrc
        return rc / port_var

    def obj(w):
        rc = risk_contrib(w)
        target = np.ones(n) / n
        return float(((rc - target) ** 2).sum())

    bounds = [(0.0, 1.0)] * n
    cons = [{"type":"eq","fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return _normalize(res.x if res.success else x0)

def efficient_frontier(rets: pd.DataFrame, points: int = 30) -> list[dict]:
    cov = _cov_ann(rets)
    mu = _mu_ann(rets)
    n = len(mu)

    w_minv = min_variance(rets)
    r_min = float(mu @ w_minv)
    r_max = float(mu.max())
    targets = np.linspace(r_min, max(r_min + 1e-6, r_max), points)

    out = []
    for tr in targets:
        def obj(w): return float(w.T @ cov @ w)
        cons = [
            {"type":"eq","fun": lambda w: np.sum(w) - 1.0},
            {"type":"eq","fun": lambda w, tr=tr: float(mu @ _normalize(w)) - tr},
        ]
        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        w = _normalize(res.x if res.success else x0)
        vol = float(np.sqrt(w.T @ cov @ w))
        ret = float(mu @ w)
        out.append({"target_return": float(tr), "ret": ret, "vol": vol, "weights": w.tolist()})

    return out

def optimize_all(req: OptimizeRequest) -> dict:
    tickers = [t.upper() for t in req.tickers]
    px = get_prices_cached(tickers, req.start, req.end)
    rets = returns_from_prices(px)

    w_minv = min_variance(rets)
    w_msr = max_sharpe(rets, req.rf_annual)
    w_rp = risk_parity(rets)
    frontier = efficient_frontier(rets, req.frontier_points)

    return {
        "tickers": tickers,
        "min_variance": w_minv.tolist(),
        "max_sharpe": w_msr.tolist(),
        "risk_parity": w_rp.tolist(),
        "frontier": frontier,
    }
