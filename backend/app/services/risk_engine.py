from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, genpareto

from ..schemas import RiskRequest
from .market_data import get_prices_cached, returns_from_prices
from .metrics import max_drawdown, ann_vol, sharpe

def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if abs(s) < 1e-12:
        return np.ones_like(w) / len(w)
    return w / s

def _ewma_cov(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    T, N = returns.shape
    cov = np.zeros((N, N))
    for t in range(T):
        x = returns[t:t+1].T
        cov = lam * cov + (1 - lam) * (x @ x.T)
    return cov

def _port_rets(returns_df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return (returns_df.values @ w).astype(float)

def _var_cvar_historical(port: np.ndarray, alpha: float) -> tuple[float, float]:
    q = np.quantile(port, 1 - alpha)
    var = -q
    cvar = -port[port <= q].mean()
    return float(var), float(cvar)

def _var_cvar_parametric(port: np.ndarray, alpha: float) -> tuple[float, float]:
    mu = port.mean()
    sd = port.std(ddof=1)
    z = norm.ppf(1 - alpha)
    q = mu + z * sd
    var = -q
    cvar = -(mu - sd * norm.pdf(z) / (1 - alpha))
    return float(var), float(cvar)

def _var_cvar_mc_correlated(
    returns_df: pd.DataFrame,
    w: np.ndarray,
    alpha: float,
    sims: int,
    use_ewma: bool,
    lam: float,
) -> tuple[float, float]:
    R = returns_df.values
    mu = R.mean(axis=0)
    cov = _ewma_cov(R, lam) if use_ewma else np.cov(R, rowvar=False, ddof=1)
    cov = (cov + cov.T) / 2
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    L = np.linalg.cholesky(cov)
    z = np.random.normal(size=(sims, cov.shape[0]))
    draws = z @ L.T + mu
    port = draws @ w
    return _var_cvar_historical(port, alpha)

def _var_cvar_evt_pot(port: np.ndarray, alpha: float, threshold_q: float) -> tuple[float, float]:
    losses = -port
    u = np.quantile(losses, threshold_q)
    exceed = losses[losses > u] - u
    if len(exceed) < 20:
        return _var_cvar_historical(port, alpha)

    c, loc, scale = genpareto.fit(exceed, floc=0.0)
    p_u = (losses > u).mean()
    tail_p = 1 - alpha
    if tail_p <= 0 or p_u <= 0:
        return _var_cvar_historical(port, alpha)

    if abs(c) < 1e-8:
        var = u + scale * np.log(p_u / tail_p)
    else:
        var = u + (scale / c) * ((tail_p / p_u) ** (-c) - 1)

    sims = 50000
    y = genpareto.rvs(c, loc=0.0, scale=scale, size=sims)
    sim_losses = u + y
    es = sim_losses[sim_losses >= var].mean()

    return float(var), float(es)

def risk_summary(req: RiskRequest) -> dict:
    tickers = [t.upper() for t in req.tickers]
    px = get_prices_cached(tickers, req.start, req.end)
    rets = returns_from_prices(px)

    w = _normalize_weights(np.array(req.weights, dtype=float))
    port = _port_rets(rets, w)
    nav = np.cumprod(1 + port)

    alpha = req.method.confidence
    method = req.method.method

    if method == "historical":
        var, cvar = _var_cvar_historical(port, alpha)
    elif method == "parametric":
        var, cvar = _var_cvar_parametric(port, alpha)
    elif method == "evt_pot":
        var, cvar = _var_cvar_evt_pot(port, alpha, req.method.evt_threshold_q)
    else:
        var, cvar = _var_cvar_mc_correlated(
            rets, w, alpha, req.method.mc_sims,
            use_ewma=req.method.use_ewma,
            lam=req.method.ewma_lambda,
        )

    return {
        "tickers": tickers,
        "weights": w.tolist(),
        "start": req.start,
        "end": req.end,
        "method": method,
        "confidence": alpha,
        "mean_daily": float(port.mean()),
        "vol_annual": ann_vol(port),
        "sharpe_approx": sharpe(port, rf_annual=0.02),
        "max_drawdown": max_drawdown(nav),
        "var_1d": var,
        "cvar_1d": cvar,
        "nav": nav.tolist(),
        "port_returns": port.tolist(),
    }
