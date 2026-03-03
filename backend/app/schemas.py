from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Literal

class PricesRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")

class RiskMethod(BaseModel):
    method: Literal["historical", "parametric", "monte_carlo", "evt_pot"] = "historical"
    confidence: float = Field(0.95, ge=0.80, le=0.999)
    mc_sims: int = Field(20000, ge=1000, le=300000)
    use_ewma: bool = True
    ewma_lambda: float = Field(0.94, ge=0.80, le=0.999)
    evt_threshold_q: float = Field(0.90, ge=0.80, le=0.99)

class RiskRequest(BaseModel):
    tickers: List[str]
    weights: List[float]
    start: str
    end: str
    method: RiskMethod = RiskMethod()

class OptimizeRequest(BaseModel):
    tickers: List[str]
    start: str
    end: str
    rf_annual: float = Field(0.02, ge=-0.05, le=0.20)
    frontier_points: int = Field(30, ge=10, le=120)

class BacktestRequest(BaseModel):
    tickers: List[str]
    start: str
    end: str
    rebalance: Literal["monthly", "quarterly"] = "monthly"
    tc_bps: float = Field(5.0, ge=0.0, le=200.0)
    strategy: Literal["equal_weight", "min_variance", "max_sharpe", "risk_parity"] = "equal_weight"
    rf_annual: float = Field(0.02, ge=-0.05, le=0.20)

class ExportRequest(BaseModel):
    kind: Literal["risk", "backtest", "optimize"]
    payload: dict
