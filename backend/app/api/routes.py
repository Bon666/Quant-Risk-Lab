from __future__ import annotations

from fastapi import APIRouter, Response
from ..schemas import PricesRequest, RiskRequest, OptimizeRequest, BacktestRequest, ExportRequest
from ..services.market_data import get_prices_cached
from ..services.risk_engine import risk_summary
from ..services.optimize_engine import optimize_all
from ..services.backtest_engine import run_backtest
from ..services.exporter import export_markdown_zip

router = APIRouter()

@router.post("/prices")
def prices(req: PricesRequest):
    px = get_prices_cached(req.tickers, req.start, req.end)
    return {
        "tickers": list(px.columns),
        "dates": [d.strftime("%Y-%m-%d") for d in px.index],
        "prices": {c: px[c].tolist() for c in px.columns},
    }

@router.post("/risk")
def risk(req: RiskRequest):
    return risk_summary(req)

@router.post("/portfolio/optimize")
def optimize(req: OptimizeRequest):
    return optimize_all(req)

@router.post("/backtest")
def backtest(req: BacktestRequest):
    return run_backtest(req)

@router.post("/export/mdzip")
def export_mdzip(req: ExportRequest):
    zbytes = export_markdown_zip(req.kind, req.payload)
    return Response(content=zbytes, media_type="application/zip")
