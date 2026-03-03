from __future__ import annotations

from datetime import datetime, date
import pandas as pd
import yfinance as yf
from sqlmodel import Session, select

from ..db import engine
from ..models import PriceBar

def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _fetch_yf_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if len(tickers) == 1:
        t = tickers[0]
        adj = df["Adj Close"].to_frame(name=t)
    else:
        adj = pd.DataFrame({t: df[t]["Adj Close"] for t in tickers})

    adj = adj.dropna(how="all").ffill().dropna()
    return adj

def _load_cached(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    with Session(engine) as s:
        data = {}
        for t in tickers:
            rows = s.exec(
                select(PriceBar)
                .where((PriceBar.ticker == t) & (PriceBar.d >= start) & (PriceBar.d < end))
                .order_by(PriceBar.d)
            ).all()
            if rows:
                data[t] = pd.Series(
                    [r.adj_close for r in rows],
                    index=pd.to_datetime([r.d for r in rows]),
                    name=t,
                )
        if not data:
            return pd.DataFrame()
        px = pd.concat(data.values(), axis=1)
        px = px.dropna(how="all").ffill().dropna()
        return px

def _upsert_prices(px: pd.DataFrame) -> None:
    with Session(engine) as s:
        for t in px.columns:
            for dt, v in px[t].items():
                d = dt.date()
                existing = s.exec(
                    select(PriceBar).where((PriceBar.ticker == t) & (PriceBar.d == d))
                ).first()
                if existing:
                    existing.adj_close = float(v)
                else:
                    s.add(PriceBar(ticker=t, d=d, adj_close=float(v)))
        s.commit()

def get_prices_cached(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    d0, d1 = _to_date(start), _to_date(end)

    cached = _load_cached(tickers, d0, d1)
    need_fetch = (cached.empty) or (set(cached.columns) != set(tickers)) or (len(cached) < 30)

    if need_fetch:
        fresh = _fetch_yf_adj_close(tickers, start, end)
        _upsert_prices(fresh)
        cached = _load_cached(tickers, d0, d1)

    return cached

def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()
