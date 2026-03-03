from __future__ import annotations

from datetime import date
from typing import Optional
from sqlmodel import SQLModel, Field

class PriceBar(SQLModel, table=True):
    __tablename__ = "price_bars"
    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True)
    d: date = Field(index=True)
    adj_close: float
