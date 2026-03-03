from fastapi import FastAPI
import numpy as np
import pandas as pd
import yfinance as yf

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/prices")
def prices(ticker:str):

    data = yf.download(ticker)

    close = data["Close"]

    return {
        "prices": close.tolist()
    }
