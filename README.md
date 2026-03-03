# Quant Risk Lab Pro (Full-Stack)

A production-style quantitative risk & portfolio analytics platform.

## Features
- Market data fetch + PostgreSQL cache
- Risk engine:
  - Historical VaR/CVaR
  - Parametric (Normal) VaR/CVaR
  - Correlated Monte Carlo VaR/CVaR (Cholesky)
  - EVT (Peaks-over-Threshold) VaR/CVaR
  - EWMA covariance option
- Portfolio optimization:
  - Min Variance (long-only)
  - Max Sharpe (long-only)
  - Risk Parity
  - Efficient Frontier
- Backtesting:
  - Monthly / Quarterly rebalancing
  - Transaction costs (bps) + turnover
  - Metrics: CAGR, Vol, Sharpe, Sortino, Max Drawdown, Calmar
- Reports:
  - Export Markdown + JSON as ZIP

## Tech Stack
Backend: FastAPI + SQLModel + PostgreSQL  
Frontend: React + Vite + TypeScript + Recharts  
Infra: Docker Compose  
Quality: Ruff + Pytest + GitHub Actions CI

## Quick Start
```bash
docker compose up --build
