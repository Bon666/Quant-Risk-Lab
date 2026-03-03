from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

def export_markdown_zip(kind: str, payload: dict) -> bytes:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    md = io.StringIO()

    md.write("# Quant Risk Lab Pro Report\n\n")
    md.write(f"- Kind: **{kind}**\n")
    md.write(f"- Generated: **{now}**\n\n")

    if kind == "risk":
        md.write("## Risk Summary\n\n")
        for k in ["tickers","weights","method","confidence","vol_annual","max_drawdown","var_1d","cvar_1d","sharpe_approx"]:
            if k in payload:
                md.write(f"- **{k}**: `{payload[k]}`\n")
        md.write("\n")

    if kind == "backtest":
        md.write("## Backtest Stats\n\n")
        stats = payload.get("stats", {})
        for k, v in stats.items():
            md.write(f"- **{k}**: `{v}`\n")
        md.write("\n")

    if kind == "optimize":
        md.write("## Optimization\n\n")
        for k in ["min_variance","max_sharpe","risk_parity"]:
            if k in payload:
                md.write(f"- **{k}**: `{payload[k]}`\n")
        md.write("\n")

    md.write("## Raw JSON\n\n```json\n")
    md.write(json.dumps(payload, indent=2))
    md.write("\n```\n")

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{kind}_report.md", md.getvalue())
        z.writestr(f"{kind}_payload.json", json.dumps(payload, indent=2))
    return zbuf.getvalue()
