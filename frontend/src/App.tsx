import React, { useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from "recharts";
import { postJSON } from "./api";

type RiskResp = {
  tickers: string[];
  weights: number[];
  method: string;
  confidence: number;
  vol_annual: number;
  sharpe_approx: number;
  max_drawdown: number;
  var_1d: number;
  cvar_1d: number;
  nav: number[];
};

type OptResp = {
  tickers: string[];
  min_variance: number[];
  max_sharpe: number[];
  risk_parity: number[];
  frontier: { ret: number; vol: number; weights: number[] }[];
};

type BtResp = {
  dates: string[];
  nav: number[];
  drawdown: number[];
  stats: Record<string, number>;
};

function parseTickers(s: string) {
  return s.split(",").map(x => x.trim().toUpperCase()).filter(Boolean);
}
function parseWeights(s: string, n: number) {
  const w = s.split(",").map(x => Number(x.trim())).filter(x => !Number.isNaN(x));
  if (w.length !== n) return Array(n).fill(1 / n);
  const sum = w.reduce((a,b)=>a+b,0) || 1;
  return w.map(x => x / sum);
}

export default function App() {
  const [tickersText, setTickersText] = useState("SPY,QQQ,TLT");
  const [weightsText, setWeightsText] = useState("0.5,0.3,0.2");
  const [start, setStart] = useState("2022-01-01");
  const [end, setEnd] = useState("2026-01-01");

  const [riskMethod, setRiskMethod] = useState("monte_carlo");
  const [confidence, setConfidence] = useState(0.95);
  const [mcSims, setMcSims] = useState(20000);
  const [useEwma, setUseEwma] = useState(true);

  const [risk, setRisk] = useState<RiskResp | null>(null);
  const [opt, setOpt] = useState<OptResp | null>(null);
  const [bt, setBt] = useState<BtResp | null>(null);
  const [strategy, setStrategy] = useState("max_sharpe");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const tickers = useMemo(() => parseTickers(tickersText), [tickersText]);
  const weights = useMemo(() => parseWeights(weightsText, tickers.length), [weightsText, tickers.length]);

  const navChart = useMemo(() => (risk ? risk.nav.map((v,i)=>({i, nav:v})) : []), [risk]);
  const btChart = useMemo(() => (bt ? bt.nav.map((v,i)=>({date: bt.dates[i], nav:v, dd: bt.drawdown[i]})) : []), [bt]);
  const frontierChart = useMemo(() => (opt ? opt.frontier.map(p=>({vol:p.vol, ret:p.ret})) : []), [opt]);

  async function runRisk() {
    setLoading(true); setError("");
    try {
      const body = {
        tickers, weights, start, end,
        method: { method: riskMethod, confidence, mc_sims: mcSims, use_ewma: useEwma }
      };
      const resp = await postJSON<RiskResp>("/risk", body);
      setRisk(resp);
    } catch (e:any) {
      setError(e.message || "Failed");
    } finally {
      setLoading(false);
    }
  }

  async function runOptimize() {
    setLoading(true); setError("");
    try {
      const resp = await postJSON<OptResp>("/portfolio/optimize", {
        tickers, start, end, rf_annual: 0.02, frontier_points: 30
      });
      setOpt(resp);
    } catch (e:any) {
      setError(e.message || "Failed");
    } finally {
      setLoading(false);
    }
  }

  async function runBacktest() {
    setLoading(true); setError("");
    try {
      const resp = await postJSON<BtResp>("/backtest", {
        tickers, start, end, rebalance: "monthly", tc_bps: 5, strategy, rf_annual: 0.02
      });
      setBt(resp);
    } catch (e:any) {
      setError(e.message || "Failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Quant Risk Lab Pro</h1>

      <div className="card">
        <div className="row">
          <div>
            <label>Tickers</label>
            <input value={tickersText} onChange={e=>setTickersText(e.target.value)} />
          </div>
          <div>
            <label>Weights</label>
            <input value={weightsText} onChange={e=>setWeightsText(e.target.value)} />
            <div className="small">Mismatch count → fallback to equal weight.</div>
          </div>
          <div><label>Start</label><input value={start} onChange={e=>setStart(e.target.value)} /></div>
          <div><label>End</label><input value={end} onChange={e=>setEnd(e.target.value)} /></div>

          <div>
            <label>Risk Method</label>
            <select value={riskMethod} onChange={e=>setRiskMethod(e.target.value)}>
              <option value="historical">Historical</option>
              <option value="parametric">Parametric (Normal)</option>
              <option value="monte_carlo">Correlated Monte Carlo</option>
              <option value="evt_pot">EVT POT</option>
            </select>
          </div>
          <div>
            <label>Confidence</label>
            <input value={confidence} onChange={e=>setConfidence(Number(e.target.value))} />
          </div>

          <div>
            <label>MC Sims</label>
            <input value={mcSims} onChange={e=>setMcSims(Number(e.target.value))} />
          </div>
          <div>
            <label>EWMA Cov (lambda=0.94)</label>
            <select value={useEwma ? "yes" : "no"} onChange={e=>setUseEwma(e.target.value==="yes")}>
              <option value="yes">On</option>
              <option value="no">Off</option>
            </select>
          </div>

          <div>
            <label>Backtest Strategy</label>
            <select value={strategy} onChange={e=>setStrategy(e.target.value)}>
              <option value="equal_weight">Equal Weight</option>
              <option value="min_variance">Min Variance</option>
              <option value="max_sharpe">Max Sharpe</option>
              <option value="risk_parity">Risk Parity</option>
            </select>
          </div>
          <div />
        </div>

        <div style={{display:"flex", gap:10, marginTop:12, flexWrap:"wrap"}}>
          <button disabled={loading} onClick={runRisk}>Run Risk</button>
          <button disabled={loading} onClick={runOptimize}>Optimize + Frontier</button>
          <button disabled={loading} onClick={runBacktest}>Run Backtest</button>
        </div>

        {error && <div style={{marginTop:10, color:"#ff9aa2"}}>Error: {error}</div>}
      </div>

      {risk && (
        <div className="card">
          <h2>Risk Summary</h2>
          <div className="kpi">
            <div className="box"><div className="small">Annual Vol</div><div>{risk.vol_annual.toFixed(4)}</div></div>
            <div className="box"><div className="small">Max Drawdown</div><div>{risk.max_drawdown.toFixed(4)}</div></div>
            <div className="box"><div className="small">Sharpe (approx)</div><div>{risk.sharpe_approx.toFixed(3)}</div></div>
            <div className="box"><div className="small">VaR (1D)</div><div>{risk.var_1d.toFixed(4)}</div></div>
            <div className="box"><div className="small">CVaR (1D)</div><div>{risk.cvar_1d.toFixed(4)}</div></div>
            <div className="box"><div className="small">Method</div><div>{risk.method} @ {risk.confidence}</div></div>
          </div>

          <h3 style={{marginTop:16}}>Portfolio NAV</h3>
          <div style={{width:"100%", height:300}}>
            <ResponsiveContainer>
              <LineChart data={navChart}>
                <XAxis dataKey="i" hide />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="nav" stroke="#4c7dff" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {opt && (
        <div className="card">
          <h2>Efficient Frontier</h2>
          <div className="small">vol vs return</div>
          <div style={{width:"100%", height:320}}>
            <ResponsiveContainer>
              <ScatterChart>
                <XAxis dataKey="vol" name="Vol" />
                <YAxis dataKey="ret" name="Return" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter data={frontierChart} fill="#7CFFCB" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="small">
            MinVar: {JSON.stringify(opt.min_variance)} | MaxSharpe: {JSON.stringify(opt.max_sharpe)} | RiskParity: {JSON.stringify(opt.risk_parity)}
          </div>
        </div>
      )}

      {bt && (
        <div className="card">
          <h2>Backtest</h2>
          <div className="kpi">
            {Object.entries(bt.stats).map(([k,v]) => (
              <div className="box" key={k}><div className="small">{k}</div><div>{Number(v).toFixed(4)}</div></div>
            ))}
          </div>

          <h3 style={{marginTop:16}}>NAV</h3>
          <div style={{width:"100%", height:320}}>
            <ResponsiveContainer>
              <LineChart data={btChart}>
                <XAxis dataKey="date" hide />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="nav" stroke="#7CFFCB" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h3 style={{marginTop:16}}>Drawdown</h3>
          <div style={{width:"100%", height:240}}>
            <ResponsiveContainer>
              <LineChart data={btChart}>
                <XAxis dataKey="date" hide />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="dd" stroke="#ff6b6b" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="card">
        <div className="small">
          Tip: Monte Carlo uses Cholesky covariance (correlated). EVT POT needs enough tail samples; if too few it falls back to Historical.
        </div>
      </div>
    </div>
  );
}
