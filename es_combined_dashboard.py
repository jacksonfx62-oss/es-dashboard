"""
ES Futures Combined Dashboard
==============================
Tabs: Liquidity Heatmap · Vol Surface 3D · Greeks (GEX / Gamma / Vanna / Charm / Term Structure)

Requirements:
    pip install yfinance plotly pandas numpy scipy dash dash-bootstrap-components

Usage:
    python3 es_combined_dashboard.py
    → http://localhost:8765
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.interpolate import griddata
from datetime import datetime, time as dtime
import time, threading, socket, base64, os, json
import http.server, socketserver

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")

# ── Config ─────────────────────────────────────────────────────────────────────
REFRESH_SECS     = 10
VOL_REFRESH_SECS = 60
TICKER           = "SPY"
MULT             = 100
MIN_OI           = 10
MAX_EXP          = 8
TOP_N_LEVELS     = 5

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = "#080b10"
PANEL = "#0b0f17"
LINE  = "#151c28"
DIM   = "#3a4560"
LABEL = "#5a6a88"
TEXT  = "#8fa3c0"
WHITE = "#dce8ff"
CYAN  = "#00c8f0"
GREEN = "#00d97e"
RED   = "#f03d5f"
AMBER = "#f0a500"
FONT  = "monospace"
SPY_TO_ES: float = 10.0

# ── Logo ───────────────────────────────────────────────────────────────────────
def _load_logo():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ["logo.jpg", "logo.png", "logo.jpeg"]:
        path = os.path.join(here, name)
        if os.path.exists(path):
            ext = name.rsplit(".", 1)[-1]
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return (f'<img src="data:image/{ext};base64,{b64}" '
                    f'style="height:28px;width:auto;margin-right:14px;'
                    f'filter:invert(1);opacity:0.85;vertical-align:middle">')
    return '<span style="color:#00ff88;font-size:13px;font-weight:bold;margin-right:14px">ES</span>'

LOGO_HTML = _load_logo()

# ── Session ────────────────────────────────────────────────────────────────────
def get_session():
    now_et = datetime.now(ET)
    t, wd  = now_et.time(), now_et.weekday()
    if wd >= 5:          return "WEEKEND",     "#ff6b35"
    if t < dtime(4,0):   return "OVERNIGHT",   "#7b68ee"
    if t < dtime(9,30):  return "PRE-MARKET",  "#f5c518"
    if t <= dtime(16,0): return "MARKET OPEN", "#00ff88"
    if t <= dtime(20,0): return "AFTER-HOURS", "#ff8c00"
    return "OVERNIGHT", "#7b68ee"

# ── Price helpers ──────────────────────────────────────────────────────────────
def fetch_price(symbol):
    tk = yf.Ticker(symbol)
    try:
        fi = tk.fast_info
        for a in ["last_price", "regularMarketPrice"]:
            v = getattr(fi, a, None)
            if v and str(v) != "nan" and float(v) > 100:
                return float(v)
    except: pass
    try:
        h = tk.history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except: pass
    return None

def get_es_ratio():
    global SPY_TO_ES
    try:
        spy_price = float(yf.Ticker("SPY").fast_info["lastPrice"])
        for ticker in ("ES=F", "MES=F"):
            try:
                es_price = float(yf.Ticker(ticker).fast_info["lastPrice"])
                if es_price and es_price > 1000 and spy_price > 0:
                    SPY_TO_ES = round(es_price / spy_price, 4)
                    return SPY_TO_ES
            except: continue
        spx_price = float(yf.Ticker("^GSPC").fast_info["lastPrice"])
        if spy_price > 0:
            SPY_TO_ES = round(spx_price / spy_price, 4)
            return SPY_TO_ES
    except: pass
    return 10.12

def spy_to_es(p): return round(p * SPY_TO_ES, 2)

# ── Black-Scholes Greeks ───────────────────────────────────────────────────────
def greeks(S, K, T_, r, q, sigma, flag):
    T_ = np.clip(T_, 1e-6, None); sigma = np.clip(sigma, 0.001, 5.0)
    is_c = (flag == "C")
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T_) / (sigma*np.sqrt(T_))
        d2 = d1 - sigma*np.sqrt(T_)
    n1, N1 = norm.pdf(d1), norm.cdf(d1)
    delta = np.where(is_c, np.exp(-q*T_)*N1, np.exp(-q*T_)*(N1-1))
    gamma = np.exp(-q*T_)*n1 / (S*sigma*np.sqrt(T_))
    vanna = -np.exp(-q*T_)*n1*d2 / (sigma*100.)
    cb    = np.exp(-q*T_)*(q*N1 - n1*(2*(r-q)*T_ - d2*sigma*np.sqrt(T_))
                           / (2*T_*sigma*np.sqrt(T_)))
    charm = np.where(is_c, cb, cb - q*np.exp(-q*T_)) / 365.
    return pd.DataFrame({"delta":delta,"gamma":gamma,"vanna":vanna,"charm":charm})

def rate():
    try:
        v = yf.Ticker("^IRX").fast_info["lastPrice"]/100.
        return float(v) if not np.isnan(v) else 0.045
    except: return 0.045

def div_yield():
    try:
        t = yf.Ticker(TICKER)
        d = t.dividends; s = t.fast_info["lastPrice"]
        return float(d.last("365D").sum()/s) if (not d.empty and s) else 0.013
    except: return 0.013

# ── Options chain fetch ────────────────────────────────────────────────────────
def fetch_chain():
    global MIN_OI, SPY_TO_ES
    SPY_TO_ES = get_es_ratio()
    t = yf.Ticker(TICKER)
    spot = float(t.fast_info["lastPrice"])
    r, q = rate(), div_yield()
    frames = []
    for exp in t.options[:MAX_EXP]:
        try: chain = t.option_chain(exp)
        except: continue
        dte = max((pd.Timestamp(exp)-pd.Timestamp.now()).days, 0)
        Tv  = max(dte/365., 1e-6)
        for raw, flag in [(chain.calls,"C"),(chain.puts,"P")]:
            raw = raw.copy()
            raw.columns = [c.lower().replace(" ","_") for c in raw.columns]
            cm = {}
            for c in raw.columns:
                cl = c.lower()
                if cl=="strike": cm[c]="strike"
                elif "impliedvol" in cl or "implied_vol" in cl: cm[c]="iv"
                elif "openinterest" in cl or "open_interest" in cl: cm[c]="oi"
                elif cl=="volume": cm[c]="volume"
            raw = raw.rename(columns=cm)
            if not {"strike","iv","oi","volume"}.issubset(raw.columns): continue
            df = raw[["strike","iv","oi","volume"]].copy()
            df["oi"]  = pd.to_numeric(df["oi"],  errors="coerce").fillna(0)
            df["iv"]  = pd.to_numeric(df["iv"],  errors="coerce").fillna(0)
            df["vol"] = pd.to_numeric(df["volume"],errors="coerce").fillna(0)
            df = df[df["oi"]>=MIN_OI]
            df = df[(df["strike"]>=0.80*spot)&(df["strike"]<=1.20*spot)&(df["iv"]>0.01)]
            if df.empty: continue
            df["flag"]=flag; df["expiry"]=exp; df["dte"]=dte; df["spot"]=spot
            g = greeks(spot, df["strike"].values, np.full(len(df),Tv),
                       r, q, df["iv"].values, df["flag"].values)
            for c in g.columns: df[c]=g[c].values
            sign = np.where(df["flag"]=="C",-1,1)
            df["ngex"]   = sign*df["gamma"]*df["oi"]*MULT
            df["nvanna"] = sign*df["vanna"]*df["oi"]*MULT
            df["ncharm"] = sign*df["charm"]*df["oi"]*MULT
            frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ── Greeks chart helpers ───────────────────────────────────────────────────────
def _theme(fig, title=""):
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=BG,
        font=dict(family=FONT, color=LABEL, size=9),
        title=dict(text=title, font=dict(color=DIM, size=9, family=FONT), x=0.01, y=0.98),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=DIM, size=8),
                   linecolor=LINE, tickcolor=LINE),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=DIM, size=8),
                   linecolor=LINE, tickcolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=DIM, size=8)),
        margin=dict(l=48, r=12, t=20, b=36),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE,
                        font=dict(color=TEXT, size=9, family=FONT)),
    )
    return fig

def key_levels(series, n=TOP_N_LEVELS):
    return series.abs().nlargest(n).index.tolist()

def _ticks(index):
    return index[::max(1,len(index)//8)].tolist()

def build_gex_chart(df):
    spot = df["spot"].iloc[0]
    gex  = df.groupby("strike")["ngex"].sum()
    keep = pd.concat([gex[gex>0].nlargest(TOP_N_LEVELS), gex[gex<0].nsmallest(TOP_N_LEVELS)]).sort_index()
    fig  = go.Figure()
    fig.add_trace(go.Bar(
        x=keep.index, y=keep.values,
        marker_color=[GREEN if v>0 else RED for v in keep.values],
        marker_line_width=0, width=[1.0]*len(keep), opacity=0.9,
        hovertemplate="ES %{x}<br>%{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=spot, line=dict(color=WHITE, width=0.8, dash="dot"))
    fig.add_hline(y=0,    line=dict(color=LINE,  width=1))
    if not keep.empty:
        fig.add_annotation(x=spot, y=keep.values.max()*0.95 if keep.values.max()!=0 else 1,
                           text=f"ES  {spy_to_es(spot):,.0f}", showarrow=False,
                           font=dict(color=WHITE, size=8, family=FONT))
    _theme(fig, "GEX  ·  KEY LEVELS")
    fig.update_xaxes(tickvals=keep.index.tolist(), ticktext=[f"{spy_to_es(k):,.0f}" for k in keep.index])
    fig.update_layout(showlegend=False)
    return fig

def _line_chart(df, col, title, pos_col, neg_col):
    spot  = df["spot"].iloc[0]
    ser   = df.groupby("strike")[col].sum().sort_index()
    lvls  = key_levels(ser)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=ser.index, y=ser.values, mode="lines",
        line=dict(color=DIM, width=1),
        fill="tozeroy", fillcolor=f"rgba(0,200,240,0.04)",
        hovertemplate="ES %{x}<br>%{y:,.4f}<extra></extra>", showlegend=False,
    ))
    for k in lvls:
        v   = float(ser.get(k, 0))
        col_ = pos_col if v>0 else neg_col
        fig.add_trace(go.Scatter(
            x=[k], y=[v], mode="markers+text",
            marker=dict(color=col_, size=6),
            text=[f"{spy_to_es(k):,.0f}"], textposition="top center",
            textfont=dict(color=col_, size=7, family=FONT),
            hovertemplate=f"ES {spy_to_es(k):,.0f}<br>{v:,.4f}<extra></extra>",
            showlegend=False,
        ))
    fig.add_vline(x=spot, line=dict(color=WHITE, width=0.8, dash="dot"))
    fig.add_hline(y=0,    line=dict(color=LINE,  width=1))
    _theme(fig, title)
    t = _ticks(ser.index)
    fig.update_xaxes(tickvals=t, ticktext=[f"{spy_to_es(k):,.0f}" for k in t])
    return fig

def build_gamma_profile(df): return _line_chart(df, "ngex",   "GAMMA PROFILE",  GREEN, RED)
def build_vanna_profile(df): return _line_chart(df, "nvanna", "VANNA PROFILE",  CYAN,  AMBER)
def build_charm_profile(df): return _line_chart(df, "ncharm", "CHARM PROFILE",  AMBER, RED)

def build_term_structure(df):
    spot = df["spot"].iloc[0]
    atm  = (df.assign(m=lambda x:(x["strike"]-spot).abs())
              .sort_values("m").groupby(["dte","flag"]).first().reset_index())
    calls = atm[atm["flag"]=="C"].sort_values("dte")
    puts  = atm[atm["flag"]=="P"].sort_values("dte")
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=calls["dte"], y=calls["iv"]*100, mode="lines+markers",
        name="call", line=dict(color=CYAN, width=1.2), marker=dict(size=4, color=CYAN),
        hovertemplate="DTE %{x}<br>%{y:.1f}%<extra>call</extra>"))
    fig.add_trace(go.Scatter(x=puts["dte"],  y=puts["iv"]*100,  mode="lines+markers",
        name="put",  line=dict(color=AMBER,width=1.2), marker=dict(size=4, color=AMBER),
        hovertemplate="DTE %{x}<br>%{y:.1f}%<extra>put</extra>"))
    _theme(fig, "IV TERM STRUCTURE")
    fig.update_xaxes(title_text="dte")
    fig.update_yaxes(title_text="IV %")
    fig.update_layout(legend=dict(x=1, xanchor="right", y=1.1, orientation="h"))
    return fig

def build_greeks_kpi(df):
    spot      = df["spot"].iloc[0]
    gex_tot   = df["ngex"].sum()
    vanna_tot = df["nvanna"].sum()
    atm_iv    = (df.assign(m=lambda x:(x["strike"]-spot).abs())
                   .sort_values("m")["iv"].iloc[0]*100)
    gex_cs    = df.groupby("strike")["ngex"].sum().sort_index().cumsum()
    crosses   = gex_cs[gex_cs.shift(1, fill_value=gex_cs.iloc[0]) * gex_cs < 0]
    es_spot   = spy_to_es(spot)
    es_flip   = f"~{spy_to_es(float(crosses.index[0])):.0f}" if not crosses.empty else "—"
    gc        = GREEN if gex_tot>0 else RED
    return {
        "es_spot":   f"{es_spot:,.0f}",
        "gex_tot":   f"{'▲' if gex_tot>0 else '▼'} {abs(gex_tot):,.0f}",
        "gex_col":   gc,
        "vanna_tot": f"{vanna_tot:,.0f}",
        "atm_iv":    f"{atm_iv:.1f}%",
        "es_flip":   es_flip,
    }

# ── Greeks tab JSON builder ────────────────────────────────────────────────────
def build_greeks_json():
    global MIN_OI
    df = fetch_chain()
    if df.empty:
        MIN_OI = 1; df = fetch_chain()
    if df.empty:
        return json.dumps({"error": "No options data"})

    kpi  = build_greeks_kpi(df)
    gex  = build_gex_chart(df)
    gam  = build_gamma_profile(df)
    van  = build_vanna_profile(df)
    ch   = build_charm_profile(df)
    term = build_term_structure(df)

    return json.dumps({
        "kpi":  kpi,
        "gex":  gex.to_json(),
        "gam":  gam.to_json(),
        "van":  van.to_json(),
        "ch":   ch.to_json(),
        "term": term.to_json(),
    })

# ── Heatmap builder ────────────────────────────────────────────────────────────
def _build_heatmap_data():
    BINS = 300; SIGMA = (5, 1.5)
    df = None; sym_used = "ES=F"
    for sym in ["ES=F", "SPY"]:
        try:
            d = yf.Ticker(sym).history(interval="1m", period="1d")
            if not d.empty and len(d) >= 10:
                df = d; sym_used = sym; break
        except: pass
    if df is None: return None, 0

    df = df.reset_index()
    dc = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={dc: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime","Open","High","Low","Close","Volume"]].dropna()

    ctr = fetch_price(sym_used) or float(df["Close"].iloc[-1])
    p_lo = df["Low"].min(); p_hi = df["High"].max()
    rng  = max(p_hi-p_lo, ctr*0.002); pad = rng*0.12
    prices = np.linspace(p_lo-pad, p_hi+pad, BINS)
    price_step = prices[1]-prices[0]
    n = len(df); mat = np.zeros((BINS,n), dtype=np.float64)

    for i,(_, row) in enumerate(df.iterrows()):
        c=float(row["Close"]); lo=float(row["Low"]); hi=float(row["High"]); v=float(row["Volume"])
        if v<=0: continue
        bar_range = max(hi-lo, price_step*2)
        mask=(prices>=lo)&(prices<=hi)
        if mask.any(): mat[mask,i]+=v*0.5
        sigma_p=max(bar_range*0.5, price_step*3)
        mat[:,i]+=v*0.5*np.exp(-0.5*((prices-c)/sigma_p)**2)

    mat=gaussian_filter(mat, sigma=SIGMA)
    if mat.max()>0:
        mat=np.log1p(mat)
        vmax=np.percentile(mat,99); vmin=np.percentile(mat[mat>0],2) if (mat>0).any() else 0
        mat=np.clip((mat-vmin)/max(vmax-vmin,1e-9),0,1)
        mat=np.power(mat,0.4)

    times=df["Datetime"].tolist(); step=max(1,n//8)
    xtvs=list(range(0,n,step)); xttx=[pd.Timestamp(times[i]).strftime("%H:%M") for i in xtvs]
    ts=max(1,BINS//12); ytvs=list(range(0,BINS,ts)); yttx=[f"${prices[i]:.0f}" for i in ytvs]
    sl,sc=get_session(); now_et=datetime.now(ET)
    cr=int(np.argmin(np.abs(prices-ctr)))
    cr_path=[max(0,min(BINS-1,int(np.argmin(np.abs(prices-float(r["Close"])))))) for _,r in df.iterrows()]

    return {
        "data": [
            {"type":"heatmap","z":mat.tolist(),"x":list(range(n)),"y":list(range(BINS)),
             "zmin":0,"zmax":1,
             "colorscale":[[0,"#000000"],[0.08,"#000033"],[0.18,"#0000aa"],
                           [0.30,"#0055ff"],[0.42,"#00aaff"],[0.54,"#00ffcc"],
                           [0.64,"#00ff44"],[0.73,"#aaff00"],[0.81,"#ffee00"],
                           [0.88,"#ff8800"],[0.94,"#ff3300"],[1,"#ff0000"]],
             "showscale":False,"zsmooth":"best"},
            {"type":"scatter","x":list(range(n)),"y":cr_path,"mode":"lines",
             "line":{"color":"rgba(255,255,255,0.55)","width":1.5},"showlegend":False}
        ],
        "layout": {
            "title":{"text":f"ES · Heatmap · <span style='color:{sc}'>{sl}</span> · <span style='color:#00ff88'>${ctr:.2f}</span> · {now_et.strftime('%H:%M ET')}",
                     "font":{"color":"white","size":12,"family":"Courier New"},"x":0.01,"xanchor":"left","y":0.99},
            "xaxis":{"tickvals":xtvs,"ticktext":xttx,"tickfont":{"color":"#555","size":9},"showgrid":False,"zeroline":False},
            "yaxis":{"tickvals":ytvs,"ticktext":yttx,"tickfont":{"color":"#555","size":9},"showgrid":False,"zeroline":False},
            "shapes":[{"type":"line","x0":0,"x1":n-1,"y0":cr,"y1":cr,
                       "line":{"color":"rgba(255,255,255,0.9)","width":1.5,"dash":"dot"}}],
            "paper_bgcolor":"#000000","plot_bgcolor":"#000000",
            "margin":{"l":60,"r":80,"t":30,"b":30},"autosize":True
        }
    }, ctr

# ── Vol surface builder ────────────────────────────────────────────────────────
def _bs(S,K,T,r,sig,opt="call"):
    if T<=0 or sig<=0: return 0.0
    d1=(np.log(S/K)+(r+0.5*sig**2)*T)/(sig*np.sqrt(T)); d2=d1-sig*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) if opt=="call" else K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def _iv(mp,S,K,T,r,opt="call"):
    if T<=0 or mp<=0: return np.nan
    if mp<max(0,(S-K) if opt=="call" else (K-S)): return np.nan
    lo,hi=1e-4,10.0
    for _ in range(200):
        mid=(lo+hi)/2.0; p=_bs(S,K,T,r,mid,opt)
        if abs(p-mp)<1e-5: return mid
        if p<mp: lo=mid
        else:    hi=mid
    return mid if 0.01<mid<9.9 else np.nan

def _build_volsurface_data():
    tk  = yf.Ticker("SPY")
    sh  = tk.history(period="1d")
    if sh.empty: return {"error":"No price data"}, 0
    spot = float(sh["Close"].iloc[-1])
    lp   = fetch_price("SPY")
    if lp: spot = lp

    rows=[]; today=datetime.today()
    for exp in tk.options[:12]:
        try: chain=tk.option_chain(exp)
        except: continue
        T=max((datetime.strptime(exp,"%Y-%m-%d")-today).days/365.0, 1/365)
        for dfo,ot in [(chain.calls,"call"),(chain.puts,"put")]:
            dfo=dfo.copy()
            dfo=dfo[(dfo["strike"]>=spot*0.85)&(dfo["strike"]<=spot*1.15)]
            for _,row in dfo.iterrows():
                yiv=float(row.get("impliedVolatility",0) or 0)
                if 0.01<yiv<3.0:
                    rows.append({"T":T,"moneyness":float(row["strike"])/spot,"iv":yiv*100}); continue
                mid=(float(row["bid"])+float(row["ask"]))/2.0 if float(row["bid"])>0 else float(row["lastPrice"])
                if mid<=0: continue
                iv=_iv(mid,spot,float(row["strike"]),T,0.05,ot)
                if iv and 0.02<iv<4.0:
                    rows.append({"T":T,"moneyness":float(row["strike"])/spot,"iv":iv*100})

    if not rows: return {"error":"No IV data"}, spot
    dfi=pd.DataFrame(rows)
    dv=np.linspace(dfi["T"].min(),dfi["T"].max(),50); mv=np.linspace(dfi["moneyness"].min(),dfi["moneyness"].max(),50)
    gT,gM=np.meshgrid(dv,mv)
    pts=dfi[["T","moneyness"]].values; vals=dfi["iv"].values
    gIV=griddata(pts,vals,(gT,gM),method="cubic"); gn=griddata(pts,vals,(gT,gM),method="nearest")
    mask=np.isnan(gIV); gIV[mask]=gn[mask]
    for j in range(gIV.shape[1]):
        col=gIV[:,j]
        if np.isnan(col).all(): gIV[:,j]=np.nanmean(gIV)
        elif np.isnan(col).any(): col[np.isnan(col)]=np.nanmean(col)

    x_dte=(gT[0,:]*365).tolist(); y_mon=gM[:,0].tolist()
    z_iv=[[float(v) if not np.isnan(v) else None for v in row] for row in gIV.tolist()]
    strike_grid=[[round(y_mon[i]*spot,2) for j in range(len(x_dte))] for i in range(len(y_mon))]
    ai=int(np.argmin(np.abs(mv-1.0))); atm_z=[float(v) if not np.isnan(v) else None for v in gIV[ai,:].tolist()]
    sl,sc=get_session(); now_et=datetime.now(ET)

    return {
        "data":[
            {"type":"surface","x":x_dte,"y":y_mon,"z":z_iv,
             "colorscale":[[0,"#0d0221"],[0.15,"#1a0a3d"],[0.30,"#0b3d91"],
                           [0.45,"#006994"],[0.55,"#00a878"],[0.65,"#f5c518"],
                           [0.80,"#ff6b35"],[1.0,"#ff0000"]],
             "colorbar":{"title":{"text":"IV %","font":{"color":"white","size":11}},
                         "tickfont":{"color":"white","size":9},"thickness":12,"len":0.6},
             "lighting":{"ambient":0.6,"diffuse":0.8,"specular":0.3,"roughness":0.5},
             "opacity":0.92,"customdata":strike_grid,
             "hovertemplate":"Strike: $%{customdata:.2f}<br>DTE: %{x:.0f}d<br>IV: %{z:.1f}%<extra></extra>"},
            {"type":"scatter3d","x":x_dte,"y":[1.0]*len(x_dte),"z":atm_z,"mode":"lines",
             "line":{"color":"white","width":4},"name":"ATM",
             "hovertemplate":f"ATM ${spot:.2f}<br>DTE: %{{x:.0f}}d<br>IV: %{{z:.1f}}%<extra>ATM</extra>"}
        ],
        "layout":{
            "title":{"text":f"<b>SPY · Implied Vol Surface</b>  <span style='color:{sc}'>{sl}</span>  <span style='color:#00ff88'>Spot ${spot:.2f}</span>",
                     "font":{"color":"white","size":13,"family":"Courier New"},"x":0.5,"xanchor":"center","y":0.97},
            "scene":{"xaxis":{"title":{"text":"Days to Expiry"},"tickfont":{"color":"#aaa","size":9},"backgroundcolor":"#080818","gridcolor":"#1a1a2e","showbackground":True},
                     "yaxis":{"title":{"text":"Moneyness (K/S)"},"tickfont":{"color":"#aaa","size":9},"tickformat":".3f","backgroundcolor":"#080818","gridcolor":"#1a1a2e","showbackground":True},
                     "zaxis":{"title":{"text":"Implied Vol (%)"},"tickfont":{"color":"#aaa","size":9},"backgroundcolor":"#080818","gridcolor":"#1a1a2e","showbackground":True},
                     "bgcolor":"#05050f","camera":{"eye":{"x":-1.5,"y":-1.5,"z":0.8}},"aspectratio":{"x":1.4,"y":1.0,"z":0.65}},
            "paper_bgcolor":"#03030a","font":{"color":"white","family":"Courier New"},
            "margin":{"l":0,"r":0,"t":60,"b":0},"autosize":True
        }
    }, spot

# ── HTML template ──────────────────────────────────────────────────────────────
PAGE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>ES Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#000;overflow:hidden;font-family:'Courier New',monospace}
#nav{position:fixed;top:0;left:0;width:100%;height:40px;z-index:1000;
     background:#080808;border-bottom:1px solid #1a1a1a;
     display:flex;align-items:center;padding:0 14px;gap:0}
.tab{height:40px;line-height:40px;padding:0 18px;color:#444;font-size:11px;
     cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:color .2s}
.tab:hover{color:#888}
.tab.on{color:#fff;border-bottom-color:#00ff88}
.sp{flex:1}
#panels{position:fixed;top:40px;left:0;right:0;bottom:0;overflow:hidden}
.panel{position:absolute;top:0;left:0;width:100%;height:100%;display:none}
.panel.on{display:block}
#p-heat .plotly-graph-div,#p-vol{width:100%!important;height:100%!important}
#badge{position:fixed;bottom:6px;right:10px;color:#1a1a1a;font-size:9px;z-index:999;pointer-events:none}
/* Greeks panel */
#p-greek{overflow-y:auto;background:#080b10;padding:10px 12px}
.kpi-strip{display:flex;border:1px solid #151c28;border-radius:3px;background:#0b0f17;margin-bottom:10px}
.kpi-cell{padding:12px 18px;border-right:1px solid #151c28;min-width:110px}
.kpi-cell:last-child{border-right:none}
.kv{font-size:17px;font-weight:500;line-height:1;font-family:monospace}
.kl{font-size:8px;color:#3a4560;margin-top:4px;letter-spacing:1px;text-transform:uppercase;font-family:monospace}
.g-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}
.g-cell{background:#0b0f17;border:1px solid #151c28;border-radius:3px}
.g-full{background:#0b0f17;border:1px solid #151c28;border-radius:3px;margin-bottom:8px}
</style>
</head><body>
<div id="nav">
  <div style="margin-right:12px">[[LOGO]]</div>
  <div class="tab on"  onclick="show('heat',this)">Liquidity Heatmap</div>
  <div class="tab"     onclick="show('vol',this)">Vol Surface 3D</div>
  <div class="tab"     onclick="show('greek',this)">Greeks</div>
  <div class="sp"></div>
  <div style="color:#333;font-size:10px" id="clk">[[NOW]]</div>
  <div style="color:#00ff88;font-size:11px;font-weight:bold;margin-left:10px" id="prc">[[PRICE]]</div>
  <div style="font-size:10px;margin-left:8px;color:[[SC]]" id="sess">[[SL]]</div>
</div>

<div id="panels">
  <div class="panel on" id="p-heat">[[HEAT_HTML]]</div>
  <div class="panel"    id="p-vol" style="width:100%;height:100%"></div>
  <div class="panel"    id="p-greek">
    <div class="kpi-strip" id="g-kpi">
      <div class="kpi-cell"><div class="kv" id="kv-spot" style="color:#dce8ff">—</div><div class="kl">ES futures</div></div>
      <div class="kpi-cell"><div class="kv" id="kv-gex"  style="color:#00d97e">—</div><div class="kl">net GEX</div></div>
      <div class="kpi-cell"><div class="kv" id="kv-van"  style="color:#00c8f0">—</div><div class="kl">net vanna</div></div>
      <div class="kpi-cell"><div class="kv" id="kv-iv"   style="color:#f0a500">—</div><div class="kl">ATM IV</div></div>
      <div class="kpi-cell"><div class="kv" id="kv-flip" style="color:#f03d5f">—</div><div class="kl">γ flip (ES)</div></div>
    </div>
    <div class="g-row">
      <div class="g-cell" style="height:280px"><div id="gc-gex"  style="width:100%;height:100%"></div></div>
      <div class="g-cell" style="height:280px"><div id="gc-gam"  style="width:100%;height:100%"></div></div>
    </div>
    <div class="g-row">
      <div class="g-cell" style="height:280px"><div id="gc-van"  style="width:100%;height:100%"></div></div>
      <div class="g-cell" style="height:280px"><div id="gc-ch"   style="width:100%;height:100%"></div></div>
    </div>
    <div class="g-full" style="height:240px"><div id="gc-term" style="width:100%;height:100%"></div></div>
  </div>
</div>
<div id="badge">live · next update <span id="cd">[[SECS]]s</span></div>

<script>
var _volData  = [[VOL_JSON]];
var _greekData = [[GREEK_JSON]];
var _volInited   = false;
var _greekInited = false;
var PCFG = {responsive:true,displaylogo:false,displayModeBar:false};

function rsz(){
  var W=window.innerWidth, H=window.innerHeight-40;
  ['p-heat'].forEach(function(id){
    var p=document.getElementById(id);
    if(!p||p.style.display==='none') return;
    p.querySelectorAll('.plotly-graph-div').forEach(function(d){
      if(window.Plotly) Plotly.relayout(d,{width:W,height:H});
    });
  });
}
window.addEventListener('load',function(){rsz();setTimeout(rsz,300);setTimeout(rsz,900)});
window.addEventListener('resize',rsz);

function renderGreeks(d){
  if(!d||d.error) return;
  var kpi=d.kpi;
  if(kpi){
    var ks=document.getElementById('kv-spot'); if(ks) ks.textContent=kpi.es_spot||'—';
    var kg=document.getElementById('kv-gex');  if(kg){kg.textContent=kpi.gex_tot||'—'; kg.style.color=kpi.gex_col||'#00d97e';}
    var kv=document.getElementById('kv-van');  if(kv) kv.textContent=kpi.vanna_tot||'—';
    var ki=document.getElementById('kv-iv');   if(ki) ki.textContent=kpi.atm_iv||'—';
    var kf=document.getElementById('kv-flip'); if(kf) kf.textContent=kpi.es_flip||'—';
  }
  var charts=[['gc-gex','gex'],['gc-gam','gam'],['gc-van','van'],['gc-ch','ch'],['gc-term','term']];
  charts.forEach(function(pair){
    var el=document.getElementById(pair[0]);
    if(!el) return;
    try{
      var fig=JSON.parse(d[pair[1]]);
      if(_greekInited){
        Plotly.react(el, fig.data, fig.layout, PCFG);
      } else {
        Plotly.newPlot(el, fig.data, fig.layout, PCFG);
      }
    } catch(e){}
  });
  _greekInited=true;
}

function show(name,el){
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('on')});
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('on');p.style.display='none'});
  el.classList.add('on');
  var panel=document.getElementById('p-'+name);
  panel.style.display='block'; panel.classList.add('on');

  if(name==='vol'&&!_volInited){
    _volInited=true;
    setTimeout(function(){
      var W=window.innerWidth,H=window.innerHeight-40;
      if(_volData.error){panel.innerHTML='<p style="color:#555;padding:4rem">'+_volData.error+'</p>';return;}
      var layout=JSON.parse(JSON.stringify(_volData.layout));
      layout.width=W; layout.height=H; layout.autosize=true;
      Plotly.newPlot(panel,_volData.data,layout,{responsive:true,scrollZoom:true,displaylogo:false});
    },80);
  } else if(name==='greek'&&!_greekInited){
    setTimeout(function(){renderGreeks(_greekData);},80);
  } else {
    setTimeout(rsz,50); setTimeout(rsz,200);
  }
}

setInterval(function(){
  var s=new Date().toLocaleTimeString('en-US',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:true})+' ET';
  var el=document.getElementById('clk'); if(el) el.textContent=s;
},1000);

var cd=[[SECS]];
setInterval(function(){cd--;var el=document.getElementById('cd');if(el)el.textContent=cd+'s';},1000);

// ── Live updates ──────────────────────────────────────────────────────────────
var _updating=false;
function liveUpdate(){
  if(_updating) return; _updating=true;
  fetch('/data').then(function(r){return r.json();}).then(function(d){
    _updating=false;
    var pe=document.getElementById('prc'); if(pe) pe.textContent=d.price;
    var hp=document.getElementById('p-heat');
    var hd=hp?hp.querySelector('.plotly-graph-div'):null;
    if(hd&&window.Plotly&&d.heat) Plotly.react(hd,d.heat.data,d.heat.layout);
    if(_volInited&&d.vol&&!d.vol.error){
      var vp=document.getElementById('p-vol');
      if(vp) Plotly.react(vp,d.vol.data,d.vol.layout);
    } else if(!_volInited&&d.vol){ _volData=d.vol; }
    if(d.greek){ _greekData=d.greek; if(_greekInited) renderGreeks(d.greek); }
    cd=[[SECS]]; var cde=document.getElementById('cd'); if(cde) cde.textContent=cd+'s';
  }).catch(function(){_updating=false;});
}
setInterval(liveUpdate,[[MS]]);
setInterval(function(){
  fetch('/price').then(function(r){return r.json();}).then(function(d){
    var pe=document.getElementById('prc'); if(pe&&d.price) pe.textContent=d.price;
    var se=document.getElementById('sess'); if(se&&d.session){se.textContent=d.session;se.style.color=d.sc||'#fff';}
  }).catch(function(){});
},3000);
</script>
</body></html>"""

# ── Initial heatmap HTML ───────────────────────────────────────────────────────
def build_initial_heatmap_html():
    import plotly.io as pio
    data, ctr = _build_heatmap_data()
    if data is None:
        return "<p style='color:#555;padding:4rem;font-family:Courier New'>No data</p>", 0
    fig = go.Figure(data=data["data"], layout=data["layout"])
    html = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       config={"responsive":True,"scrollZoom":True,"displayModeBar":True,
                               "displaylogo":False})
    return html, ctr

# ── Build full page ────────────────────────────────────────────────────────────
def build_dashboard():
    print("\n  Building heatmap...")
    heat_html, price = build_initial_heatmap_html()
    print("  Building vol surface...")
    vol_data, _      = _build_volsurface_data()
    print("  Building greeks...")
    greek_json       = build_greeks_json()
    sl, sc           = get_session()
    now_str          = datetime.now(ET).strftime("%I:%M:%S %p ET")

    page = PAGE
    page = page.replace("[[LOGO]]",      LOGO_HTML)
    page = page.replace("[[NOW]]",       now_str)
    page = page.replace("[[PRICE]]",     f"${price:.2f}")
    page = page.replace("[[SC]]",        sc)
    page = page.replace("[[SL]]",        sl)
    page = page.replace("[[HEAT_HTML]]", heat_html)
    page = page.replace("[[VOL_JSON]]",  json.dumps(vol_data))
    page = page.replace("[[GREEK_JSON]]",greek_json)
    page = page.replace("[[SECS]]",      str(REFRESH_SECS))
    page = page.replace("[[MS]]",        str(REFRESH_SECS * 1000))
    return page

# ── Server ─────────────────────────────────────────────────────────────────────
HTML_CONTENT = ""
JSON_DATA    = "{}"
PRICE_DATA   = "{}"

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = JSON_DATA.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)
        elif self.path == "/price":
            body = PRICE_DATA.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            body = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass

def refresh_loop():
    global HTML_CONTENT, JSON_DATA, PRICE_DATA
    vol_cache   = None
    greek_cache = None
    vol_tick    = 0
    greek_tick  = 0
    while True:
        time.sleep(REFRESH_SECS)
        try:
            ts = datetime.now(ET).strftime("%H:%M:%S")
            print(f"\n[{ts}] Refreshing heatmap...")
            heat_data, price = _build_heatmap_data()
            sl, sc = get_session()
            PRICE_DATA = json.dumps({"price":f"${price:.2f}","session":sl,"sc":sc})

            vol_tick   += REFRESH_SECS
            greek_tick += REFRESH_SECS

            if vol_cache is None or vol_tick >= VOL_REFRESH_SECS:
                print("  Rebuilding vol surface...")
                vol_cache, _ = _build_volsurface_data()
                vol_tick = 0

            if greek_cache is None or greek_tick >= VOL_REFRESH_SECS:
                print("  Rebuilding greeks...")
                greek_cache = json.loads(build_greeks_json())
                greek_tick = 0

            JSON_DATA = json.dumps({
                "price":   f"${price:.2f}",
                "session": sl, "sc": sc,
                "heat":    heat_data,
                "vol":     vol_cache,
                "greek":   greek_cache,
            })
            print(f"  Done — ${price:.2f}")
        except Exception as e:
            print(f"  Error: {e}")

def free_port(start=8765):
    for p in range(start, start+20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p)); return p
        except: continue
    raise RuntimeError("No free port")

def main():
    global HTML_CONTENT
    print("\n╔══════════════════════════════════════════╗")
    print("║   ES FUTURES COMBINED DASHBOARD          ║")
    print("║   Heatmap · Vol Surface · Greeks         ║")
    print("╚══════════════════════════════════════════╝\n")
    HTML_CONTENT = build_dashboard()
    port = free_port()
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=refresh_loop, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    def go():
        time.sleep(0.8)
        import webbrowser; webbrowser.open(url)
        print(f"  Opened: {url}  (Ctrl+C to stop)\n")
    threading.Thread(target=go, daemon=True).start()
    print(f"  Server: {url}")
    try: httpd.serve_forever()
    except KeyboardInterrupt: print("\n  Stopped.")
    finally: httpd.shutdown()

if __name__ == "__main__":
    main()
