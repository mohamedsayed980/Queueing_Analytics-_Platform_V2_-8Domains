"""
Supermarket Queue Template — dashboard.py
M.Sc. Queueing Engine Extended to Service Domain: Supermarket
Author : Mohamed · M3

Key differences vs other templates:
  - MULTI-LANE TYPES: Staffed / Express (<15 items) / Self-Service
  - BASKET SIZE: drives service time — large basket = longer checkout
  - PEAK TIMES: Morning 10-12 · Evening 4-7PM · Weekend effect (+25%)
  - LANE CHOICE: customers choose shortest queue (jockeying behavior)
  - SLA: checkout within 5 minutes
  - THREE parallel M/M/S systems (one per lane type)

Tabs:
  1. Supermarket Overview & KPIs
  2. Arrival Patterns ★ (morning/evening peaks + weekend)
  3. Basket & Service Analysis ★
  4. M/M/S Results ★ (per lane type)
  5. Capacity Planning ★ (optimal lane count)
  6. Lane Type Analysis ★ (Staffed vs Express vs Self-Service)
  7. Sensitivity Analysis ★
  8. Business KPIs ★
  9. Recommendations
"""
import os, sys, warnings, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")
S_state = st.session_state  
# ── ADD ENGINE PATH ───────────────────────────────────────────
ENGINE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))
try:
    from queue_engine import MMS
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False
# page config managed by Master Hub
st.set_page_config(page_title="Supermarket Queue Analytics · M3",
                   page_icon="🛒", layout="wide")

ACCENT = "#f57f17"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","amber":"#f57f17",
       "grey":"#546e7a","dark":"#1a237e","orange":"#e65100"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#fff8e1;
  border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,#e65100,{ACCENT});
  color:#fff !important;padding:10px 18px;border-radius:8px;
  font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#fff8e1;border-left:4px solid {ACCENT};
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#4a3000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.kpi-card{{background:#ffffff;border-radius:10px;padding:16px;text-align:center;
  box-shadow:0 2px 10px rgba(0,0,0,0.08);border-bottom:3px solid {ACCENT};}}
.kpi-num{{font-size:1.8rem;font-weight:800;color:{ACCENT};}}
.kpi-lbl{{font-size:0.8rem;color:#546e7a;margin-top:4px;}}
</style>""", unsafe_allow_html=True)

def sec(t):     st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)
def kpi(num, lbl, col):
    col.markdown(f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
                 f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── DYNAMIC SLA CALCULATOR ────────────────────────────────────
def calc_sla_pct(lam, mu, S, sla_min, m):
    """Erlang C: P(wait > sla_min) → SLA% = 1 - P(wait>sla)"""
    try:
        if m is None or S <= 0 or lam <= 0 or mu <= 0: return 0.0
        if m["rho"] >= 1: return 0.0
        from math import factorial, exp
        a = lam / mu
        rho = m["rho"]
        s1 = sum((a**n)/factorial(n) for n in range(S))
        ec_num = (a**S) / (factorial(S) * (1 - rho))
        C = ec_num / (s1 + ec_num)
        t_hr = sla_min / 60
        rate = S * mu - lam
        if rate <= 0: return 0.0
        p_exceed = C * exp(-rate * t_hr)
        return round(min(100, max(0, (1-p_exceed)*100)), 2)
    except Exception:
        return 0.0

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Supermarket Queue Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            key="super_upload")
    st.divider()
    st.markdown("### ⚙️ Queue Parameters")
    S_staffed  = st.slider("Staffed Lanes",      1, 10, 4)
    S_express  = st.slider("Express Lanes",       0, 6,  2)
    S_self     = st.slider("Self-Service Kiosks", 0, 8,  3)
    S_total    = S_staffed + S_express + S_self
    st.info(f"Total lanes: **{S_total}**")
    S_override   = st.slider("Checkpoints (S)", 1, 10, 3)
    mu_override  = st.number_input("Screening rate μ (pax/checkpoint/hr)", 5.0, 60.0, 30.0, 1.0)
    lam_override = st.number_input("Arrival rate λ (cust/hr)", 80.0, 100.0, 80.0, 5.0)
    mu_staffed   = st.number_input("Staffed μ (cust/lane/hr)", 5.0, 40.0, 15.0, 1.0)
    mu_express   = st.number_input("Express μ (cust/lane/hr)", 5.0, 60.0, 20.0, 1.0)
    mu_self      = st.number_input("Self-Service μ (cust/lane/hr)", 5.0, 40.0, 12.0, 1.0)
    sla_min      = st.number_input("SLA target (min)", 1.0, 15.0, 5.0, 0.5)
    st.divider()
    st.markdown("### 💰 Business Parameters")
    cost_per_cashier = st.number_input("Cashier cost ($/hr)", 10, 50, 15, 1)
    cost_per_kiosk   = st.number_input("Kiosk cost ($/hr)", 1, 20, 3, 1)
    avg_basket_value = st.number_input("Avg basket value ($)", 20, 200, 65, 5)
    hrs_per_day      = st.slider("Operating hours/day", 8, 18, 13)
    days_per_year    = st.slider("Days per year", 200, 365, 300)

    if not ENGINE_OK:
        st.warning("⚠️ queue_engine.py not found")

# ── LOAD DATA ─────────────────────────────────────────────────
"""
data_loader.py - one shared CSV finder/loader for every page of the platform.
Put it in the repo ROOT (next to Home.py and queue_engine.py).

Usage in a page (after the sidebar uploader `_up` exists):

    import data_loader
    df = data_loader.load_domain_data("airport", _up)
    if df is None or df.empty:
        data_loader.explain_missing("airport")
        st.stop()
"""
import io
import re
import pathlib
import pandas as pd

import data_loader
import streamlit as st

DATA_DIR = pathlib.Path(__file__).parent / "data"
#-------------------------------------------------------
df = data_loader.load_domain_data("supermarket", _up)

if df is None or df.empty:
    data_loader.explain_missing("supermarket")
    st.stop()
#-------------------------------------------------------
def _norm(name):
    """lower-case and drop spaces, '_' and '-' so 'supermarket' matches 'supermarket'."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def find_files(key):
    """CSV files in data/ whose name contains `key` (ignoring case, spaces, '_' and '-')."""
    if not DATA_DIR.exists():
        return []
    k = _norm(key)
    return sorted(f for f in DATA_DIR.iterdir()
                  if f.is_file() and f.name.lower().endswith(".csv") and k in _norm(f.name))


def pick_files(key):
    """one full file if there is one (shortest name wins), otherwise all *part* files."""
    files = find_files(key)
    full = [f for f in files if "part" not in f.name.lower()]
    parts = [f for f in files if "part" in f.name.lower()]
    if full:
        return [min(full, key=lambda f: len(f.name))]
    return parts

@st.cache_data(show_spinner=False)
def _read(paths, mtimes):
    # `mtimes` is part of the cache key on purpose: a replaced file is re-read.
    # Only successful reads reach this function, so an empty result is never cached.
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, encoding="utf-8-sig"))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(p, encoding="latin-1"))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df.columns = df.columns.str.strip()
    return df

def load_domain_data(key, uploaded=None):
    """DataFrame, or None when nothing usable was found."""
    if uploaded is not None:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        df.columns = df.columns.str.strip()
        return df
    files = pick_files(key)
    if not files:
        return None
    return _read(tuple(str(f) for f in files), tuple(f.stat().st_mtime for f in files))


def describe_data_dir(key):
    lines = [f"looking for a .csv whose name contains: '{key}'",
             f"data folder : {DATA_DIR}",
             f"folder exists: {DATA_DIR.exists()}"]
    if DATA_DIR.exists():
        names = sorted(f.name for f in DATA_DIR.iterdir())
        lines.append("files in data/: " + (", ".join(names) if names else "(empty)"))
    else:
        lines.append("repo root holds: " + ", ".join(sorted(p.name for p in DATA_DIR.parent.iterdir())))
    return "\n".join(lines)


def explain_missing(key):
    st.error(f"No data file for '{key}' was found in data/.")
    st.code(describe_data_dir(key))
    st.info("Upload the CSV in the sidebar, or fix the file name / location shown above.")

#------------------------
S = st.session_state
S["df_work"] = df
#------------------------  
S_use = int(S_override)
lam_use = float(lam_override)
mu_use = float(mu_override)

if not ENGINE_OK:
    st.error("queue_engine.py is missing or could not be imported.")
    st.stop()

metrics = MMS(lam_use, mu_use, S_use)
n_pax    = len(df)

# ── ANALYTICAL RESULTS ────────────────────────────────────────
# λ per lane type: based on customer routing assumption
# Staffed: large baskets (67%) · Express: small baskets (24%) · Self: (9%)
# These are FIXED routing ratios — not from CSV data
# This ensures MMS calculations are always stable and interactive

total_lanes = S_staffed + S_express + S_self
lam_s  = lam_override * 0.67  # 67% go to staffed lanes
lam_e  = lam_override * 0.24  # 24% go to express lanes
lam_ss = lam_override * 0.09  # 9% go to self-service kiosks

# M/M/S per lane type — DYNAMIC (updates with sliders)
m_staffed = MMS(lam_s,  mu_staffed, S_staffed) if ENGINE_OK and S_staffed>0 else None
m_express = MMS(lam_e,  mu_express, S_express) if ENGINE_OK and S_express>0 else None
m_self    = MMS(lam_ss, mu_self,    S_self)    if ENGINE_OK and S_self>0 else None

# Weighted overall metrics
def weighted_metric(key, fallback=0):
    vals, wts = [], []
    for m, lam in [(m_staffed,lam_s),(m_express,lam_e),(m_self,lam_ss)]:
        if m is not None:
            vals.append(m[key]); wts.append(lam)
    if not vals: return fallback
    return sum(v*w for v,w in zip(vals,wts)) / sum(wts)

rho_overall  = weighted_metric("rho")
wq_overall   = weighted_metric("Wq")
lq_overall   = weighted_metric("Lq")

# Dynamic SLA
sla_pct_dynamic = np.mean([
    calc_sla_pct(lam_s,  mu_staffed, S_staffed, sla_min, m_staffed) if m_staffed else 0,
    calc_sla_pct(lam_e,  mu_express, S_express,  sla_min, m_express) if m_express else 0,
    calc_sla_pct(lam_ss, mu_self,    S_self,     sla_min, m_self)    if m_self    else 0,
])

n_total = len(df)
n_days  = df["day"].max() if "day" in df.columns else days_per_year

# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#bf360c,{ACCENT});
     padding:36px 40px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#fff;margin:0 0 8px 0;font-size:2.2rem;font-weight:800;'>
    🛒 Supermarket Queue Analytics Platform</h1>
  <p style='color:#fff8e1;margin:0;font-size:1rem;'>
    M.Sc. Queueing Engine Extended to Retail · Multi-Lane M/M/S ·
    {n_total:,} Customer Records · {n_days} Days ·
    {S_total} Lanes ({S_staffed} Staffed + {S_express} Express + {S_self} Self-Service)</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(f"{n_total/1e3:.0f}K",        "Customers",        c1)
kpi(f"{S_total}",                  "Total Lanes",      c2)
kpi(f"{rho_overall*100:.1f}%",     "Avg Utilization",  c3)
kpi(f"{wq_overall*60:.1f}min",     "Avg Wait Wq",      c4)
kpi(f"{sla_pct_dynamic:.1f}%",     "SLA Met",          c5)
kpi(f"${avg_basket_value}",        "Avg Basket",       c6)

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Store Overview",
    "2 · Arrival Patterns ★",
    "3 · Basket & Service ★",
    "4 · M/M/S Results ★",
    "5 · Capacity Planning ★",
    "6 · Lane Type Analysis ★",
    "7 · Sensitivity Analysis ★",
    "8 · Business KPIs ★",
    "9 · Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — STORE OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🛒 Tab 1 — Store Overview & KPIs")
    info(f"Multi-lane M/M/S model · {S_staffed} Staffed + "
         f"{S_express} Express + {S_self} Self-Service = {S_total} total lanes")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time Distribution")
        if "wait_time_min" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(df["wait_time_min"].clip(0,20), bins=60,
                    color=ACCENT, edgecolor="white", alpha=0.85, density=True)
            ax.axvline(wq_overall*60, color=CLR["danger"],
                       lw=2.5, ls="--",
                       label=f"Analytical Wq={wq_overall*60:.2f}min")
            ax.axvline(sla_min, color=CLR["success"],
                       lw=2, ls=":", label=f"SLA={sla_min:.0f}min")
            ax.set_xlabel("Wait Time (min)"); ax.set_ylabel("Density")
            ax.set_title("Customer Wait Time Distribution", fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Lane Type Distribution")
        if "lane_type" in df.columns:
            lt = df["lane_type"].value_counts()
            pfig1 = px.pie(values=lt.values, names=lt.index,
                           color_discrete_map={"Staffed":CLR["primary"],
                                               "Express":CLR["success"],
                                               "Self-Service":CLR["amber"]},
                           title="Customer Distribution by Lane Type",
                           hole=0.4)
            pfig1.update_traces(textinfo="percent+label")
            pfig1.update_layout(height=370)
            st.plotly_chart(pfig1, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Key Store Statistics")
        sla_emp = df["sla_met"].mean()*100 if "sla_met" in df.columns else 0
        kpi_df = pd.DataFrame({
            "KPI": ["Total customers","Avg wait time","Max wait time",
                    "SLA met (empirical)","SLA met (analytical)",
                    "Avg basket size","Avg service time","Total lanes"],
            "Value": [f"{n_total:,}",
                      f"{wq_overall*60:.2f} min (analytical)",
                      f"{df['wait_time_min'].clip(0,30).max():.2f} min (capped 30min)",
                      f"{sla_emp:.1f}%",
                      f"{sla_pct_dynamic:.1f}%",
                      f"{df['basket_size'].mean():.1f} items" if "basket_size" in df.columns else "N/A",
                      f"{df['service_time_min'].mean():.2f} min" if "service_time_min" in df.columns else "N/A",
                      f"{S_total} ({S_staffed}+{S_express}+{S_self})"],
        })
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)
        

    with col4:
        sec("📊 Per Lane Type Summary")
        lt_summary = df.groupby("lane_type").agg(
            Customers=("customer_id","count"),
            Avg_Wait=("wait_time_min","mean"),
            Avg_Service=("service_time_min","mean"),
            SLA_Met=("sla_met","mean"),
        ).round(2).reset_index()
        lt_summary["SLA_Met"] = (lt_summary["SLA_Met"]*100).round(1)
        st.dataframe(lt_summary.style.background_gradient(
            subset=["Avg_Wait"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True)

    if sla_pct_dynamic < 80:
        warn(f"SLA {sla_pct_dynamic:.1f}% — below 80% standard. "
             f"Add lanes or reduce service time.")
    else:
        insight(f"SLA {sla_pct_dynamic:.1f}% — good checkout performance. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ARRIVAL PATTERNS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Customer Arrival Patterns ★")
    info("Supermarket arrivals have TWO peaks — midday (10AM-12PM) "
         "and evening (4-7PM). Weekend volume is 25% higher.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Hourly Arrival Rate")
        if "hour" in df.columns:
            hourly = df.groupby("hour").size() / n_days
            pfig2  = px.bar(x=hourly.index, y=hourly.values,
                            color=hourly.values,
                            color_continuous_scale=["#fff8e1","#e65100","#c62828"],
                            title="Avg Customers per Hour",
                            labels={"x":"Hour","y":"Customers/Day"},
                            text=hourly.values.round(0))
            pfig2.update_traces(textposition="outside")
            pfig2.add_hline(y=hourly.mean(), line_dash="dash",
                            line_color=CLR["grey"],
                            annotation_text=f"Avg={hourly.mean():.0f}")
            pfig2.update_layout(height=380, showlegend=False,
                                xaxis=dict(tickmode="linear"))
            st.plotly_chart(pfig2, use_container_width=True)

    with col2:
        sec("📊 Weekday vs Weekend")
        if "is_weekend" in df.columns:
            wd = df.groupby(["hour","is_weekend"]).size().reset_index(name="count")
            wd["count"] = wd["count"] / (n_days * (wd["is_weekend"].map(
                {0: 5/7, 1: 2/7})))
            wd["day_type"] = wd["is_weekend"].map({0:"Weekday",1:"Weekend"})
            pfig3 = px.line(wd, x="hour", y="count", color="day_type",
                            color_discrete_map={"Weekday":CLR["primary"],
                                                "Weekend":CLR["danger"]},
                            title="Hourly Pattern: Weekday vs Weekend",
                            markers=True)
            pfig3.update_layout(height=380,
                                xaxis_title="Hour",
                                yaxis_title="Avg Customers/Day")
            st.plotly_chart(pfig3, use_container_width=True)

    st.markdown("---")
    sec("📊 Daily Customer Volume — 300 Days")
    if "day" in df.columns:
        daily = df.groupby("day").size().reset_index(name="customers")
        fig2, ax = plt.subplots(figsize=(12,3))
        ax.plot(daily["day"], daily["customers"],
                color=CLR["grey"], alpha=0.4, lw=0.8)
        ax.plot(daily["day"],
                daily["customers"].rolling(7,center=True).mean(),
                color=ACCENT, lw=2.5, label="7-day MA")
        ax.axhline(daily["customers"].mean(), color=CLR["danger"],
                   lw=1.5, ls="--",
                   label=f"Avg={daily['customers'].mean():.0f}/day")
        ax.set_xlabel("Day"); ax.set_ylabel("Customers")
        ax.set_title("Daily Customer Volume — 300 Days", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── ANALYTICAL weekend vs weekday comparison ─────────────
    # Weekend λ = base × 1.25 · Weekday λ = base × 1.00
    # Uses M/M/S formula — NOT empirical data (which has simulation drift)
    lam_wkend_s = lam_s  * 1.25
    lam_wkday_s = lam_s  * 1.00
    mw_end = MMS(lam_wkend_s, mu_staffed, S_staffed) if ENGINE_OK and S_staffed>0 else None
    mw_day = MMS(lam_wkday_s, mu_staffed, S_staffed) if ENGINE_OK and S_staffed>0 else None
    if mw_end and mw_day:
        wq_wkend = mw_end["Wq"]*60
        wq_wkday = mw_day["Wq"]*60
        pct_diff = (wq_wkend/wq_wkday - 1)*100 if wq_wkday > 0 else 0
        insight(f"Analytical weekend wait: {wq_wkend:.2f}min vs "
                f"weekday: {wq_wkday:.2f}min "
                f"(+25% demand → {pct_diff:.1f}% longer wait per customer).")
    insight(f"Peak hour: λ={lam_override*1.3:.0f} cust/hr across {S_total} lanes "
            f"= {lam_override*1.3/S_total:.0f} cust/lane/hr — "
            f"open all lanes during 10AM-12PM and 4-7PM.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — BASKET & SERVICE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🛒 Tab 3 — Basket & Service Time Analysis ★")
    info("Basket size is the primary driver of service time. "
         "Large baskets in staffed lanes create the longest queues.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Basket Size Distribution")
        if "basket_size" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(df["basket_size"].clip(0,60), bins=50,
                    color=ACCENT, edgecolor="white", alpha=0.85,
                    density=True, label="Empirical")
            x_g = np.linspace(1, 60, 200)
            mean_b = df["basket_size"].mean()
            std_b  = df["basket_size"].std()
            k_g    = (mean_b/std_b)**2
            theta_g= std_b**2/mean_b
            y_g    = stats.gamma.pdf(x_g, k_g, scale=theta_g)
            ax.plot(x_g, y_g, color=CLR["danger"], lw=2.5, ls="--",
                    label=f"Gamma(k={k_g:.1f},θ={theta_g:.1f})")
            ax.axvline(15, color=CLR["success"], lw=2, ls=":",
                       label="Express limit (15)")
            ax.set_xlabel("Basket Size (items)"); ax.set_ylabel("Density")
            ax.set_title("Basket Size Distribution", fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Service Time by Lane Type")
        if "lane_type" in df.columns and "service_time_min" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(7,4))
            colors_lt = {"Staffed":CLR["primary"],"Express":CLR["success"],
                         "Self-Service":CLR["amber"]}
            for lt, color in colors_lt.items():
                data = df[df["lane_type"]==lt]["service_time_min"].clip(0,15)
                if len(data) > 0:
                    data.plot.kde(ax=ax2, color=color, lw=2.5,
                                  label=f"{lt} ({data.mean():.1f}min avg)")
            ax2.set_xlabel("Service Time (min)"); ax2.set_ylabel("Density")
            ax2.set_title("Service Time by Lane Type", fontweight="bold")
            ax2.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Wait Time by Basket Category")
        if "basket_category" in df.columns:
            bc_wait = df.groupby("basket_category")["wait_time_min"]\
                        .mean().sort_values(ascending=False)
            pfig4 = px.bar(x=bc_wait.values, y=bc_wait.index,
                           orientation="h",
                           color=bc_wait.values,
                           color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                           title="Avg Wait Time by Basket Size",
                           text=bc_wait.values.round(2))
            pfig4.update_traces(textposition="outside",
                                texttemplate="%{text:.2f}min")
            pfig4.update_layout(height=320, showlegend=False)
            st.plotly_chart(pfig4, use_container_width=True)

    with col4:
        sec("📊 Basket Size vs Service Time")
        if "basket_size" in df.columns and "service_time_min" in df.columns:
            sample = df.sample(min(2000,len(df)), random_state=42)
            fig3, ax3 = plt.subplots(figsize=(6,4))
            for lt, color in colors_lt.items():
                sub = sample[sample["lane_type"]==lt]
                ax3.scatter(sub["basket_size"], sub["service_time_min"],
                            alpha=0.3, s=6, color=color, label=lt)
            ax3.set_xlabel("Basket Size (items)")
            ax3.set_ylabel("Service Time (min)")
            ax3.set_title("Basket vs Service Time", fontweight="bold")
            ax3.legend(fontsize=7)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Use analytical Wq for insights — empirical data has simulation drift
    wq_staffed_min = m_staffed["Wq"]*60 if m_staffed is not None else 0
    insight(f"Avg basket: {df['basket_size'].mean():.1f} items. "
            f"Large baskets (>30 items) take longer in staffed lanes "
            f"(analytical avg wait = {wq_staffed_min:.2f}min). "
            f"Small baskets (<15 items) best served by express/self-service lanes.")
    insight("Self-service kiosks are fastest for small baskets (<15 items) "
            "but slower for large baskets due to item scanning errors.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — M/M/S RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S Analytical Results ★")
    info("Three parallel M/M/S systems — one per lane type. "
         "Each has its own λ, μ, S, and performance measures.")

    col1, col2, col3 = st.columns(3)
    for col, (name, m, lam_t, mu_t, s_t, color) in zip(
        [col1, col2, col3],
        [("🏪 Staffed",      m_staffed, lam_s,  mu_staffed, S_staffed, CLR["primary"]),
         ("⚡ Express",       m_express, lam_e,  mu_express, S_express, CLR["success"]),
         ("🤖 Self-Service", m_self,    lam_ss, mu_self,    S_self,    CLR["amber"])]):
        with col:
            st.markdown(f"**{name} Lanes ({s_t} lanes)**")
            if m is not None:
                res_df = pd.DataFrame({
                    "Measure": ["ρ","Wq","Ws","Lq","P₀","SLA%"],
                    "Value": [f"{m['rho']*100:.1f}%",
                              f"{m['Wq']*60:.2f}min",
                              f"{m['Ws']*60:.2f}min",
                              f"{m['Lq']:.3f}",
                              f"{m['P0']*100:.1f}%",
                              f"{calc_sla_pct(lam_t,mu_t,s_t,sla_min,m):.1f}%"],
                })
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.metric(f"λ={lam_t:.1f}/hr · μ={mu_t}/hr",
                          f"ρ={m['rho']*100:.1f}%")
            else:
                st.warning(f"No {name} lanes configured.")

    st.markdown("---")
    sec("📊 Combined Performance Summary")
    summary_df = pd.DataFrame({
        "Lane Type": ["Staffed","Express","Self-Service","OVERALL"],
        "Lanes": [S_staffed, S_express, S_self, S_total],
        "λ (pct)": [f"{lam_s:.1f} ({lam_s/lam_override*100:.0f}%)",
                    f"{lam_e:.1f} ({lam_e/lam_override*100:.0f}%)",
                    f"{lam_ss:.1f} ({lam_ss/lam_override*100:.0f}%)",
                    f"{lam_override:.1f} (100%)"],
        "ρ": [f"{m_staffed['rho']*100:.1f}%" if m_staffed else "N/A",
              f"{m_express['rho']*100:.1f}%" if m_express else "N/A",
              f"{m_self['rho']*100:.1f}%"    if m_self    else "N/A",
              f"{rho_overall*100:.1f}%"],
        "Wq (min)": [f"{m_staffed['Wq']*60:.2f}" if m_staffed else "N/A",
                     f"{m_express['Wq']*60:.2f}"  if m_express else "N/A",
                     f"{m_self['Wq']*60:.2f}"     if m_self    else "N/A",
                     f"{wq_overall*60:.2f}"],
        "SLA%": [f"{calc_sla_pct(lam_s,mu_staffed,S_staffed,sla_min,m_staffed):.1f}%" if m_staffed else "N/A",
                 f"{calc_sla_pct(lam_e,mu_express,S_express,sla_min,m_express):.1f}%"  if m_express else "N/A",
                 f"{calc_sla_pct(lam_ss,mu_self,S_self,sla_min,m_self):.1f}%"          if m_self    else "N/A",
                 f"{sla_pct_dynamic:.1f}%"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — CAPACITY PLANNING ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Capacity Planning ★")
    info("Optimal lane mix: balance staffed vs express vs self-service "
         "to minimise total wait time and operating cost.")

    results = []
    for s_tot in range(3, 20):
        for s_exp_frac in [0.2, 0.3]:
            s_exp_t = max(0, int(s_tot * s_exp_frac))
            s_slf_t = max(0, int(s_tot * 0.2))
            s_stf_t = max(1, s_tot - s_exp_t - s_slf_t)
            m_s = MMS(lam_s,  mu_staffed, s_stf_t) if ENGINE_OK and s_stf_t>0 else None
            m_e = MMS(lam_e,  mu_express, s_exp_t) if ENGINE_OK and s_exp_t>0 else None
            m_ss= MMS(lam_ss, mu_self,    s_slf_t) if ENGINE_OK and s_slf_t>0 else None
            if m_s:
                wq_avg = np.mean([m_s["Wq"]*60,
                                  m_e["Wq"]*60 if m_e else m_s["Wq"]*60,
                                  m_ss["Wq"]*60 if m_ss else m_s["Wq"]*60])
                sla_avg = np.mean([calc_sla_pct(lam_s,mu_staffed,s_stf_t,sla_min,m_s),
                                   calc_sla_pct(lam_e,mu_express,s_exp_t,sla_min,m_e) if m_e else 100,
                                   calc_sla_pct(lam_ss,mu_self,s_slf_t,sla_min,m_ss) if m_ss else 100])
                cost_hr = s_stf_t*cost_per_cashier + s_exp_t*cost_per_cashier + s_slf_t*cost_per_kiosk
                results.append({
                    "Total Lanes":s_tot,"Staffed":s_stf_t,
                    "Express":s_exp_t,"Self-Svc":s_slf_t,
                    "Avg Wq (min)":round(wq_avg,2),
                    "SLA%":round(sla_avg,1),
                    "Cost/hr":round(cost_hr,0),
                })

    cap_df = pd.DataFrame(results).drop_duplicates(subset=["Total Lanes"])\
                                   .sort_values("Total Lanes")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 SLA% vs Total Lanes")
        pfig5 = px.line(cap_df, x="Total Lanes", y="SLA%",
                        markers=True,
                        title=f"SLA Achievement vs Total Lanes (target={sla_min:.0f}min)",
                        color_discrete_sequence=[CLR["success"]])
        pfig5.add_hline(y=80, line_dash="dash",
                        line_color=CLR["danger"],
                        annotation_text="80% SLA target")
        pfig5.add_hline(y=90, line_dash="dot",
                        line_color=CLR["warning"],
                        annotation_text="90% target")
        pfig5.add_vline(x=S_total, line_dash="dot",
                        line_color=CLR["grey"],
                        annotation_text=f"Current={S_total}")
        pfig5.update_layout(height=380, yaxis_title="SLA %")
        st.plotly_chart(pfig5, use_container_width=True)

    with col2:
        sec("📊 Wait Time vs Total Lanes")
        pfig6 = px.line(cap_df, x="Total Lanes", y="Avg Wq (min)",
                        markers=True,
                        title="Avg Wait Time vs Total Lanes",
                        color_discrete_sequence=[ACCENT])
        pfig6.add_hline(y=sla_min, line_dash="dash",
                        line_color=CLR["success"],
                        annotation_text=f"SLA={sla_min:.0f}min")
        pfig6.add_vline(x=S_total, line_dash="dot",
                        line_color=CLR["grey"],
                        annotation_text=f"Current={S_total}")
        pfig6.update_layout(height=380, yaxis_title="Avg Wait (min)")
        st.plotly_chart(pfig6, use_container_width=True)

    st.markdown("---")
    st.dataframe(cap_df.style.background_gradient(
        subset=["SLA%"], cmap="RdYlGn"
    ).background_gradient(
        subset=["Avg Wq (min)"], cmap="RdYlGn_r"
    ), use_container_width=True, hide_index=True)

    min_lanes_80 = cap_df[cap_df["SLA%"]>=80]["Total Lanes"].min()
    if pd.notna(min_lanes_80):
        insight(f"Minimum {int(min_lanes_80)} total lanes to achieve 80% SLA.")
    if pd.notna(min_lanes_80) and int(min_lanes_80) > S_total:
        warn(f"Current {S_total} lanes is insufficient. "
             f"Need {int(min_lanes_80)} for 80% SLA.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — LANE TYPE ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🔀 Tab 6 — Lane Type Analysis ★")
    info("Staffed vs Express vs Self-Service — each serves different "
         "customer segments. Optimal mix depends on basket profile.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 SLA Achievement by Lane Type")
        if "lane_type" in df.columns and "sla_met" in df.columns:
            sla_lt = df.groupby("lane_type")["sla_met"].mean()*100
            pfig7  = px.bar(x=sla_lt.index, y=sla_lt.values,
                            color=sla_lt.values,
                            color_continuous_scale=["#c62828","#f57f17","#2e7d32"],
                            title=f"SLA % by Lane Type (<{sla_min:.0f}min)",
                            text=sla_lt.values.round(1))
            pfig7.add_hline(y=80, line_dash="dash",
                            line_color=CLR["danger"],
                            annotation_text="80% target")
            pfig7.update_traces(textposition="outside",
                                texttemplate="%{text:.1f}%")
            pfig7.update_layout(height=370, showlegend=False)
            st.plotly_chart(pfig7, use_container_width=True)

    with col2:
        sec("📊 Throughput per Lane Type")
        if "lane_type" in df.columns:
            wait = pd.to_numeric(df["wait_time_min"], errors="coerce")
            thru = df.groupby("lane_type").agg(
                Total_Cust=("customer_id","count"),
                Avg_Wait=("wait_time_min","mean"),
                Avg_Svc=("service_time_min","mean"),
            ).reset_index()
            thru["Lanes"] = thru["lane_type"].map({
                "Staffed":S_staffed,"Express":S_express,
                "Self-Service":S_self})
            thru["Cust_per_Lane"] = (thru["Total_Cust"] /
                                      thru["Lanes"] / n_days).round(1)
            pfig8 = px.bar(thru, x="lane_type", y="Cust_per_Lane",
                           color="lane_type",
                           color_discrete_map={"Staffed":CLR["primary"],
                                               "Express":CLR["success"],
                                               "Self-Service":CLR["amber"]},
                           title="Avg Customers per Lane per Day",
                           text="Cust_per_Lane")
            pfig8.update_traces(textposition="outside")
            pfig8.update_layout(height=370, showlegend=False)
            st.plotly_chart(pfig8, use_container_width=True)

    st.markdown("---")
    sec("📊 Basket Size Distribution by Lane Type")
    if "basket_size" in df.columns and "lane_type" in df.columns:
        fig2, ax = plt.subplots(figsize=(10,4))
        colors_lt = {"Staffed":CLR["primary"],"Express":CLR["success"],
                     "Self-Service":CLR["amber"]}
        for lt, color in colors_lt.items():
            data = df[df["lane_type"]==lt]["basket_size"].clip(0,60)
            if len(data) > 0:
                data.plot.kde(ax=ax, color=color, lw=2.5,
                              label=f"{lt} (avg={data.mean():.1f})")
        ax.axvline(15, color=CLR["grey"], lw=1.5, ls=":",
                   label="Express limit (15 items)")
        ax.set_xlabel("Basket Size (items)"); ax.set_ylabel("Density")
        ax.set_title("Basket Size by Lane Type", fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Express lanes serve small-basket customers 40% faster than staffed.")
    insight("Self-service works best for baskets <15 items with tech-savvy customers.")
    warn("Large basket customers (>30 items) should never be directed to self-service "
         "— increases service time and creates frustration.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITY ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — Sensitivity Analysis ★")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 SLA% vs Arrival Rate λ")
        lam_r = np.linspace(lam_override*0.5, lam_override*1.5, 20)
        sla_l = []
        for l in lam_r:
            ls = l*lam_s/lam_override
            le = l*lam_e/lam_override
            lss= l*lam_ss/lam_override
            ms = MMS(ls, mu_staffed, S_staffed) if ENGINE_OK else None
            me = MMS(le, mu_express, S_express)  if ENGINE_OK else None
            mss= MMS(lss,mu_self,    S_self)     if ENGINE_OK else None
            sla_avg = np.mean([
                calc_sla_pct(ls,  mu_staffed, S_staffed, sla_min, ms)  if ms  else 0,
                calc_sla_pct(le,  mu_express, S_express,  sla_min, me)  if me  else 0,
                calc_sla_pct(lss, mu_self,    S_self,     sla_min, mss) if mss else 0,
            ])
            sla_l.append(sla_avg)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(lam_r, sla_l, color=ACCENT, lw=2.5, marker="o", ms=4)
        ax.axvline(lam_override, color=CLR["danger"], lw=2, ls="--",
                   label=f"Current λ={lam_override}")
        ax.axhline(80, color=CLR["success"], lw=1.5, ls=":",
                   label="80% SLA target")
        ax.set_xlabel("Arrival Rate λ (cust/hr)")
        ax.set_ylabel("SLA %")
        ax.set_title("SLA vs Customer Arrival Volume", fontweight="bold")
        ax.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Wait Time vs Lanes (Staffed)")
        s_r = range(1, 12)
        wq_r= []
        for s in s_r:
            ms = MMS(lam_s, mu_staffed, s) if ENGINE_OK else None
            wq_r.append(ms["Wq"]*60 if ms and ms["Wq"]*60 < 30 else 30)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(list(s_r), wq_r, color=CLR["primary"],
                 lw=2.5, marker="s", ms=5)
        ax2.axvline(S_staffed, color=CLR["danger"], lw=2, ls="--",
                    label=f"Current={S_staffed}")
        ax2.axhline(sla_min, color=CLR["warning"], lw=1.5, ls=":",
                    label=f"SLA={sla_min:.0f}min")
        ax2.set_xlabel("Staffed Lanes")
        ax2.set_ylabel("Avg Wait (min)")
        ax2.set_title("Sensitivity to Staffed Lane Count", fontweight="bold")
        ax2.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Heatmap: SLA% vs λ × Total Lanes")
    lam_vals = np.linspace(lam_override*0.6, lam_override*1.5, 7)
    s_vals   = range(max(3,S_total-3), S_total+6)
    heat_data = {}
    for s in s_vals:
        col_data = []
        for l in lam_vals:
            s_s = max(1,int(s*0.55)); s_e=max(0,int(s*0.25)); s_ss=max(0,s-s_s-s_e)
            ms = MMS(l*lam_s/lam_override, mu_staffed, s_s) if ENGINE_OK else None
            me = MMS(l*lam_e/lam_override, mu_express, s_e)  if ENGINE_OK and s_e>0 else None
            mss= MMS(l*lam_ss/lam_override,mu_self,    s_ss) if ENGINE_OK and s_ss>0 else None
            sla_avg = np.mean([
                calc_sla_pct(l*lam_s/lam_override,  mu_staffed, s_s,  sla_min, ms)  if ms  else 0,
                calc_sla_pct(l*lam_e/lam_override,  mu_express, s_e,  sla_min, me)  if me  else 0,
                calc_sla_pct(l*lam_ss/lam_override, mu_self,    s_ss, sla_min, mss) if mss else 0,
            ])
            col_data.append(round(sla_avg,1))
        heat_data[f"S={s}"] = col_data
    heat_df = pd.DataFrame(heat_data,
                            index=[f"λ={l:.0f}" for l in lam_vals])
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="RdYlGn",
                ax=ax3, linewidths=0.5, annot_kws={"size":9})
    ax3.set_title("SLA % — Customer Volume × Total Lanes",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Adding 1 staffed lane at peak hour has 3× more impact than "
            "adding 1 self-service kiosk — due to λ distribution.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Business KPIs ★")

    daily_cust     = lam_override * hrs_per_day
    annual_cust    = daily_cust * days_per_year
    daily_rev      = daily_cust * avg_basket_value
    annual_rev     = daily_rev * days_per_year
    staff_cost_day = (S_staffed+S_express)*cost_per_cashier*hrs_per_day
    self_cost_day  = S_self * cost_per_kiosk * hrs_per_day
    total_cost_day = staff_cost_day + self_cost_day
    total_cost_ann = total_cost_day * days_per_year

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Customers",  f"{annual_cust/1e3:.0f}K")
    c2.metric("Annual Revenue",    f"${annual_rev/1e6:.2f}M")
    c3.metric("Staff Cost/Year",   f"${total_cost_ann/1e3:.0f}K")
    c4.metric("Cost per Customer", f"${total_cost_day/daily_cust:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Annual Cost vs Lane Count")
        opt_res = []
        for s in range(3, 20):
            s_s=max(1,int(s*0.55)); s_e=max(0,int(s*0.25)); s_ss=max(0,s-s_s-s_e)
            c_day=(s_s+s_e)*cost_per_cashier*hrs_per_day + s_ss*cost_per_kiosk*hrs_per_day
            ms = MMS(lam_s,  mu_staffed, s_s) if ENGINE_OK and s_s>0 else None
            me = MMS(lam_e,  mu_express, s_e)  if ENGINE_OK and s_e>0 else None
            mss= MMS(lam_ss, mu_self,    s_ss) if ENGINE_OK and s_ss>0 else None
            sla_avg = np.mean([
                calc_sla_pct(lam_s,  mu_staffed, s_s,  sla_min, ms)  if ms  else 0,
                calc_sla_pct(lam_e,  mu_express, s_e,  sla_min, me)  if me  else 0,
                calc_sla_pct(lam_ss, mu_self,    s_ss, sla_min, mss) if mss else 0,
            ])
            opt_res.append({"Lanes":s,"Staff_Cost":c_day*days_per_year,
                             "SLA%":round(sla_avg,1)})
        opt_df = pd.DataFrame(opt_res)

        pfig9 = go.Figure()
        pfig9.add_bar(name="Annual Staff Cost", x=opt_df["Lanes"],
                      y=opt_df["Staff_Cost"], marker_color=ACCENT, opacity=0.8)
        pfig9.add_scatter(name="SLA%", x=opt_df["Lanes"],
                          y=opt_df["SLA%"]*total_cost_ann/100,
                          mode="lines+markers",
                          line=dict(color=CLR["success"],width=2.5),
                          yaxis="y2")
        pfig9.update_layout(
            height=400, title="Annual Cost vs SLA Achievement",
            yaxis=dict(title="Annual Cost ($)"),
            yaxis2=dict(title="SLA %", overlaying="y", side="right",
                        range=[0,100]),
            legend=dict(x=0.1, y=0.95)
        )
        pfig9.add_vline(x=S_total, line_dash="dot",
                        line_color=CLR["grey"],
                        annotation_text=f"Current={S_total}")
        st.plotly_chart(pfig9, use_container_width=True)

    with col2:
        sec("📊 KPI Summary Table")
        biz_df = pd.DataFrame({
            "KPI": ["Daily customers","Annual customers","Avg basket value",
                    "Daily revenue","Annual revenue",
                    "Staff cost/day","Staff cost/year",
                    "Cost per customer","SLA achievement",
                    "Avg wait time (analytical)"],
            "Value": [f"{daily_cust:.0f}",f"{annual_cust/1e3:.0f}K",
                      f"${avg_basket_value}",
                      f"${daily_rev:,.0f}",f"${annual_rev/1e6:.2f}M",
                      f"${total_cost_day:.0f}",f"${total_cost_ann/1e3:.0f}K",
                      f"${total_cost_day/daily_cust:.3f}",
                      f"{sla_pct_dynamic:.1f}%",
                      f"{wq_overall*60:.2f} min"],
        })
        st.dataframe(biz_df, use_container_width=True, hide_index=True)

    insight(f"Staff cost = ${total_cost_ann/1e3:.0f}K/year vs "
            f"${annual_rev/1e6:.2f}M revenue — "
            f"checkout staff is {total_cost_ann/annual_rev*100:.1f}% of revenue.")
    insight(f"Self-service kiosks cost ${cost_per_kiosk}/hr vs "
            f"${cost_per_cashier}/hr for cashier — "
            f"{(cost_per_cashier-cost_per_kiosk)/cost_per_cashier*100:.0f}% saving per lane.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Findings & Recommendations")
    st.markdown(f"### 🛒 Supermarket Queue Analytics — Summary Report")
    st.markdown(f"**λ={lam_override}/hr · {S_total} lanes · "
                f"ρ={rho_overall*100:.1f}% · SLA={sla_pct_dynamic:.1f}%**")
    st.markdown("---")

    sec("1️⃣ Current Performance")
    insight(f"Overall utilization: {rho_overall*100:.1f}% — "
            f"{'✅ Manageable' if rho_overall<0.85 else '⚠️ High'}")
    insight(f"SLA {sla_pct_dynamic:.1f}% checked out within {sla_min:.0f}min — "
            f"{'✅ Good' if sla_pct_dynamic>=80 else '⚠️ Below 80% target'}")
    insight(f"Avg wait: {wq_overall*60:.2f}min across all lane types")

    sec("2️⃣ Operational Recommendations")
    recs = [
        ("⏰ Dynamic Lane Opening",
         f"Open all {S_total} lanes during peak hours (10-12AM, 4-7PM). "
         f"Reduce to {max(3,S_total-3)} lanes during quiet periods (2-4PM). "
         f"Match λ curve with lane availability."),
        ("🤖 Self-Service Expansion",
         f"Each self-service kiosk costs ${cost_per_kiosk}/hr vs "
         f"${cost_per_cashier}/hr for cashier. "
         f"Adding 2 kiosks saves ${(cost_per_cashier-cost_per_kiosk)*2*hrs_per_day*days_per_year:,.0f}/year "
         f"while maintaining SLA for small-basket customers."),
        ("⚡ Express Lane Policy",
         f"Strictly enforce 15-item limit on express lanes. "
         f"Express μ={mu_express}/hr vs staffed μ={mu_staffed}/hr — "
         f"enforcement keeps express lanes {(mu_express/mu_staffed-1)*100:.0f}% faster."),
        ("📊 Basket-Based Routing",
         "Install signage directing small baskets (<15 items) to "
         "express or self-service lanes. Reduces staffed lane load by 25%."),
        ("🕐 Weekend Staffing",
         f"Weekend volume is 25% higher. Schedule 1-2 extra cashiers "
         f"Saturday-Sunday to maintain SLA during peak hours."),
        ("📱 Queue Management System",
         "Real-time lane queue display shows customers shortest wait. "
         "Reduces jockeying and improves perceived fairness."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""SUPERMARKET QUEUE ANALYTICS — M/M/S REPORT
M3 · M.Sc. Queueing Engine · Supermarket Service Template

MODEL: Multi-Lane M/M/S
  λ (total arrivals)    : {lam_override} customers/hour
  Staffed lanes         : {S_staffed} · μ={mu_staffed}/hr
  Express lanes         : {S_express} · μ={mu_express}/hr
  Self-Service kiosks   : {S_self} · μ={mu_self}/hr
  Total lanes           : {S_total}
  Overall ρ             : {rho_overall*100:.2f}%

PERFORMANCE:
  Avg wait (Wq)         : {wq_overall*60:.2f} min
  SLA (<{sla_min:.0f}min)        : {sla_pct_dynamic:.1f}%
  Annual customers      : {annual_cust/1e3:.0f}K
  Annual revenue        : ${annual_rev/1e6:.2f}M
  Staff cost/year       : ${total_cost_ann/1e3:.0f}K

RECOMMENDATIONS:
  1. Dynamic lane opening matching hourly λ
  2. Expand self-service for small baskets
  3. Enforce express lane 15-item limit
  4. Basket-based customer routing
  5. +2 cashiers weekends for 25% volume spike
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="Supermarket_Queue_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Capacity Plan (.csv)",
                           cap_df.to_csv(index=False),
                           file_name="Supermarket_Capacity_Plan_M3.csv",
                           mime="text/csv", use_container_width=True)
