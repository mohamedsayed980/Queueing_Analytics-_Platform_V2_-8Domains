"""
Restaurant Queue Template — dashboard.py
M.Sc. Queueing Engine Extended to Service Domain: Restaurant
Author : Mohamed · M3

Key differences vs Bank/Healthcare/CallCenter/Airport:
  - FINITE CAPACITY: M/M/S/K model — when full, parties turned away
  - REVENUE KPI: covers/hr · revenue per seat · lost revenue
  - TWO SESSIONS: Lunch (40min) vs Dinner (70min) meal durations
  - NO-SHOW RATE: 15% of reservations don't arrive
  - PARTY SIZE: 1-8 people per table
  - BLOCKING PROBABILITY: P(turned away) = key metric

Tabs:
  1. Restaurant Overview & KPIs
  2. Arrival Patterns ★ (lunch/dinner peaks)
  3. Meal Duration Analysis ★
  4. M/M/S/K Results ★ (finite capacity)
  5. Capacity Planning ★ (optimal table count)
  6. Revenue Analysis ★ (unique to restaurant)
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
from math import factorial, exp
import streamlit as st

warnings.filterwarnings("ignore")

ENGINE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))
try:
    from queue_engine import MMS, MMSS
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False

st.set_page_config(page_title="Restaurant Queue Analytics · M3",
                   page_icon="🍽️", layout="wide")

ACCENT = "#2e7d32"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","amber":"#f57f17",
       "grey":"#546e7a","dark":"#1a237e","green":"#2e7d32"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#e8f5e9;
  border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#1b5e20);
  color:#fff !important;padding:10px 18px;border-radius:8px;
  font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e8f5e9;border-left:4px solid {ACCENT};
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
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

# ── FINITE CAPACITY M/M/S/K SOLVER ───────────────────────────
def mmsk_metrics(lam, mu, S, K=None):
    """
    M/M/S/K: S servers, finite capacity K (max in system)
    If K=None → use K=S (no waiting, only blocking)
    For restaurant: S=tables, K=S (no queue — either seated or turned away)
    """
    if K is None:
        K = S
    if lam <= 0 or mu <= 0 or S <= 0:
        return None
    a = lam / mu
    rho = lam / (S * mu)

    # State probabilities P(n) for M/M/S/K
    P = {}
    P[0] = 1.0
    for n in range(1, K+1):
        if n <= S:
            P[n] = (a**n / factorial(n)) * P[0]
        else:
            P[n] = (a**n / (factorial(S) * S**(n-S))) * P[0]

    # Normalize
    total = sum(P.values())
    for n in P:
        P[n] /= total

    P0      = P[0]
    PK      = P[K]          # blocking probability (turned away)
    lam_eff = lam * (1 - PK)  # effective arrival rate

    # Average number in system
    Ls = sum(n * P[n] for n in P)

    # Average number in queue
    Lq = sum(max(0, n-S) * P[n] for n in P)

    # Little's Law
    Ws = Ls / lam_eff if lam_eff > 0 else 0
    Wq = Lq / lam_eff if lam_eff > 0 else 0

    # Utilization
    rho_eff = lam_eff / (S * mu)

    return {
        "rho":      round(rho_eff, 4),
        "rho_nom":  round(rho, 4),
        "Lq":       round(Lq, 4),
        "Ls":       round(Ls, 4),
        "Wq":       round(Wq, 4),
        "Ws":       round(Ws, 4),
        "P0":       round(P0, 4),
        "P_block":  round(PK, 4),   # probability of being turned away
        "lam_eff":  round(lam_eff, 4),
        "throughput": round(lam_eff, 4),
    }

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Restaurant Queue Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            key="rest_upload")
    st.divider()
    st.markdown("### ⚙️ Queue Parameters")
    S_override   = st.slider("Tables (S)", 5, 50, 20)
    lam_override = st.number_input("Peak arrival rate λ (parties/hr)",
                                    1.0, 100.0, 25.0, 1.0)
    mu_override  = st.number_input("Table turn rate μ (tables/hr)",
                                    1.0, 30.0, 12.0, 0.5)
    st.divider()
    st.markdown("### 💰 Business Parameters")
    avg_spend     = st.number_input("Avg spend per person ($)", 10, 200, 35, 5)
    avg_party     = st.number_input("Avg party size", 1.0, 8.0, 3.3, 0.1)
    no_show_rate  = st.slider("No-show rate (%)", 0, 30, 15) / 100
    res_rate      = st.slider("Reservation rate (%)", 0, 100, 40) / 100
    hrs_per_day   = st.slider("Operating hours/day", 6, 16, 11)
    days_per_year = st.slider("Days per year", 200, 365, 300)

    if not ENGINE_OK:
        st.warning("⚠️ queue_engine.py not found")

# ── LOAD DATA ─────────────────────────────────────────────────

import io
import re
import pathlib
import pandas as pd
import data_loader
import streamlit as st

DATA_DIR = pathlib.Path(__file__).parent / "data"
#-------------------------------------------------------
df = data_loader.load_domain_data("restaurant", _up)

if df is None or df.empty:
    data_loader.explain_missing("restaurant")
    st.stop()
#-------------------------------------------------------
def _norm(name):
    """lower-case and drop spaces, '_' and '-' so 'restaurant' matches 'restaurant'."""
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


# ── ANALYTICAL RESULTS (M/M/S/K) ─────────────────────────────
metrics = mmsk_metrics(lam_use, mu_use, S_use, K=S_use)

if metrics is None:
    st.error("❌ Invalid parameters."); st.stop()

# Subsets
served   = df[(df["turned_away"]==0) & (df["no_show"]==0)].copy()
turned   = df[df["turned_away"]==1].copy()
noshows  = df[df["no_show"]==1].copy()
n_total  = len(df)

# Dynamic KPIs from analytical model
block_pct    = metrics["P_block"] * 100
throughput   = metrics["lam_eff"]
rho_pct      = metrics["rho"] * 100
daily_rev    = throughput * hrs_per_day * avg_party * avg_spend
annual_rev   = daily_rev * days_per_year
lost_rev_day = (lam_use * metrics["P_block"]) * hrs_per_day * avg_party * avg_spend
lost_rev_ann = lost_rev_day * days_per_year

# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1b5e20,{ACCENT});
     padding:36px 40px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#fff;margin:0 0 8px 0;font-size:2.2rem;font-weight:800;'>
    🍽️ Restaurant Queue Analytics Platform</h1>
  <p style='color:#c8e6c9;margin:0;font-size:1rem;'>
    M.Sc. Queueing Engine Extended to Restaurant · M/M/S/K Finite Capacity ·
    {n_total:,} Party Records · {df["day"].max() if "day" in df.columns else 300} Days</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(f"{n_total:,}",          "Total Parties",      c1)
kpi(f"{S_use}",              "Tables",             c2)
kpi(f"{rho_pct:.1f}%",       "Table Utilization",  c3)
kpi(f"{block_pct:.1f}%",     "Turned Away",        c4)
kpi(f"${daily_rev:,.0f}",    "Est. Daily Revenue", c5)
kpi(f"{metrics['Ws']*60:.1f}min","Avg Meal Time",  c6)

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Restaurant Overview",
    "2 · Arrival Patterns ★",
    "3 · Meal Duration ★",
    "4 · M/M/S/K Results ★",
    "5 · Capacity Planning ★",
    "6 · Revenue Analysis ★",
    "7 · Sensitivity Analysis ★",
    "8 · Business KPIs ★",
    "9 · Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RESTAURANT OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🍽️ Tab 1 — Restaurant Overview & KPIs")
    info(f"M/M/{S_use}/{S_use} model · λ={lam_use}/hr · μ={mu_use}/hr · "
         f"ρ={metrics['rho']:.3f} · Finite capacity = {S_use} tables")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Party Outcome Distribution")
        outcome = pd.DataFrame({
            "Outcome": ["Served","Turned Away","No-Show"],
            "Count":   [len(served), len(turned), len(noshows)],
            "Pct":     [len(served)/n_total*100,
                        len(turned)/n_total*100,
                        len(noshows)/n_total*100],
        })
        pfig1 = px.bar(outcome, x="Outcome", y="Count",
                       color="Outcome",
                       color_discrete_map={"Served":CLR["success"],
                                            "Turned Away":CLR["danger"],
                                            "No-Show":CLR["warning"]},
                       title="Party Outcome Distribution",
                       text=outcome["Pct"].apply(lambda x: f"{x:.1f}%"))
        pfig1.update_traces(textposition="outside")
        pfig1.update_layout(height=370, showlegend=False)
        st.plotly_chart(pfig1, use_container_width=True)

    with col2:
        sec("📊 Session Mix — Lunch vs Dinner")
        if "session" in served.columns:
            sess = served["session"].value_counts()
            pfig2 = px.pie(values=sess.values, names=sess.index,
                           color_discrete_map={"Lunch":CLR["amber"],
                                               "Dinner":CLR["dark"]},
                           title="Served Parties: Lunch vs Dinner",
                           hole=0.4)
            pfig2.update_traces(textinfo="percent+label")
            pfig2.update_layout(height=370)
            st.plotly_chart(pfig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Key Restaurant Statistics")
        kpi_df = pd.DataFrame({
            "KPI": ["Total parties","Served","Turned away","No-shows",
                    "Avg meal duration","Avg party size",
                    "Avg revenue per table","Total revenue (data)"],
            "Value": [f"{n_total:,}",
                      f"{len(served):,} ({len(served)/n_total*100:.1f}%)",
                      f"{len(turned):,} ({len(turned)/n_total*100:.1f}%)",
                      f"{len(noshows):,} ({len(noshows)/n_total*100:.1f}%)",
                      f"{served['meal_duration_min'].mean():.1f} min",
                      f"{served['party_size'].mean():.1f} people",
                      f"${served['revenue'].mean():.2f}",
                      f"${df['revenue'].sum():,.0f}"],
        })
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    with col4:
        sec("📊 M/M/S/K Model Summary")
        model_df = pd.DataFrame({
            "Parameter": ["Model","λ (arrival rate)","μ (turn rate)",
                          "S (tables)","K (capacity)",
                          "ρ (utilization)","P_block (turned away)",
                          "Effective throughput","Avg meal time (Ws)"],
            "Value": [f"M/M/{S_use}/{S_use}",
                      f"{lam_use} parties/hr",
                      f"{mu_use} tables/hr ({60/mu_use:.0f} min/turn)",
                      f"{S_use}",f"{S_use}",
                      f"{metrics['rho']:.4f} ({rho_pct:.1f}%)",
                      f"{metrics['P_block']:.4f} ({block_pct:.1f}%)",
                      f"{metrics['lam_eff']:.2f} parties/hr",
                      f"{metrics['Ws']*60:.1f} min"],
        })
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    if block_pct > 15:
        warn(f"{block_pct:.1f}% of parties turned away — "
             f"add tables to capture lost revenue.")
    elif block_pct > 5:
        warn(f"{block_pct:.1f}% turned away — consider adding "
             f"{max(1, int(S_use*0.1))} tables during peak hours.")
    else:
        insight(f"Only {block_pct:.1f}% turned away — good capacity management. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ARRIVAL PATTERNS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Arrival Patterns ★")
    info("Restaurant arrivals have TWO peaks — lunch (12-2PM) and dinner (7-9PM). "
         "Staffing and table allocation must match this pattern.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Hourly Party Arrivals")
        if "hour" in df.columns:
            hourly_all    = df.groupby("hour").size() / df["day"].max()
            hourly_served = served.groupby("hour").size() / df["day"].max()
            hourly_turned = turned.groupby("hour").size() / df["day"].max()

            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(hourly_all.index, hourly_all.values,
                   color=CLR["grey"], alpha=0.4, label="Total Arrivals")
            ax.bar(hourly_served.index, hourly_served.values,
                   color=CLR["success"], alpha=0.85, label="Served")
            ax.bar(hourly_turned.index, hourly_turned.values,
                   color=CLR["danger"], alpha=0.85,
                   bottom=[hourly_served.get(h,0) for h in hourly_turned.index],
                   label="Turned Away")
            ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Parties/Day")
            ax.set_title("Hourly Arrivals — Served vs Turned Away",
                         fontweight="bold")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Turned Away by Hour")
        if "hour" in turned.columns:
            ta_hourly = turned.groupby("hour").size() / df["day"].max()
            pfig3 = px.bar(x=ta_hourly.index, y=ta_hourly.values,
                           color=ta_hourly.values,
                           color_continuous_scale=["#fff3e0","#e65100","#c62828"],
                           title="Avg Turned Away per Hour",
                           labels={"x":"Hour","y":"Parties/Day"},
                           text=ta_hourly.values.round(1))
            pfig3.update_traces(textposition="outside")
            pfig3.update_layout(height=370, showlegend=False)
            st.plotly_chart(pfig3, use_container_width=True)

    st.markdown("---")
    sec("📊 Daily Party Volume — 300 Days")
    if "day" in df.columns:
        daily = df.groupby("day").agg(
            Total=("day","count"),
            Served=("turned_away", lambda x: (x==0).sum()),
            Turned=("turned_away","sum"),
            Revenue=("revenue","sum"),
        ).reset_index()
        pfig4 = go.Figure()
        pfig4.add_scatter(x=daily["day"],
                          y=daily["Served"].rolling(7,center=True).mean(),
                          name="Served (7-day MA)", mode="lines",
                          line=dict(color=CLR["success"],width=2.5))
        pfig4.add_scatter(x=daily["day"],
                          y=daily["Turned"].rolling(7,center=True).mean(),
                          name="Turned Away (7-day MA)", mode="lines",
                          line=dict(color=CLR["danger"],width=2,dash="dot"))
        pfig4.update_layout(height=340,
                             title="Daily Party Volume Trend",
                             yaxis_title="Parties/Day")
        st.plotly_chart(pfig4, use_container_width=True)

    peak_hours = [12,13,19,20]
    peak_turned = turned[turned["hour"].isin(peak_hours)]
    insight(f"Peak hours {peak_hours} account for "
            f"{len(peak_turned)/len(turned)*100:.1f}% of all turned-away parties.")
    insight(f"Dinner peak (7-9PM) generates "
            f"{served[served['session']=='Dinner']['revenue'].sum()/served['revenue'].sum()*100:.1f}% "
            f"of total revenue.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — MEAL DURATION ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("⏱ Tab 3 — Meal Duration Analysis ★")
    info("Table turn rate μ = 1/avg_meal_duration. "
         "Faster turns = more revenue — but must balance with dining experience.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Meal Duration Distribution")
        if "meal_duration_min" in served.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(served["meal_duration_min"].clip(0,150), bins=60,
                    color=CLR["success"], edgecolor="white",
                    alpha=0.85, density=True, label="All Sessions")
            lunch_dur  = served[served["session"]=="Lunch"]["meal_duration_min"]
            dinner_dur = served[served["session"]=="Dinner"]["meal_duration_min"]
            if len(lunch_dur) > 0:
                lunch_dur.clip(0,120).plot.kde(ax=ax, color=CLR["amber"],
                                               lw=2.5, label=f"Lunch (μ={lunch_dur.mean():.0f}min)")
            if len(dinner_dur) > 0:
                dinner_dur.clip(0,150).plot.kde(ax=ax, color=CLR["dark"],
                                                lw=2.5, label=f"Dinner (μ={dinner_dur.mean():.0f}min)")
            ax.set_xlabel("Meal Duration (min)"); ax.set_ylabel("Density")
            ax.set_title("Meal Duration: Lunch vs Dinner", fontweight="bold")
            ax.legend(fontsize=8); plt.tight_layout()
            st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Meal Duration by Party Size")
        if "party_size" in served.columns and "meal_duration_min" in served.columns:
            ps_dur = served.groupby("party_size")["meal_duration_min"]\
                           .mean().round(1)
            pfig5 = px.bar(x=ps_dur.index, y=ps_dur.values,
                           color=ps_dur.values,
                           color_continuous_scale=["#e8f5e9","#2e7d32"],
                           title="Avg Meal Duration by Party Size (min)",
                           text=ps_dur.values.round(0),
                           labels={"x":"Party Size","y":"Duration (min)"})
            pfig5.update_traces(textposition="outside")
            pfig5.update_layout(height=370, showlegend=False)
            st.plotly_chart(pfig5, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Table Occupancy by Hour")
        if "hour" in served.columns and "meal_duration_min" in served.columns:
            occ = served.groupby("hour").agg(
                Avg_Duration=("meal_duration_min","mean"),
                Avg_Revenue=("revenue","mean"),
            ).round(2)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.bar(occ.index, occ["Avg_Duration"],
                    color=CLR["teal"], edgecolor="white", alpha=0.85)
            ax2.axhline(served["meal_duration_min"].mean(),
                        color=CLR["danger"], lw=2, ls="--",
                        label=f"Overall avg={served['meal_duration_min'].mean():.0f}min")
            ax2.set_xlabel("Hour"); ax2.set_ylabel("Avg Duration (min)")
            ax2.set_title("Avg Meal Duration by Hour", fontweight="bold")
            ax2.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    with col4:
        sec("📊 Revenue by Party Size")
        if "party_size" in served.columns and "revenue" in served.columns:
            ps_rev = served.groupby("party_size")["revenue"].mean().round(2)
            pfig6 = px.bar(x=ps_rev.index, y=ps_rev.values,
                           color=ps_rev.values,
                           color_continuous_scale=["#e8f5e9","#1b5e20"],
                           title="Avg Revenue by Party Size ($)",
                           text=ps_rev.values.round(0),
                           labels={"x":"Party Size","y":"Revenue ($)"})
            pfig6.update_traces(textposition="outside",
                                texttemplate="$%{text:.0f}")
            pfig6.update_layout(height=320, showlegend=False)
            st.plotly_chart(pfig6, use_container_width=True)

    mu_emp = 60 / served["meal_duration_min"].mean() if "meal_duration_min" in served.columns else mu_use
    insight(f"Empirical μ={mu_emp:.2f} tables/hr · Avg meal: "
            f"{served['meal_duration_min'].mean():.1f} min.")
    insight(f"Dinner meals avg {served[served['session']=='Dinner']['meal_duration_min'].mean():.0f}min "
            f"vs Lunch {served[served['session']=='Lunch']['meal_duration_min'].mean():.0f}min "
            f"— plan staffing accordingly.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — M/M/S/K RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S/K Analytical Results ★")
    info(f"M/M/{S_use}/{S_use} finite capacity model. "
         "When all tables full, arriving parties are turned away (lost business).")

    c1,c2,c3 = st.columns(3)
    c1.metric("Table Utilization ρ", f"{metrics['rho']*100:.2f}%")
    c2.metric("Blocking P_block",    f"{metrics['P_block']*100:.2f}%")
    c3.metric("Effective Throughput",f"{metrics['lam_eff']:.2f} parties/hr")
    c4,c5,c6 = st.columns(3)
    c4.metric("Avg Tables Occupied", f"{metrics['Ls']:.2f}")
    c5.metric("Avg Meal Time Ws",    f"{metrics['Ws']*60:.1f} min")
    c6.metric("P(all tables free)",  f"{metrics['P0']*100:.2f}%")

    st.markdown("---")
    sec("📊 Full M/M/S/K Performance Table")
    full_df = pd.DataFrame({
        "Symbol":["λ","μ","S","K","ρ","P₀","P_K (block)","Ls","Lq","Ws","Wq","λ_eff"],
        "Name":  ["Arrival rate","Turn rate","Tables","Max capacity",
                  "Utilization","All-tables-free prob","Blocking prob",
                  "Avg tables occupied","Avg queue","Avg meal time",
                  "Avg queue wait","Effective throughput"],
        "Value": [f"{lam_use} parties/hr",
                  f"{mu_use} tables/hr ({60/mu_use:.0f} min/meal)",
                  f"{S_use} tables",f"{S_use} (no waiting queue)",
                  f"{metrics['rho']:.4f} ({metrics['rho']*100:.1f}%)",
                  f"{metrics['P0']:.4f} ({metrics['P0']*100:.1f}%)",
                  f"{metrics['P_block']:.4f} ({block_pct:.1f}%)",
                  f"{metrics['Ls']:.4f} tables",
                  f"{metrics['Lq']:.4f}",
                  f"{metrics['Ws']*60:.2f} min",
                  f"{metrics['Wq']*60:.2f} min",
                  f"{metrics['lam_eff']:.3f} parties/hr"],
        "Restaurant Meaning": [
            "Walk-in + reservation arrivals",
            "How fast tables turn over",
            "Total seating capacity",
            "No waiting — seat or turn away",
            "% of time tables are busy",
            "% of time all tables are empty",
            "% of parties turned away",
            "How many tables occupied on avg",
            "No queue — parties leave if full",
            "Avg time party occupies table",
            "No waiting queue — 0 or turned away",
            "Actual parties served per hour"],
    })
    st.dataframe(full_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    sec("📊 Analytical vs Empirical Comparison")
    comp_df = pd.DataFrame({
        "Metric": ["Blocking % (turned away)","Table utilization ρ",
                   "Avg meal duration","Effective throughput"],
        "Analytical": [f"{block_pct:.2f}%",f"{rho_pct:.1f}%",
                       f"{metrics['Ws']*60:.1f} min",
                       f"{metrics['lam_eff']:.2f} parties/hr"],
        "Empirical":  [f"{len(turned)/n_total*100:.2f}%",
                       f"{served['meal_duration_min'].mean()/60*mu_use*100:.1f}%" if "meal_duration_min" in served.columns else "N/A",
                       f"{served['meal_duration_min'].mean():.1f} min" if "meal_duration_min" in served.columns else "N/A",
                       f"{len(served)/df['day'].max()/hrs_per_day:.2f} parties/hr"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    if block_pct > 20:
        warn(f"Blocking {block_pct:.1f}% — 1 in 5 parties turned away. "
             f"Add {max(2,int(S_use*0.15)):.0f} tables to reduce below 10%.")
    elif block_pct > 10:
        warn(f"Blocking {block_pct:.1f}% — consider adding "
             f"{max(1,int(S_use*0.10)):.0f} tables during peak hours.")
    else:
        insight(f"Blocking {block_pct:.1f}% is manageable. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 5 — CAPACITY PLANNING ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Capacity Planning ★")
    info("How many tables maximise revenue? "
         "More tables = less blocking + more revenue, but higher fixed costs.")

    results = []
    for s in range(5, 41):
        m = mmsk_metrics(lam_use, mu_use, s, K=s)
        if m:
            results.append({
                "Tables":    s,
                "ρ":         round(m["rho"],4),
                "Block_%":   round(m["P_block"]*100,2),
                "Throughput":round(m["lam_eff"],3),
                "Ws_min":    round(m["Ws"]*60,1),
                "Daily_Rev": round(m["lam_eff"]*hrs_per_day*avg_party*avg_spend,0),
            })

    cap_df = pd.DataFrame(results)

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Blocking % vs Tables")
        pfig7 = px.line(cap_df, x="Tables", y="Block_%",
                        markers=True,
                        title="Blocking Probability vs Table Count",
                        color_discrete_sequence=[CLR["danger"]])
        pfig7.add_hline(y=10, line_dash="dash",
                        line_color=CLR["warning"],
                        annotation_text="10% target")
        pfig7.add_hline(y=5, line_dash="dot",
                        line_color=CLR["success"],
                        annotation_text="5% target")
        pfig7.add_vline(x=S_use, line_dash="dot",
                        line_color=CLR["grey"],
                        annotation_text=f"Current S={S_use}")
        pfig7.update_layout(height=380,
                             yaxis_title="Blocking %")
        st.plotly_chart(pfig7, use_container_width=True)

    with col2:
        sec("📊 Daily Revenue vs Tables")
        pfig8 = px.line(cap_df, x="Tables", y="Daily_Rev",
                        markers=True,
                        title="Est. Daily Revenue vs Table Count",
                        color_discrete_sequence=[CLR["success"]])
        pfig8.add_vline(x=S_use, line_dash="dot",
                        line_color=CLR["grey"],
                        annotation_text=f"Current S={S_use}")
        pfig8.update_layout(height=380,
                             yaxis_title="Daily Revenue ($)")
        st.plotly_chart(pfig8, use_container_width=True)

    st.markdown("---")
    st.dataframe(cap_df.style.background_gradient(
        subset=["Block_%"], cmap="RdYlGn_r"
    ).background_gradient(
        subset=["Daily_Rev"], cmap="Greens"
    ).format({"ρ":"{:.4f}","Block_%":"{:.2f}%",
              "Daily_Rev":"${:,.0f}"}),
        use_container_width=True)

    s_10 = cap_df[cap_df["Block_%"] <= 10]["Tables"].min()
    s_5  = cap_df[cap_df["Block_%"] <= 5]["Tables"].min()
    if pd.notna(s_10):
        insight(f"Need {int(s_10)} tables to achieve <10% blocking.")
    if pd.notna(s_5):
        insight(f"Need {int(s_5)} tables to achieve <5% blocking.")
    if pd.notna(s_10) and int(s_10) > S_use:
        warn(f"Current {S_use} tables has {block_pct:.1f}% blocking. "
             f"Add {int(s_10)-S_use} tables to reach 10% target.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — REVENUE ANALYSIS ★ (UNIQUE TO RESTAURANT)
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("💰 Tab 6 — Revenue Analysis ★")
    info("Restaurant's most important metric: Revenue per Available Seat Hour (RevPASH). "
         "This combines occupancy + average spend + turn rate.")

    # RevPASH = Revenue / (Available Seats × Operating Hours)
    rev_pash = served["revenue"].sum() / (S_use * hrs_per_day * df["day"].max()) \
               if "revenue" in served.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Revenue",      f"${df['revenue'].sum():,.0f}")
    c2.metric("Daily Revenue",      f"${df['revenue'].sum()/df['day'].max():,.0f}")
    c3.metric("RevPASH",            f"${rev_pash:.2f}")
    c4.metric("Lost Revenue/Day",   f"${lost_rev_day:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Revenue by Hour")
        if "hour" in served.columns and "revenue" in served.columns:
            rev_hr = served.groupby("hour")["revenue"].sum() / df["day"].max()
            pfig9 = px.bar(x=rev_hr.index, y=rev_hr.values,
                           color=rev_hr.values,
                           color_continuous_scale=["#e8f5e9","#1b5e20"],
                           title="Avg Daily Revenue by Hour ($)",
                           text=rev_hr.values.round(0),
                           labels={"x":"Hour","y":"Revenue ($)"})
            pfig9.update_traces(textposition="outside",
                                texttemplate="$%{text:.0f}")
            pfig9.update_layout(height=380, showlegend=False)
            st.plotly_chart(pfig9, use_container_width=True)

    with col2:
        sec("📊 Revenue: Lunch vs Dinner")
        if "session" in served.columns:
            sess_rev = served.groupby("session").agg(
                Total_Rev=("revenue","sum"),
                Avg_Rev=("revenue","mean"),
                Parties=("revenue","count"),
            ).round(2).reset_index()
            sess_rev["Daily_Rev"] = sess_rev["Total_Rev"] / df["day"].max()
            fig2, axes = plt.subplots(1,2,figsize=(7,4))
            colors_s = [CLR["amber"],CLR["dark"]]
            axes[0].bar(sess_rev["session"], sess_rev["Daily_Rev"],
                        color=colors_s, edgecolor="white")
            axes[0].set_title("Daily Revenue ($)", fontweight="bold")
            axes[0].set_ylabel("Revenue ($)")
            for i,(idx,row) in enumerate(sess_rev.iterrows()):
                axes[0].text(i, row["Daily_Rev"]+50,
                             f"${row['Daily_Rev']:,.0f}", ha="center", fontsize=9)

            axes[1].bar(sess_rev["session"], sess_rev["Avg_Rev"],
                        color=colors_s, edgecolor="white")
            axes[1].set_title("Avg Revenue/Table ($)", fontweight="bold")
            axes[1].set_ylabel("Avg Revenue ($)")
            for i,(idx,row) in enumerate(sess_rev.iterrows()):
                axes[1].text(i, row["Avg_Rev"]+1,
                             f"${row['Avg_Rev']:.0f}", ha="center", fontsize=9)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Revenue Opportunity — Tables vs Annual Revenue")
    ann_rev_by_s = cap_df[["Tables","Daily_Rev"]].copy()
    ann_rev_by_s["Annual_Rev"] = ann_rev_by_s["Daily_Rev"] * days_per_year
    ann_rev_by_s["Lost_Rev"]   = (ann_rev_by_s["Annual_Rev"].max()
                                   - ann_rev_by_s["Annual_Rev"])

    pfig10 = go.Figure()
    pfig10.add_bar(name="Annual Revenue", x=ann_rev_by_s["Tables"],
                   y=ann_rev_by_s["Annual_Rev"],
                   marker_color=CLR["success"])
    pfig10.add_scatter(name="Lost Revenue", x=ann_rev_by_s["Tables"],
                       y=ann_rev_by_s["Lost_Rev"], mode="lines+markers",
                       line=dict(color=CLR["danger"],width=2.5,dash="dot"))
    pfig10.add_vline(x=S_use, line_dash="dot",
                     line_color=CLR["grey"],
                     annotation_text=f"Current S={S_use}")
    pfig10.update_layout(height=380,
                         title="Annual Revenue & Lost Revenue vs Table Count",
                         yaxis_title="Annual Revenue ($)")
    st.plotly_chart(pfig10, use_container_width=True)

    insight(f"RevPASH = ${rev_pash:.2f}/seat/hr. "
            f"Industry benchmark: $8-15 for casual dining.")
    warn(f"Lost revenue from blocked parties: ${lost_rev_ann:,.0f}/year — "
         f"${lost_rev_day:,.0f}/day.")
    if "session" in served.columns:
        dinner_rev_pct = served[served["session"]=="Dinner"]["revenue"].sum() / \
                         served["revenue"].sum() * 100
        insight(f"Dinner generates {dinner_rev_pct:.1f}% of total revenue — "
                f"prioritise dinner-time capacity.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITY ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — Sensitivity Analysis ★")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Blocking % vs Arrival Rate λ")
        lam_r  = np.linspace(lam_use*0.5, lam_use*2.0, 20)
        blk_l  = []
        for l in lam_r:
            m = mmsk_metrics(l, mu_use, S_use, K=S_use)
            blk_l.append(m["P_block"]*100 if m else 100)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(lam_r, blk_l, color=CLR["danger"], lw=2.5,
                marker="o", ms=4)
        ax.axvline(lam_use, color=CLR["dark"], lw=2, ls="--",
                   label=f"Current λ={lam_use}")
        ax.axhline(10, color=CLR["warning"], lw=1.5, ls=":",
                   label="10% target")
        ax.fill_between(lam_r, blk_l, 10,
                        where=[b>10 for b in blk_l],
                        alpha=0.1, color=CLR["danger"])
        ax.set_xlabel("Arrival Rate λ (parties/hr)")
        ax.set_ylabel("Blocking %")
        ax.set_title("Blocking vs Arrival Volume", fontweight="bold")
        ax.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Blocking % vs Turn Rate μ")
        mu_r   = np.linspace(mu_use*0.4, mu_use*2.0, 20)
        blk_m  = []
        for m_val in mu_r:
            m = mmsk_metrics(lam_use, m_val, S_use, K=S_use)
            blk_m.append(m["P_block"]*100 if m else 100)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot([60/m for m in mu_r], blk_m,
                 color=CLR["success"], lw=2.5, marker="s", ms=4)
        ax2.axvline(60/mu_use, color=CLR["dark"], lw=2, ls="--",
                    label=f"Current={60/mu_use:.0f}min")
        ax2.axhline(10, color=CLR["warning"], lw=1.5, ls=":",
                    label="10% target")
        ax2.set_xlabel("Avg Meal Duration (min)")
        ax2.set_ylabel("Blocking %")
        ax2.set_title("Blocking vs Meal Duration", fontweight="bold")
        ax2.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Heatmap: Blocking % vs λ × Tables")
    lam_vals = np.linspace(lam_use*0.6, lam_use*1.5, 7)
    s_vals   = range(max(5,S_use-5), S_use+8)
    heat_data = {}
    for s in s_vals:
        col_data = []
        for l in lam_vals:
            m = mmsk_metrics(l, mu_use, s, K=s)
            val = round(m["P_block"]*100,1) if m else 100.0
            col_data.append(val)
        heat_data[f"S={s}"] = col_data
    heat_df = pd.DataFrame(heat_data,
                            index=[f"λ={l:.0f}" for l in lam_vals])
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax3, linewidths=0.5, annot_kws={"size":9})
    ax3.set_title("Blocking % — Arrival Rate × Table Count",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Shorter meal durations (faster μ) reduce blocking "
            "more effectively than adding tables at high load.")
    insight("Blocking is non-linear — doubling tables doesn't halve blocking.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Business KPIs ★")

    table_cost_day  = S_use * 50   # $50/table/day (rough operating cost)
    table_cost_ann  = table_cost_day * days_per_year
    profit_ann      = annual_rev - table_cost_ann
    rev_per_table   = annual_rev / S_use

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Revenue",     f"${annual_rev:,.0f}")
    c2.metric("Lost Revenue/Year",  f"${lost_rev_ann:,.0f}")
    c3.metric("Revenue per Table",  f"${rev_per_table:,.0f}")
    c4.metric("Table Turns/Day",    f"{throughput*hrs_per_day/S_use:.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Revenue vs Tables — ROI Analysis")
        roi_res = []
        for s in range(max(5,S_use-5), S_use+15):
            m = mmsk_metrics(lam_use, mu_use, s, K=s)
            if m:
                rev  = m["lam_eff"] * hrs_per_day * avg_party * avg_spend * days_per_year
                cost = s * 50 * days_per_year
                roi_res.append({
                    "Tables":s, "Revenue":rev, "Cost":cost,
                    "Profit":rev-cost,
                    "Blocking":round(m["P_block"]*100,1)
                })
        roi_df  = pd.DataFrame(roi_res)
        best_s  = roi_df.loc[roi_df["Profit"].idxmax()]

        pfig11 = go.Figure()
        pfig11.add_bar(name="Annual Revenue", x=roi_df["Tables"],
                       y=roi_df["Revenue"], marker_color=CLR["success"],
                       opacity=0.8)
        pfig11.add_scatter(name="Net Profit", x=roi_df["Tables"],
                           y=roi_df["Profit"], mode="lines+markers",
                           line=dict(color=CLR["dark"],width=3))
        pfig11.add_vline(x=best_s["Tables"], line_dash="dash",
                         line_color=CLR["amber"],
                         annotation_text=f"Optimal S={int(best_s['Tables'])}")
        pfig11.add_vline(x=S_use, line_dash="dot",
                         line_color=CLR["grey"],
                         annotation_text=f"Current S={S_use}")
        pfig11.update_layout(height=400,
                              title="Revenue & Profit vs Table Count",
                              yaxis_title="Annual ($)")
        st.plotly_chart(pfig11, use_container_width=True)

    with col2:
        sec("📊 KPI Summary Table")
        biz_df = pd.DataFrame({
            "KPI": ["Daily parties served","Annual parties",
                    "Avg revenue/party","Est. annual revenue",
                    "Lost revenue/year","Revenue per table/year",
                    "Table turns/day","RevPASH",
                    "Optimal tables","Current blocking"],
            "Value": [f"{throughput*hrs_per_day:.0f}",
                      f"{throughput*hrs_per_day*days_per_year:,.0f}",
                      f"${avg_party*avg_spend:.2f}",
                      f"${annual_rev:,.0f}",
                      f"${lost_rev_ann:,.0f}",
                      f"${rev_per_table:,.0f}",
                      f"{throughput*hrs_per_day/S_use:.1f}",
                      f"${rev_pash:.2f}/seat/hr",
                      f"{int(best_s['Tables'])} tables",
                      f"{block_pct:.1f}%"],
        })
        st.dataframe(biz_df, use_container_width=True, hide_index=True)

    if int(best_s["Tables"]) != S_use:
        rev_gain = best_s["Profit"] - (annual_rev - table_cost_ann)
        insight(f"Optimal {int(best_s['Tables'])} tables improves profit by "
                f"${rev_gain:,.0f}/year vs current {S_use} tables.")
    insight(f"Lost revenue from blocking: ${lost_rev_ann:,.0f}/year = "
            f"${lost_rev_ann/annual_rev*100:.1f}% of current revenue.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Findings & Recommendations")
    st.markdown(f"### 🍽️ Restaurant Queue Analytics — Summary Report")
    st.markdown(f"**λ={lam_use}/hr · μ={mu_use}/hr · S={S_use} tables · "
                f"ρ={metrics['rho']*100:.1f}% · M/M/{S_use}/{S_use} Model**")
    st.markdown("---")

    sec("1️⃣ Current Performance")
    insight(f"Table utilization: {rho_pct:.1f}% — "
            f"{'✅ Good' if rho_pct < 85 else '⚠️ Very high'}")
    insight(f"Blocking: {block_pct:.1f}% turned away — "
            f"${lost_rev_ann:,.0f}/year in lost revenue")
    insight(f"Avg table turn: {metrics['Ws']*60:.0f} min — "
            f"{throughput*hrs_per_day/S_use:.1f} turns/table/day")

    sec("2️⃣ Revenue Recommendations")
    recs = [
        ("🪑 Add Tables (Priority)",
         f"Adding {max(2,int(best_s['Tables'])-S_use)} tables reduces blocking "
         f"from {block_pct:.1f}% to {roi_df[roi_df['Tables']==int(best_s['Tables'])]['Blocking'].values[0]:.1f}%. "
         f"Recovers ${lost_rev_ann*0.6:,.0f}/year in currently lost revenue."),
        ("⏱ Reduce Meal Duration",
         f"Reducing avg meal time from {metrics['Ws']*60:.0f}min to "
         f"{metrics['Ws']*60*0.9:.0f}min (10% reduction via efficient service) "
         f"increases daily covers by {throughput*hrs_per_day*0.1:.0f} parties."),
        ("📅 Reservation Management",
         f"Current no-show rate: {no_show_rate*100:.0f}%. "
         "Implement deposit system or SMS confirmation. "
         "Recovering 50% of no-shows = significant extra covers."),
        ("🌙 Dinner Focus",
         f"Dinner generates higher RevPASH. "
         "Add 1 early seating at 6PM to capture extra covers before peak. "
         "Consider prix-fixe menus to control meal duration."),
        ("📊 Dynamic Pricing",
         "Offer early-bird discounts (5-6PM) to smooth arrival curve. "
         "Reduce peak blocking without adding tables."),
        ("📱 Waitlist App",
         "For blocked parties: offer digital waitlist with SMS callback. "
         "Recover 30-40% of turned-away parties who would otherwise leave."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""RESTAURANT QUEUE ANALYTICS — M/M/S/K REPORT
M3 · M.Sc. Queueing Engine · Restaurant Service Template

MODEL: M/M/{S_use}/{S_use} (Finite Capacity)
  λ (arrival rate)   : {lam_use} parties/hour
  μ (turn rate)      : {mu_use} tables/hour ({60/mu_use:.0f} min avg meal)
  S (tables)         : {S_use}
  K (capacity)       : {S_use} (no waiting — seat or turn away)
  ρ (utilization)    : {metrics['rho']*100:.2f}%
  P_block            : {block_pct:.2f}% turned away

PERFORMANCE:
  Avg meal time      : {metrics['Ws']*60:.1f} min
  Table turns/day    : {throughput*hrs_per_day/S_use:.1f}
  Daily revenue      : ${daily_rev:,.0f}
  Annual revenue     : ${annual_rev:,.0f}
  Lost revenue/year  : ${lost_rev_ann:,.0f}
  RevPASH            : ${rev_pash:.2f}/seat/hr

RECOMMENDATIONS:
  1. Add {max(2,int(best_s['Tables'])-S_use)} tables → optimal = {int(best_s['Tables'])}
  2. Reduce meal time by 10% via efficient service
  3. Deposit system to reduce {no_show_rate*100:.0f}% no-show rate
  4. Add 6PM early seating to capture extra dinner covers
  5. Digital waitlist for turned-away parties
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="Restaurant_Queue_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Capacity Plan (.csv)",
                           cap_df.to_csv(index=False),
                           file_name="Restaurant_Capacity_Plan_M3.csv",
                           mime="text/csv", use_container_width=True)
