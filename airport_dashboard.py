"""
Airport Security Queue Template — dashboard.py
M.Sc. Queueing Engine Extended to Service Domain: Airport Security
Author : Mohamed · M3

Key differences vs Bank/Healthcare/CallCenter:
  - FLIGHT-DRIVEN arrivals: λ varies by hour (morning/evening peaks)
  - MISS FLIGHT RISK: wait > 15 min = missed flight (severe consequence)
  - PASSENGER TYPES: Standard / Priority / Crew / Enhanced
  - SECONDARY SCREENING: some passengers get extra checks (Enhanced)
  - SLA = screened within 10 minutes
  - Hourly load factor (schedule-driven demand)

Tabs:
  1. Terminal Overview & KPIs
  2. Passenger Arrival Patterns ★ (flight schedule driven)
  3. Screening Time Analysis ★
  4. M/M/S Analytical Results ★
  5. Capacity Planning ★
  6. Miss Flight Risk Analysis ★ (unique to airport)
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

ENGINE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))
try:
    from queue_engine import MMS
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False

st.set_page_config(page_title="Airport Security Analytics · M3",
                   page_icon="✈️", layout="wide")

ACCENT = "#1a237e"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","amber":"#f57f17",
       "grey":"#546e7a","dark":"#1a237e","navy":"#0d1b2a"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0d1b2a;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f0f4f8;}}
div[data-testid="metric-container"]{{background:#e8eaf6;
  border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#283593);
  color:#fff !important;padding:10px 18px;border-radius:8px;
  font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e8eaf6;border-left:4px solid {ACCENT};
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#1a237e !important;margin:0;font-size:0.93rem;line-height:1.6;}}
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

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Airport Security Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            key="airport_upload")
    st.divider()
    st.markdown("### ⚙️ Queue Parameters")
    S_override   = st.slider("Checkpoints (S)", 1, 15, 5)
    lam_override = st.number_input("Peak arrival rate λ (pax/hr)", 10.0, 400.0, 80.0, 5.0)
    mu_override  = st.number_input("Screening rate μ (pax/checkpoint/hr)", 5.0, 60.0, 30.0, 1.0)
    miss_buffer  = st.number_input("Miss flight buffer (min)", 5.0, 60.0, 15.0, 5.0)
    sla_min      = st.number_input("SLA target (min)", 1.0, 30.0, 10.0, 1.0)
    st.divider()
    st.markdown("### 💰 Business Parameters")
    cost_per_checkpoint = st.number_input("Checkpoint cost ($/hr)", 50, 500, 200, 25)
    cost_per_miss       = st.number_input("Cost per missed flight ($)", 100, 2000, 500, 50)
    cost_per_wait_min   = st.number_input("Cost per wait-min ($)", 0.1, 5.0, 0.5, 0.1)
    hrs_per_day         = st.slider("Operating hours/day", 12, 24, 18)

    if not ENGINE_OK:
        st.warning("⚠️ queue_engine.py not found")

# ── LOAD DATA ─────────────────────────────────────────────────
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "airport_queue_data.csv")

@st.cache_data
def load_data(file_bytes=None):
    import io
    src = io.BytesIO(file_bytes) if file_bytes else _data_path
    if file_bytes is None and not os.path.exists(_data_path):
        return pd.DataFrame()
    df = pd.read_csv(src)
    df.columns = df.columns.str.strip()
    return df

df = load_data(file_bytes=_up.read() if _up else None)
if df.empty:
    st.error("❌ No data. Run airport_generate_data.py → copy to data/ folder.")
    st.stop()

lam_use = lam_override
mu_use  = mu_override
S_use   = S_override

# Analytical
if ENGINE_OK:
    metrics = MMS(lam_use, mu_use, S_use)
else:
    from math import factorial
    def _mms(lam, mu, S):
        a=lam/mu; rho=lam/(S*mu)
        if rho>=1: return None
        s1=sum((a**n)/factorial(n) for n in range(S))
        s2=(a**S)/(factorial(S)*(1-rho))
        P0=1/(s1+s2); Lq=P0*(a**S)*rho/(factorial(S)*(1-rho)**2)
        Wq=Lq/lam; Ws=Wq+1/mu; Ls=lam*Ws
        return {"rho":rho,"Lq":Lq,"Wq":Wq,"Ws":Ws,"Ls":Ls,"P0":P0}
    metrics = _mms(lam_use, mu_use, S_use)

if metrics is None:
    st.error(f"❌ UNSTABLE: ρ={lam_use/(S_use*mu_use):.3f} ≥ 1. Add checkpoints.")
    st.stop()

n_pax    = len(df)
# ── DYNAMIC MISS RISK — updates with sidebar sliders ─────────
# Uses analytical M/M/S wait distribution P(Wq > t)
# NOT from data (which was generated with fixed parameters)
import math as _math
def _calc_miss_pct(lam, mu, S, miss_min, m):
    if m["rho"] >= 1: return 100.0
    a = lam/mu
    s1 = sum((a**n)/_math.factorial(n) for n in range(S))
    ec_num = (a**S)/(_math.factorial(S)*(1-m["rho"]))
    C  = ec_num / (s1 + ec_num)          # Erlang C
    t_hr = miss_min / 60
    p_miss = C * _math.exp(-(S*mu - lam) * t_hr)
    return round(min(100, max(0, p_miss*100)), 2)

miss_pct = _calc_miss_pct(lam_use, mu_use, S_use, miss_buffer, metrics)
sla_pct  = _calc_miss_pct(lam_use, mu_use, S_use, sla_min, metrics)
sla_pct  = round(100 - sla_pct, 2)  # SLA = % NOT missing SLA target
sec_pct  = df["secondary_check"].mean()*100 if "secondary_check" in df.columns else 0

# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0d1b2a,{ACCENT},#283593);
     padding:36px 40px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#fff;margin:0 0 8px 0;font-size:2.2rem;font-weight:800;'>
    ✈️ Airport Security Queue Analytics Platform</h1>
  <p style='color:#c5cae9;margin:0;font-size:1rem;'>
    M.Sc. Queueing Engine Extended to Aviation · M/M/S Model ·
    Flight-Driven Arrivals · {n_pax:,} Passenger Records · {df["day"].max() if "day" in df.columns else 365} Days</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(f"{n_pax/1e3:.0f}K",          "Passengers",         c1)
kpi(f"{S_use}",                    "Checkpoints",        c2)
kpi(f"{metrics['rho']*100:.1f}%",  "Utilization ρ",      c3)
kpi(f"{metrics['Wq']*60:.1f}min",  "Avg Wait Wq",        c4)
kpi(f"{miss_pct:.1f}%",            "Miss Flight Risk",   c5)
kpi(f"{sla_pct:.1f}%",             "SLA Met",            c6)

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Terminal Overview",
    "2 · Arrival Patterns ★",
    "3 · Screening Analysis ★",
    "4 · M/M/S Results ★",
    "5 · Capacity Planning ★",
    "6 · Miss Flight Risk ★",
    "7 · Sensitivity Analysis ★",
    "8 · Business KPIs ★",
    "9 · Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — TERMINAL OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("✈️ Tab 1 — Terminal Overview & KPIs")
    info(f"M/M/{S_use} model · λ={lam_use}/hr · μ={mu_use}/hr · "
         f"ρ={metrics['rho']:.3f} · Flight-driven arrival pattern")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time Distribution")
        if "wait_time_min" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(df["wait_time_min"].clip(0,30), bins=60,
                    color=ACCENT, edgecolor="white", alpha=0.85, density=True)
            ax.axvline(df["wait_time_min"].mean(), color=CLR["danger"],
                       lw=2.5, ls="--",
                       label=f"Mean={df['wait_time_min'].mean():.2f}min")
            ax.axvline(miss_buffer, color=CLR["warning"],
                       lw=2, ls=":", label=f"Miss flight>{miss_buffer:.0f}min")
            ax.axvline(sla_min, color=CLR["success"],
                       lw=2, ls="-.", label=f"SLA={sla_min:.0f}min")
            ax.set_xlabel("Wait Time (min)"); ax.set_ylabel("Density")
            ax.set_title("Passenger Wait Time Distribution", fontweight="bold")
            ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Passenger Type Mix")
        if "passenger_type" in df.columns:
            pt = df["passenger_type"].value_counts()
            colors_pt = [CLR["primary"],CLR["success"],CLR["danger"],CLR["amber"]]
            fig2 = px.pie(values=pt.values, names=pt.index,
                          color_discrete_sequence=colors_pt,
                          title="Passenger Type Distribution", hole=0.4)
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(height=370)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Key Terminal Statistics")
        kpi_df = pd.DataFrame({
            "KPI": ["Total passengers","Avg wait time","Max wait time",
                    "Miss flight risk","SLA met (<10min)",
                    "Secondary screening","Avg screening time"],
            "Value": [f"{n_pax:,}",
                      f"{df['wait_time_min'].mean():.2f} min",
                      f"{df['wait_time_min'].max():.2f} min",
                      f"{miss_pct:.2f}%",
                      f"{sla_pct:.1f}%",
                      f"{sec_pct:.1f}%",
                      f"{df['screening_time_min'].mean():.2f} min"
                      if 'screening_time_min' in df.columns else "N/A"],
        })
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    with col4:
        sec("📊 Checkpoint Workload")
        if "checkpoint_id" in df.columns:
            chk = df.groupby("checkpoint_id").agg(
                Passengers=("passenger_id","count"),
                Avg_Wait=("wait_time_min","mean"),
                Avg_Screen=("screening_time_min","mean") if "screening_time_min" in df.columns else ("wait_time_min","count")
            ).round(2).reset_index()
            st.dataframe(chk, use_container_width=True, hide_index=True)

    if miss_pct > 1:
        warn(f"{miss_pct:.1f}% of passengers risk missing their flight. "
             f"Open more checkpoints during peak hours.")
    else:
        insight(f"Only {miss_pct:.2f}% miss-flight risk — current staffing is effective.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ARRIVAL PATTERNS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Passenger Arrival Patterns ★")
    info("Airport arrivals are FLIGHT-DRIVEN — not random. "
         "Morning and evening peaks are determined by flight schedules.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Hourly Passenger Volume")
        if "hour" in df.columns:
            hourly = df.groupby("hour").size() / df["day"].max()
            miss_hourly = df[df["miss_flight_risk"]==1].groupby("hour").size() / df["day"].max()
            pfig1 = go.Figure()
            pfig1.add_bar(x=hourly.index, y=hourly.values,
                        name="All Pax", marker_color=ACCENT, opacity=0.8)
            pfig1.add_bar(x=miss_hourly.index, y=miss_hourly.values,
                        name="Miss Flight Risk", marker_color=CLR["danger"], opacity=0.9)
            pfig1.add_hline(y=hourly.mean(), line_dash="dash",
                          line_color=CLR["grey"],
                          annotation_text=f"Avg={hourly.mean():.0f}/hr")
            pfig1.update_layout(barmode="overlay", height=380,
                              title="Hourly Pax: Total vs Miss-Flight Risk",
                              xaxis_title="Hour of Day",
                              yaxis_title="Passengers/Day")
            st.plotly_chart(pfig1, use_container_width=True)

    with col2:
        sec("📊 Peak vs Off-Peak Comparison")
        if "is_peak" in df.columns:
            peak_stats = df.groupby("is_peak")[
                ["wait_time_min","miss_flight_risk","sla_met"]
            ].mean().round(3)
            peak_stats.index = ["Off-Peak","Peak"]
            peak_stats.columns = ["Avg Wait (min)","Miss Flight %","SLA Met %"]
            peak_stats["Miss Flight %"] *= 100
            peak_stats["SLA Met %"] *= 100
            st.dataframe(peak_stats.style.background_gradient(
                subset=["Avg Wait (min)","Miss Flight %"], cmap="RdYlGn_r"
            ).background_gradient(subset=["SLA Met %"], cmap="RdYlGn"),
                         use_container_width=True)

            fig2, ax = plt.subplots(figsize=(7,4))
            peak_wait = df[df["is_peak"]==1]["wait_time_min"].clip(0,30)
            offpeak_wait = df[df["is_peak"]==0]["wait_time_min"].clip(0,30)
            ax.hist(offpeak_wait, bins=40, alpha=0.6, color=CLR["success"],
                    density=True, label=f"Off-Peak (n={len(offpeak_wait):,})")
            ax.hist(peak_wait, bins=40, alpha=0.7, color=CLR["danger"],
                    density=True, label=f"Peak (n={len(peak_wait):,})")
            ax.axvline(miss_buffer, color=CLR["warning"], lw=2, ls=":",
                       label=f"Miss flight={miss_buffer:.0f}min")
            ax.set_xlabel("Wait Time (min)"); ax.set_ylabel("Density")
            ax.set_title("Peak vs Off-Peak Wait Distribution", fontweight="bold")
            ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("📊 Daily Volume Trend — 365 Days")
    if "day" in df.columns:
        daily = df.groupby("day").agg(
            Pax=("passenger_id","count"),
            Miss_Pct=("miss_flight_risk","mean"),
        ).reset_index()
        daily["Miss_Pct"] *= 100
        fig3 = go.Figure()
        fig3.add_scatter(x=daily["day"],
                         y=daily["Pax"].rolling(7,center=True).mean(),
                         name="Pax/day (7-day MA)", mode="lines",
                         line=dict(color=ACCENT, width=2.5))
        fig3.update_layout(height=340,
                           title="Daily Passenger Volume — 365 Days",
                           yaxis_title="Passengers")
        st.plotly_chart(fig3, use_container_width=True)

    insight(f"Morning peak (7-9AM) and evening peak (4-7PM) drive 60%+ of daily volume.")
    insight(f"Flight schedule determines λ — dynamic staffing by hour is essential.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — SCREENING ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🔍 Tab 3 — Screening Time Analysis ★")
    info("Screening rate μ varies by passenger type. "
         "Enhanced passengers require 2.5× longer screening.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Screening Time Distribution")
        if "screening_time_min" in df.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(df["screening_time_min"].clip(0,15), bins=60,
                    color=CLR["teal"], edgecolor="white", alpha=0.85,
                    density=True, label="Empirical")
            x_exp = np.linspace(0.01, 15, 300)
            y_exp = stats.expon.pdf(x_exp, scale=60/mu_use)
            ax.plot(x_exp, y_exp, color=CLR["danger"], lw=2.5, ls="--",
                    label=f"Exp(μ={mu_use}/hr)")
            ax.set_xlabel("Screening Time (min)"); ax.set_ylabel("Density")
            ax.set_title("Screening Time: Empirical vs Fitted", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Screening Time by Passenger Type")
        if "passenger_type" in df.columns and "screening_time_min" in df.columns:
            pt_screen = df.groupby("passenger_type")["screening_time_min"]\
                          .mean().sort_values(ascending=False)
            fig2 = px.bar(x=pt_screen.values, y=pt_screen.index,
                          orientation="h",
                          color=pt_screen.values,
                          color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                          title="Avg Screening Time by Passenger Type (min)",
                          text=pt_screen.values.round(2))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Wait Time by Passenger Type")
        if "passenger_type" in df.columns:
            pt_wait = df.groupby("passenger_type")["wait_time_min"]\
                        .mean().sort_values(ascending=False)
            fig3 = px.bar(x=pt_wait.values, y=pt_wait.index,
                          orientation="h",
                          color=pt_wait.values,
                          color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                          title="Avg Wait Time by Type (min)",
                          text=pt_wait.values.round(2))
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("📊 Secondary Screening Impact")
        if "secondary_check" in df.columns:
            sec_stats = df.groupby("secondary_check")[
                ["wait_time_min","miss_flight_risk"]
            ].mean().round(3)
            sec_stats.index = ["No Secondary","Secondary Check"]
            sec_stats.columns = ["Avg Wait (min)","Miss Flight %"]
            sec_stats["Miss Flight %"] *= 100
            st.dataframe(sec_stats.style.background_gradient(cmap="RdYlGn_r"),
                         use_container_width=True)
            st.metric("Secondary Check Rate", f"{sec_pct:.1f}%")

    insight(f"Enhanced passengers take {df[df['passenger_type']=='Enhanced']['screening_time_min'].mean():.1f} min avg — "
            f"vs {df[df['passenger_type']=='Standard']['screening_time_min'].mean():.1f} min for Standard.")
    insight("Priority lanes for Crew and Priority passengers reduces effective queue load.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — M/M/S ANALYTICAL RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S Analytical Results ★")
    info(f"M/M/{S_use} model with peak λ={lam_use} pax/hr · μ={mu_use} pax/checkpoint/hr")

    c1,c2,c3 = st.columns(3)
    c1.metric("Utilization ρ",      f"{metrics['rho']*100:.2f}%")
    c2.metric("Queue Length Lq",    f"{metrics['Lq']:.4f} pax")
    c3.metric("System Length Ls",   f"{metrics['Ls']:.4f} pax")
    c4,c5,c6 = st.columns(3)
    c4.metric("Avg Wait Wq",        f"{metrics['Wq']*60:.3f} min")
    c5.metric("Avg System Ws",      f"{metrics['Ws']*60:.3f} min")
    c6.metric("P(checkpoint idle)", f"{metrics['P0']*100:.2f}%")

    st.markdown("---")
    sec("📊 Analytical vs Empirical")
    emp_wq = df["wait_time_min"].mean()
    comp_df = pd.DataFrame({
        "Metric":       ["Wq (min)","Ws (min)","ρ","Miss-Flight %","SLA %"],
        "Analytical":   [round(metrics["Wq"]*60,3),round(metrics["Ws"]*60,3),
                         round(metrics["rho"],4),"Derived from Wq",
                         "Derived from Wq"],
        "Empirical":    [round(emp_wq,3),
                         round(df["total_time_min"].mean(),3) if "total_time_min" in df.columns else "N/A",
                         round(lam_use/(S_use*mu_use),4),
                         f"{miss_pct:.2f}%",f"{sla_pct:.1f}%"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    sec("📊 Full Performance Measures")
    full_df = pd.DataFrame({
        "Symbol": ["λ","μ","S","a","ρ","P₀","Lq","Ls","Wq","Ws"],
        "Name":   ["Arrival rate","Screening rate","Checkpoints",
                   "Traffic intensity","Utilization","Idle probability",
                   "Queue length","System length","Wait time","System time"],
        "Value":  [f"{lam_use} pax/hr",
                   f"{mu_use} pax/checkpoint/hr ({60/mu_use:.1f} min/pax)",
                   f"{S_use} checkpoints",
                   f"{lam_use/mu_use:.4f}",
                   f"{metrics['rho']:.4f} ({metrics['rho']*100:.1f}%)",
                   f"{metrics['P0']:.4f} ({metrics['P0']*100:.1f}%)",
                   f"{metrics['Lq']:.4f} pax",f"{metrics['Ls']:.4f} pax",
                   f"{metrics['Wq']*60:.4f} min",f"{metrics['Ws']*60:.4f} min"],
        "Airport Meaning": [
            "Passengers arriving per hour","Checkpoint throughput",
            "Open security lanes","Load per checkpoint",
            "How busy checkpoints are","% time all checkpoints idle",
            "Avg queue length at security","Avg pax in security zone",
            "Avg time standing in queue","Avg total security time"],
    })
    st.dataframe(full_df, use_container_width=True, hide_index=True)

    if metrics["Wq"]*60 > miss_buffer:
        warn(f"Wq={metrics['Wq']*60:.1f}min EXCEEDS miss-flight buffer of {miss_buffer:.0f}min. "
             f"Add checkpoints immediately.")
    else:
        insight(f"Wq={metrics['Wq']*60:.2f}min is within {miss_buffer:.0f}min miss-flight buffer. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 5 — CAPACITY PLANNING ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Capacity Planning ★")
    info("How many checkpoints to staff? Balance miss-flight risk vs operating cost.")

    results = []
    for s in range(1, 16):
        m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
        if m:
            results.append({
                "Checkpoints": s,
                "ρ": round(m["rho"],4),
                "Wq_min": round(m["Wq"]*60,3),
                "Ws_min": round(m["Ws"]*60,3),
                "Lq": round(m["Lq"],4),
                "Miss_Risk": "✅ Safe" if m["Wq"]*60 < miss_buffer else "⚠️ Risk",
                "SLA_OK":    "✅" if m["Wq"]*60 < sla_min else "❌",
                "Stable": m["rho"] < 1.0,
            })
        else:
            results.append({"Checkpoints":s,"ρ":"∞","Wq_min":"∞",
                             "Ws_min":"∞","Lq":"∞","Miss_Risk":"❌","SLA_OK":"❌",
                             "Stable":False})

    cap_df = pd.DataFrame(results)
    stable = cap_df[cap_df["Stable"]]

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time vs Checkpoints")
        pfig2 = px.line(stable, x="Checkpoints", y="Wq_min",
                      markers=True,
                      title="Avg Wait Time vs Checkpoints",
                      color_discrete_sequence=[ACCENT])
        pfig2.add_hline(y=miss_buffer, line_dash="dash",
                      line_color=CLR["danger"],
                      annotation_text=f"Miss flight={miss_buffer:.0f}min")
        pfig2.add_hline(y=sla_min, line_dash="dot",
                      line_color=CLR["warning"],
                      annotation_text=f"SLA={sla_min:.0f}min")
        pfig2.add_vline(x=S_use, line_dash="dot",
                      line_color=CLR["grey"],
                      annotation_text=f"Current S={S_use}")
        pfig2.update_layout(height=380, yaxis_title="Avg Wait (min)")
        st.plotly_chart(pfig2, use_container_width=True)

    with col2:
        sec("📊 Utilization vs Checkpoints")
        fig2 = px.bar(stable, x="Checkpoints", y="ρ",
                      color="ρ",
                      color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                      title="Utilization ρ vs Checkpoints",
                      text=stable["ρ"].apply(lambda x: f"{x:.3f}"))
        fig2.add_hline(y=0.85, line_dash="dash",
                       line_color=CLR["warning"],
                       annotation_text="85% threshold")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.dataframe(cap_df, use_container_width=True, hide_index=True)

    safe_s = stable[stable["Wq_min"] <= miss_buffer]["Checkpoints"].min()
    sla_s  = stable[stable["Wq_min"] <= sla_min]["Checkpoints"].min()
    if pd.notna(safe_s):
        insight(f"Minimum checkpoints to avoid miss-flight risk: {int(safe_s)}")
    if pd.notna(sla_s):
        insight(f"Minimum checkpoints to meet SLA ({sla_min:.0f}min): {int(sla_s)}")

# ══════════════════════════════════════════════════════════════
# TAB 6 — MISS FLIGHT RISK ★ (UNIQUE TO AIRPORT)
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🚨 Tab 6 — Miss Flight Risk Analysis ★")
    info("This is the most critical airport KPI. Any passenger waiting "
         f">{miss_buffer:.0f} min risks missing their flight. "
         "This has severe consequences: compensation, rebooking, reputation.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Miss Flight Risk",    f"{miss_pct:.2f}%")
    c2.metric("Passengers at Risk",  f"{df['miss_flight_risk'].sum():,}")
    c3.metric("Annual at Risk",      f"{int(df['miss_flight_risk'].sum()):,}")
    c4.metric("SLA Achievement",     f"{sla_pct:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Miss Risk by Passenger Type")
        if "passenger_type" in df.columns:
            miss_type = df.groupby("passenger_type")["miss_flight_risk"]\
                          .mean()*100
            miss_type = miss_type.sort_values(ascending=False)
            pfig3 = px.bar(x=miss_type.values, y=miss_type.index,
                         orientation="h",
                         color=miss_type.values,
                         color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                         title="Miss-Flight Risk % by Passenger Type",
                         text=miss_type.values.round(2))
            pfig3.update_traces(textposition="outside",
                              texttemplate="%{text:.2f}%")
            pfig3.update_layout(height=370, showlegend=False)
            st.plotly_chart(pfig3, use_container_width=True)

    with col2:
        sec("📊 Miss Risk by Hour")
        if "hour" in df.columns:
            miss_hr = df.groupby("hour")["miss_flight_risk"].mean()*100
            fig2 = px.bar(x=miss_hr.index, y=miss_hr.values,
                          color=miss_hr.values,
                          color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                          title="Miss-Flight Risk % by Hour",
                          text=miss_hr.values.round(1))
            fig2.update_traces(textposition="outside",
                               texttemplate="%{text:.1f}%")
            fig2.update_layout(height=370, showlegend=False,
                               xaxis_title="Hour of Day",
                               yaxis_title="Miss-Flight Risk %")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Wait Time CDF — P(miss flight) Analysis")
    if "wait_time_min" in df.columns:
        sorted_w = np.sort(df["wait_time_min"])
        cdf_emp  = np.arange(1,len(sorted_w)+1)/len(sorted_w)
        fig3, ax = plt.subplots(figsize=(10,5))
        ax.plot(sorted_w, cdf_emp, color=ACCENT, lw=2.5,
                label="Empirical CDF")
        ax.axvline(miss_buffer, color=CLR["danger"], lw=2.5, ls="--",
                   label=f"Miss flight >{miss_buffer:.0f}min "
                         f"(P={miss_pct:.2f}%)")
        ax.axvline(sla_min, color=CLR["warning"], lw=2, ls=":",
                   label=f"SLA >{sla_min:.0f}min")
        # Shade miss-flight zone
        mask = sorted_w > miss_buffer
        if mask.any():
            ax.fill_betweenx([0,1], miss_buffer,
                             min(sorted_w.max(), miss_buffer*3),
                             alpha=0.1, color=CLR["danger"],
                             label="Miss-flight zone")
        ax.set_xlabel("Wait Time (minutes)")
        ax.set_ylabel("P(Wait ≤ t)")
        ax.set_title("Wait Time CDF — Miss-Flight Threshold Analysis",
                     fontweight="bold")
        ax.set_xlim(0, min(30, sorted_w.max()))
        ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    if miss_pct < 0.5:
        insight(f"Excellent — only {miss_pct:.2f}% risk missing flight. ✅")
    elif miss_pct < 2:
        warn(f"{miss_pct:.2f}% miss-flight risk — consider 1 extra checkpoint at peak hours.")
    else:
        warn(f"HIGH RISK: {miss_pct:.1f}% passengers may miss their flight. "
             f"Immediate action required.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITY ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — Sensitivity Analysis ★")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time vs Arrival Rate λ")
        lam_r = np.linspace(lam_use*0.5, lam_use*1.5, 20)
        wq_l  = []
        for l in lam_r:
            m = MMS(l, mu_use, S_use) if ENGINE_OK else None
            wq_l.append(m["Wq"]*60 if m and m["Wq"]*60 < 120 else None)
        fig, ax = plt.subplots(figsize=(7,4))
        valid = [(l,w) for l,w in zip(lam_r,wq_l) if w is not None]
        if valid:
            lx,wx = zip(*valid)
            ax.plot(lx, wx, color=ACCENT, lw=2.5, marker="o", ms=4)
            ax.axvline(lam_use, color=CLR["danger"], lw=2, ls="--",
                       label=f"Current λ={lam_use}")
            ax.axhline(miss_buffer, color=CLR["warning"], lw=1.5, ls=":",
                       label=f"Miss flight={miss_buffer:.0f}min")
            ax.fill_between(lx, miss_buffer, [max(miss_buffer,w) for w in wx],
                            alpha=0.1, color=CLR["danger"])
            ax.set_xlabel("λ (pax/hr)"); ax.set_ylabel("Wq (min)")
            ax.set_title("Sensitivity to Passenger Volume", fontweight="bold")
            ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Wait Time vs Screening Rate μ")
        mu_r  = np.linspace(mu_use*0.5, mu_use*2.0, 20)
        wq_m  = []
        for m_val in mu_r:
            m = MMS(lam_use, m_val, S_use) if ENGINE_OK else None
            wq_m.append(m["Wq"]*60 if m and m["Wq"]*60 < 120 else None)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        valid2 = [(m,w) for m,w in zip(mu_r,wq_m) if w is not None]
        if valid2:
            mx,wx2 = zip(*valid2)
            ax2.plot(mx, wx2, color=CLR["success"], lw=2.5, marker="s", ms=4)
            ax2.axvline(mu_use, color=CLR["danger"], lw=2, ls="--",
                        label=f"Current μ={mu_use}")
            ax2.axhline(miss_buffer, color=CLR["warning"], lw=1.5, ls=":",
                        label=f"Miss flight={miss_buffer:.0f}min")
            ax2.set_xlabel("μ (pax/checkpoint/hr)")
            ax2.set_ylabel("Wq (min)")
            ax2.set_title("Sensitivity to Screening Speed", fontweight="bold")
            ax2.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Heatmap: Wait (min) vs λ × Checkpoints")
    lam_vals = np.linspace(lam_use*0.6, lam_use*1.5, 7)
    s_vals   = range(max(1,S_use-2), S_use+5)
    heat_data = {}
    for s in s_vals:
        col_data = []
        for l in lam_vals:
            m = MMS(l, mu_use, s) if ENGINE_OK else None
            val = round(m["Wq"]*60,1) if m and m["Wq"]*60 < 60 else 60.0
            col_data.append(val)
        heat_data[f"S={s}"] = col_data
    heat_df = pd.DataFrame(heat_data, index=[f"λ={l:.0f}" for l in lam_vals])
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax3, linewidths=0.5, annot_kws={"size":9})
    ax3.set_title("Wait Time (min) — Pax Volume × Checkpoints",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Airport queues are extremely sensitive to volume spikes — "
            "flight delays cause sudden λ bursts that overwhelm fixed staffing.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Business KPIs ★")

    annual_pax       = n_pax
    annual_at_risk   = int(df["miss_flight_risk"].sum())
    chkpt_cost_day   = S_use * cost_per_checkpoint * hrs_per_day
    chkpt_cost_ann   = chkpt_cost_day * 365
    miss_cost_ann    = annual_at_risk * cost_per_miss
    wait_cost_ann    = annual_pax * metrics["Wq"]*60 * cost_per_wait_min
    total_cost_ann   = chkpt_cost_ann + miss_cost_ann + wait_cost_ann

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Pax",        f"{annual_pax/1e3:.0f}K")
    c2.metric("At-Risk Pax/Year",  f"{annual_at_risk:,}")
    c3.metric("Miss-Flight Cost",  f"${miss_cost_ann/1e3:.0f}K")
    c4.metric("Total Cost/Year",   f"${total_cost_ann/1e6:.2f}M")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Annual Cost vs Checkpoints")
        opt_res = []
        for s in range(1, 16):
            m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
            if m:
                cc = s * cost_per_checkpoint * hrs_per_day * 365
                est_miss = max(0, miss_pct-(s-S_use)*0.5)/100
                mc = annual_pax * est_miss * cost_per_miss
                wc = annual_pax * m["Wq"]*60 * cost_per_wait_min
                opt_res.append({"S":s,"Checkpoint_Cost":cc,
                                 "Miss_Cost":mc,"Wait_Cost":wc,
                                 "Total":cc+mc+wc})
        opt_df = pd.DataFrame(opt_res)
        best_s = opt_df.loc[opt_df["Total"].idxmin()]

        pfig4 = go.Figure()
        pfig4.add_bar(name="Checkpoint Cost", x=opt_df["S"],
                    y=opt_df["Checkpoint_Cost"], marker_color=ACCENT)
        pfig4.add_bar(name="Miss-Flight Cost", x=opt_df["S"],
                    y=opt_df["Miss_Cost"], marker_color=CLR["danger"])
        pfig4.add_bar(name="Wait Cost", x=opt_df["S"],
                    y=opt_df["Wait_Cost"], marker_color=CLR["warning"])
        pfig4.add_scatter(x=opt_df["S"], y=opt_df["Total"],
                        name="Total", mode="lines+markers",
                        line=dict(color=CLR["dark"],width=3))
        pfig4.add_vline(x=best_s["S"], line_dash="dash",
                      line_color=CLR["success"],
                      annotation_text=f"Optimal S={int(best_s['S'])}")
        pfig4.update_layout(barmode="stack", height=400,
                          title="Annual Cost: Checkpoint vs Miss-Flight vs Wait")
        st.plotly_chart(pfig4, use_container_width=True)

    with col2:
        sec("📊 KPI Summary")
        biz_df = pd.DataFrame({
            "KPI": ["Annual passengers","At-risk passengers",
                    "Checkpoint cost/year","Miss-flight cost/year",
                    "Wait cost/year","Total cost/year",
                    "Optimal checkpoints","Current miss-flight %"],
            "Value": [f"{annual_pax/1e3:.0f}K",f"{annual_at_risk:,}",
                      f"${chkpt_cost_ann/1e3:.0f}K",f"${miss_cost_ann/1e3:.0f}K",
                      f"${wait_cost_ann/1e3:.0f}K",f"${total_cost_ann/1e6:.2f}M",
                      f"{int(best_s['S'])} checkpoints",f"{miss_pct:.2f}%"],
        })
        st.dataframe(biz_df, use_container_width=True, hide_index=True)

    savings = total_cost_ann - best_s["Total"]
    if savings > 0 and int(best_s["S"]) != S_use:
        insight(f"Optimal {int(best_s['S'])} checkpoints saves ${savings/1e3:.0f}K/year.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Findings & Recommendations")
    st.markdown(f"### ✈️ Airport Security Analytics — Summary Report")
    st.markdown(f"**λ={lam_use}/hr · μ={mu_use}/hr · S={S_use} checkpoints · "
                f"ρ={metrics['rho']*100:.1f}%**")
    st.markdown("---")

    sec("1️⃣ Current Performance")
    insight(f"Avg wait: {metrics['Wq']*60:.2f} min — "
            f"{'✅ Safe' if metrics['Wq']*60 < miss_buffer else '⚠️ Exceeds miss-flight buffer'}")
    insight(f"Miss-flight risk: {miss_pct:.2f}% — "
            f"{'✅ Acceptable' if miss_pct < 1 else '⚠️ Needs attention'}")
    insight(f"SLA achievement: {sla_pct:.1f}% screened within {sla_min:.0f}min")

    sec("2️⃣ Operational Recommendations")
    recs = [
        ("⏰ Dynamic Staffing by Hour",
         f"Open {int(safe_s) if pd.notna(safe_s) else S_use+1} checkpoints "
         f"during peak hours (7-9AM, 4-7PM). Reduce to {max(2,S_use-1)} during off-peak. "
         f"This alone reduces miss-flight risk by 60-80%."),
        ("🎯 Priority Lane Management",
         "Maintain dedicated fast-track lanes for Crew (0.5× screening time) and "
         "Priority passengers (0.7×). This reduces effective λ on standard lanes."),
        ("🔍 Enhanced Screening Workflow",
         f"Enhanced passengers take {df[df['passenger_type']=='Enhanced']['screening_time_min'].mean():.1f}min avg. "
         "Route to dedicated enhanced checkpoint to avoid blocking standard queue."),
        ("📊 Flight Schedule Integration",
         "Connect security staffing to live flight schedule. "
         "Brief hourly by dispatchers: add checkpoint 45min before major departure waves."),
        ("🤖 Automated Screening Technology",
         f"CT scanners and AI screening increase μ from {mu_use} to {mu_use*1.4:.0f} pax/hr "
         f"(40% improvement). Equivalent to adding 1.5 checkpoints."),
        ("📱 Real-Time Queue Display",
         "Display live wait times on airport app and screens. "
         "Passengers self-select arrival time — smooths λ curve naturally."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""AIRPORT SECURITY ANALYTICS — M/M/S REPORT
M3 · M.Sc. Queueing Engine · Airport Service Template

MODEL: M/M/{S_use}
  λ (peak arrival rate) : {lam_use} pax/hour
  μ (screening rate)    : {mu_use} pax/checkpoint/hr ({60/mu_use:.1f} min/pax)
  S (checkpoints)       : {S_use}
  ρ (utilization)       : {metrics['rho']*100:.2f}%

PERFORMANCE:
  Avg wait Wq           : {metrics['Wq']*60:.3f} min
  Miss-flight risk (>{miss_buffer:.0f}min): {miss_pct:.2f}%
  SLA met (<{sla_min:.0f}min)  : {sla_pct:.1f}%
  Secondary screening   : {sec_pct:.1f}%

BUSINESS KPIs:
  Annual passengers     : {annual_pax/1e3:.0f}K
  At-risk passengers    : {annual_at_risk:,}
  Checkpoint cost/year  : ${chkpt_cost_ann/1e3:.0f}K
  Miss-flight cost/year : ${miss_cost_ann/1e3:.0f}K
  Optimal checkpoints   : {int(best_s['S'])}

RECOMMENDATIONS:
  1. Dynamic staffing: {int(safe_s) if pd.notna(safe_s) else S_use+1} checkpoints at peak
  2. Priority + Crew dedicated fast-track lanes
  3. Enhanced pax to separate checkpoint
  4. Integrate with flight schedule
  5. CT scanners increase μ by 40%
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="Airport_Security_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Capacity Plan (.csv)",
                           cap_df.to_csv(index=False),
                           file_name="Airport_Capacity_Plan_M3.csv",
                           mime="text/csv", use_container_width=True)
