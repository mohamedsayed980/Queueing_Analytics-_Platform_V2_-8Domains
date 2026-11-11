"""
Call Center Queue Template — dashboard.py
M.Sc. Queueing Engine Extended to Service Domain: Call Center
Author : Mohamed · M3

Key difference from Bank/Healthcare templates:
  - ABANDONMENT: customers hang up if wait too long
  - SLA: % calls answered within 20 seconds
  - Call types: Billing/Technical/Account/General/Complaint
  - After-Call Work (ACW) tracked in handle time
  - Occupancy = % time agents are busy (not same as utilization)

Tabs:
  1. Queue Profile & Call Center KPIs
  2. Call Volume & Arrival Patterns ★
  3. Handle Time Analysis ★
  4. M/M/S Analytical Results ★
  5. Capacity Planning ★
  6. SLA & Abandonment Analysis ★
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

st.set_page_config(page_title="Call Center Analytics · M3",
                   page_icon="📞", layout="wide")

ACCENT = "#e65100"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","amber":"#f57f17",
       "grey":"#546e7a","dark":"#1a237e","orange":"#e65100"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#fff3e0;
  border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#c62828);
  color:#fff !important;padding:10px 18px;border-radius:8px;
  font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e3f2fd;border-left:4px solid #1565c0;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}}
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
    st.markdown("## 📞 Call Center Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            key="cc_upload")
    st.divider()
    st.markdown("### ⚙️ Queue Parameters")
    S_override   = st.slider("Agents (S)", 1, 20, 3)
    lam_override = st.number_input("Arrival rate λ (calls/hr)", 1.0, 200.0, 50.0, 5.0)
    mu_override  = st.number_input("Service rate μ (calls/agent/hr)", 1.0, 60.0, 20.0, 1.0)
    sla_sec      = st.slider("SLA target (seconds)", 10, 120, 20)
    st.divider()
    st.markdown("### 💰 Business Parameters")
    cost_per_agent   = st.number_input("Agent cost ($/hr)", 10, 100, 18, 2)
    cost_per_abandon = st.number_input("Cost per abandoned call ($)", 1, 50, 10, 1)
    cost_per_wait_min= st.number_input("Cost per wait-min ($)", 0.01, 2.0, 0.20, 0.05)
    hrs_per_day      = st.slider("Hours/day", 4, 24, 8)

    if not ENGINE_OK:
        st.warning("⚠️ queue_engine.py not found\nPlace it in same folder.")

# ── LOAD DATA ─────────────────────────────────────────────────
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "callcenter_queue_data.csv")

@st.cache_data
def load_data(file_bytes=None):
    import io
    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    elif os.path.exists(_data_path):
        df = pd.read_csv(_data_path)
    else:
        return pd.DataFrame()
    df.columns = df.columns.str.strip()
    return df

df = load_data(file_bytes=_up.read() if _up else None)

if df.empty:
    st.error("❌ No data. Run callcenter_generate_data.py → copy to data/ folder.")
    st.stop()

lam_use = lam_override
mu_use  = mu_override
S_use   = S_override

# Analytical M/M/S
if ENGINE_OK:
    metrics = MMS(lam_use, mu_use, S_use)
else:
    from math import factorial
    def _mms(lam, mu, S):
        a = lam/mu; rho = lam/(S*mu)
        if rho >= 1: return None
        s1 = sum((a**n)/factorial(n) for n in range(S))
        s2 = (a**S)/(factorial(S)*(1-rho))
        P0 = 1/(s1+s2)
        Lq = P0*(a**S)*rho/(factorial(S)*(1-rho)**2)
        Wq = Lq/lam; Ws = Wq+1/mu; Ls = lam*Ws
        return {"rho":rho,"Lq":Lq,"Wq":Wq,"Ws":Ws,"Ls":Ls,"P0":P0}
    metrics = _mms(lam_use, mu_use, S_use)

if metrics is None:
    st.error(f"❌ System UNSTABLE: ρ={lam_use/(S_use*mu_use):.3f} ≥ 1. Add agents.")
    st.stop()

# Subsets
answered = df[df["abandoned"]==0].copy()
sla_pct  = df["sla_met"].mean()*100 if "sla_met" in df.columns else 0
abn_pct  = df["abandoned"].mean()*100 if "abandoned" in df.columns else 0
n_calls  = len(df)

# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#bf360c,{ACCENT});
     padding:36px 40px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#fff;margin:0 0 8px 0;font-size:2.2rem;font-weight:800;'>
    📞 Call Center Queue Analytics Platform</h1>
  <p style='color:#ffe0b2;margin:0;font-size:1rem;'>
    M.Sc. Queueing Engine Extended to Call Center · M/M/S + Abandonment ·
    {n_calls:,} Call Records · {df["day"].max() if "day" in df.columns else 250} Days</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(f"{n_calls:,}",          "Total Calls",        c1)
kpi(f"{S_use}",              "Agents",             c2)
kpi(f"{metrics['rho']*100:.1f}%","Utilization ρ",  c3)
kpi(f"{sla_pct:.1f}%",       "SLA Met",            c4)
kpi(f"{abn_pct:.1f}%",       "Abandonment",        c5)
kpi(f"{metrics['Wq']*60:.1f}min","Avg Wait Wq",    c6)

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Queue Profile",
    "2 · Call Volume ★",
    "3 · Handle Time ★",
    "4 · M/M/S Results ★",
    "5 · Capacity Planning ★",
    "6 · SLA & Abandonment ★",
    "7 · Sensitivity Analysis ★",
    "8 · Business KPIs ★",
    "9 · Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — QUEUE PROFILE
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Queue Profile & Call Center KPIs")
    info(f"M/M/{S_use} with abandonment · λ={lam_use}/hr · μ={mu_use}/hr · "
         f"ρ={metrics['rho']:.3f} · SLA={sla_sec}s")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time Distribution (Answered Calls)")
        if "wait_time_min" in answered.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(answered["wait_time_min"].clip(0,10), bins=60,
                    color=CLR["orange"], edgecolor="white", alpha=0.85, density=True)
            ax.axvline(answered["wait_time_min"].mean(), color=CLR["danger"],
                       lw=2.5, ls="--",
                       label=f"Mean={answered['wait_time_min'].mean():.2f}min")
            ax.axvline(sla_sec/60, color=CLR["success"],
                       lw=2, ls=":", label=f"SLA={sla_sec}s")
            ax.set_xlabel("Wait Time (minutes)"); ax.set_ylabel("Density")
            ax.set_title("Wait Time — Answered Calls", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Call Type Distribution")
        if "call_type" in df.columns:
            ct = df["call_type"].value_counts()
            fig2 = px.bar(x=ct.values, y=ct.index, orientation="h",
                          color=ct.values,
                          color_continuous_scale=["#fff3e0","#e65100","#c62828"],
                          title="Call Volume by Type",
                          text=ct.values)
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, showlegend=False,
                               yaxis_title="", xaxis_title="Call Count")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Key Call Center Metrics")
        kpi_df = pd.DataFrame({
            "KPI": ["Total calls","Answered","Abandoned","SLA met",
                    "Avg wait (answered)","Avg handle time","Avg total time"],
            "Value": [f"{n_calls:,}",
                      f"{len(answered):,} ({100-abn_pct:.1f}%)",
                      f"{df['abandoned'].sum():,} ({abn_pct:.1f}%)",
                      f"{sla_pct:.1f}%",
                      f"{answered['wait_time_min'].mean():.2f} min",
                      f"{answered['handle_time_min'].mean():.2f} min",
                      f"{answered['total_time_min'].mean():.2f} min"],
        })
        st.dataframe(kpi_df, use_container_width=True)

    with col4:
        sec("📊 Agent Performance Summary")
        if "agent_id" in answered.columns:
            ag = answered.groupby("agent_id").agg(
                Calls=("call_id","count"),
                Avg_Handle=("handle_time_min","mean"),
                Avg_Wait=("wait_time_min","mean"),
            ).round(2).reset_index()
            st.dataframe(ag, use_container_width=True)

    if abn_pct > 10:
        warn(f"Abandonment rate {abn_pct:.1f}% is HIGH. Industry standard < 5%. Add agents.")
    else:
        insight(f"Abandonment rate {abn_pct:.1f}% is within acceptable range.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — CALL VOLUME ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Call Volume & Arrival Patterns ★")
    info("Call center staffing is driven by WHEN calls arrive — hourly patterns are critical.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Hourly Call Volume")
        if "hour" in df.columns:
            hourly_all = df.groupby("hour").size() / df["day"].max()
            hourly_abn = df[df["abandoned"]==1].groupby("hour").size() / df["day"].max()
            fig = go.Figure()
            fig.add_bar(x=hourly_all.index, y=hourly_all.values,
                        name="All Calls", marker_color=CLR["orange"], opacity=0.8)
            fig.add_bar(x=hourly_abn.index, y=hourly_abn.values,
                        name="Abandoned", marker_color=CLR["danger"], opacity=0.9)
            fig.add_hline(y=hourly_all.mean(), line_dash="dash",
                          line_color=CLR["grey"],
                          annotation_text=f"Avg={hourly_all.mean():.1f}")
            fig.update_layout(barmode="overlay", height=380,
                              title="Hourly Calls: Total vs Abandoned",
                              xaxis_title="Hour", yaxis_title="Calls/Day")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 SLA Achievement by Hour")
        if "hour" in df.columns and "sla_met" in df.columns:
            sla_hourly = df.groupby("hour")["sla_met"].mean() * 100
            fig2 = px.bar(x=sla_hourly.index, y=sla_hourly.values,
                          color=sla_hourly.values,
                          color_continuous_scale=["#c62828","#f57f17","#2e7d32"],
                          title=f"SLA % by Hour (target={sla_sec}s)",
                          text=sla_hourly.values.round(1))
            fig2.add_hline(y=80, line_dash="dash", line_color=CLR["danger"],
                           annotation_text="80% SLA target")
            fig2.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
            fig2.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Daily Call Volume Trend")
    if "day" in df.columns:
        daily = df.groupby("day").agg(
            Total=("call_id","count"),
            Abandoned=("abandoned","sum"),
            SLA_Pct=("sla_met","mean")
        ).reset_index()
        daily["SLA_Pct"] *= 100
        fig3 = go.Figure()
        fig3.add_scatter(x=daily["day"], y=daily["Total"],
                         name="Total Calls", mode="lines",
                         line=dict(color=CLR["orange"],width=2))
        fig3.add_scatter(x=daily["day"], y=daily["Abandoned"],
                         name="Abandoned", mode="lines",
                         line=dict(color=CLR["danger"],width=1.5,dash="dot"))
        fig3.update_layout(height=360, title="Daily Call Volume — 250 Days",
                           yaxis_title="Calls", xaxis_title="Day")
        st.plotly_chart(fig3, use_container_width=True)

    if "is_peak" in df.columns:
        peak_abn = df[df["is_peak"]==1]["abandoned"].mean()*100
        offpeak_abn = df[df["is_peak"]==0]["abandoned"].mean()*100
        insight(f"Peak hour abandonment: {peak_abn:.1f}% vs off-peak: {offpeak_abn:.1f}%.")
    insight(f"λ={lam_use}/hr → {lam_use*hrs_per_day:.0f} calls/day expected over {hrs_per_day}hr shift.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — HANDLE TIME ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("⏱ Tab 3 — Call Handle Time Analysis ★")
    info("Average Handle Time (AHT) = Talk Time + After-Call Work. "
         "Drives μ (service rate) directly.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Handle Time Distribution")
        if "handle_time_min" in answered.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            svc = answered["handle_time_min"].clip(0,20)
            ax.hist(svc, bins=60, color=CLR["success"],
                    edgecolor="white", alpha=0.85, density=True, label="Empirical")
            x_exp = np.linspace(0.01, svc.max(), 300)
            y_exp = stats.expon.pdf(x_exp, scale=60/mu_use)
            ax.plot(x_exp, y_exp, color=CLR["danger"], lw=2.5, ls="--",
                    label=f"Exp(μ={mu_use}/hr)")
            ax.set_xlabel("Handle Time (min)"); ax.set_ylabel("Density")
            ax.set_title("AHT: Empirical vs Fitted", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Handle Time by Call Type")
        if "call_type" in answered.columns and "handle_time_min" in answered.columns:
            ct_ht = answered.groupby("call_type")["handle_time_min"]\
                            .mean().sort_values(ascending=False)
            fig2 = px.bar(x=ct_ht.values, y=ct_ht.index, orientation="h",
                          color=ct_ht.values,
                          color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                          title="Avg Handle Time by Call Type (min)",
                          text=ct_ht.values.round(1))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Abandonment Rate by Wait Time Bucket")
    if "wait_time_min" in df.columns:
        df["wait_bucket"] = pd.cut(df["wait_time_min"],
                                    bins=[0,0.5,1,2,3,5,100],
                                    labels=["<30s","30s-1m","1-2m","2-3m","3-5m",">5m"])
        wb = df.groupby("wait_bucket", observed=True)["abandoned"].mean() * 100
        fig3 = px.bar(x=wb.index.astype(str), y=wb.values,
                      color=wb.values,
                      color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                      title="Abandonment Rate % by Wait Time",
                      text=wb.values.round(1))
        fig3.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
        fig3.update_layout(height=380, showlegend=False,
                           xaxis_title="Wait Time", yaxis_title="Abandonment %")
        st.plotly_chart(fig3, use_container_width=True)

    if "handle_time_min" in answered.columns:
        mu_emp = 60 / answered["handle_time_min"].mean()
        insight(f"Empirical AHT={answered['handle_time_min'].mean():.2f} min → "
                f"μ={mu_emp:.1f} calls/agent/hr.")
    insight("Complaint calls have longest AHT — prioritise routing to senior agents.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — M/M/S ANALYTICAL RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S Analytical Results ★")
    info(f"M/M/{S_use} model · λ={lam_use}/hr · μ={mu_use}/hr · "
         f"Note: M/M/S assumes no abandonment — compare with empirical to see abandonment effect.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Utilization ρ",    f"{metrics['rho']*100:.2f}%")
    c2.metric("Queue Length Lq",  f"{metrics['Lq']:.4f} callers")
    c3.metric("System Length Ls", f"{metrics['Ls']:.4f} callers")
    c4,c5,c6 = st.columns(3)
    c4.metric("Avg Wait Wq",      f"{metrics['Wq']*60:.3f} min")
    c5.metric("Avg System Ws",    f"{metrics['Ws']*60:.3f} min")
    c6.metric("P(agent idle) P₀", f"{metrics['P0']*100:.2f}%")

    st.markdown("---")
    sec("📊 Analytical vs Empirical Comparison")
    emp_wq = answered["wait_time_min"].mean() if "wait_time_min" in answered.columns else "N/A"
    comp_df = pd.DataFrame({
        "Metric":     ["Wq (min)","Ws (min)","ρ","Lq","Abandonment","SLA %"],
        "Analytical (M/M/S)": [
            round(metrics["Wq"]*60,3), round(metrics["Ws"]*60,3),
            round(metrics["rho"],4), round(metrics["Lq"],4),
            "0% (M/M/S assumes no abandonment)", "Computed from Wq"],
        "Empirical":  [
            round(emp_wq,3) if isinstance(emp_wq,float) else emp_wq,
            round(answered["total_time_min"].mean(),3) if "total_time_min" in answered.columns else "N/A",
            round(lam_use/(S_use*mu_use),4),
            round(df["queue_at_arrival"].mean(),4) if "queue_at_arrival" in df.columns else "N/A",
            f"{abn_pct:.1f}%",
            f"{sla_pct:.1f}%"],
    })
    st.dataframe(comp_df, use_container_width=True)

    sec("📊 Full Performance Measures")
    full_df = pd.DataFrame({
        "Symbol": ["λ","μ","S","a","ρ","P₀","Lq","Ls","Wq","Ws"],
        "Name":   ["Arrival rate","Handle rate","Agents","Traffic intensity",
                   "Utilization","Idle probability","Queue length","System length",
                   "Queue wait","System time"],
        "Value":  [f"{lam_use} calls/hr",
                   f"{mu_use} calls/agent/hr (AHT={60/mu_use:.1f}min)",
                   f"{S_use}",f"{lam_use/mu_use:.4f}",
                   f"{metrics['rho']:.4f} ({metrics['rho']*100:.1f}%)",
                   f"{metrics['P0']:.4f} ({metrics['P0']*100:.1f}%)",
                   f"{metrics['Lq']:.4f} callers",f"{metrics['Ls']:.4f} callers",
                   f"{metrics['Wq']*60:.4f} min ({metrics['Wq']*3600:.1f}s)",
                   f"{metrics['Ws']*60:.4f} min"],
        "Call Center Meaning": [
            "Inbound call demand","Agent throughput","Staffed agents",
            "Load per agent","Agent occupancy",
            "% time all agents idle","Avg callers on hold",
            "Avg callers in system","Avg time on hold",
            "Avg total call duration"],
    })
    st.dataframe(full_df, use_container_width=True)

    warn("M/M/S formula assumes infinite patience — actual abandonment reduces empirical Lq and Wq.")
    insight(f"Erlang C gives P(wait>0)={1-metrics['P0']*100:.1f}% — probability caller waits.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — CAPACITY PLANNING ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Capacity Planning ★")
    info("How many agents to staff? Balance SLA achievement vs agent cost.")

    results = []
    for s in range(1, 21):
        m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
        if m:
            # Erlang C probability (P(wait>0))
            P_wait = 1 - m["P0"] * (1 + sum(
                (lam_use/mu_use)**n/
                __import__("math").factorial(n)
                for n in range(1,s+1)
            ) / ((lam_use/mu_use)**s/
                 __import__("math").factorial(s)/(1-m["rho"])))
            sla_approx = (1 - max(0,P_wait) *
                          np.exp(-(s*mu_use-lam_use)*sla_sec/3600)) * 100
            results.append({
                "Agents": s, "ρ": round(m["rho"],4),
                "Wq_sec": round(m["Wq"]*3600,1),
                "Wq_min": round(m["Wq"]*60,3),
                "Lq": round(m["Lq"],4),
                "SLA_Approx": round(min(100,max(0,sla_approx)),1),
                "Stable": m["rho"] < 1.0,
            })
        else:
            results.append({"Agents":s,"ρ":"∞","Wq_sec":"∞",
                             "Wq_min":"∞","Lq":"∞","SLA_Approx":0,"Stable":False})

    cap_df = pd.DataFrame(results)
    stable = cap_df[cap_df["Stable"]]

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 SLA % vs Number of Agents")
        fig = px.line(stable, x="Agents", y="SLA_Approx",
                      markers=True,
                      title=f"SLA Achievement vs Agents (target={sla_sec}s)",
                      color_discrete_sequence=[CLR["success"]])
        fig.add_hline(y=80, line_dash="dash", line_color=CLR["danger"],
                      annotation_text="80% SLA target")
        fig.add_hline(y=90, line_dash="dot", line_color=CLR["warning"],
                      annotation_text="90% stretch")
        fig.add_vline(x=S_use, line_dash="dot", line_color=CLR["primary"],
                      annotation_text=f"Current S={S_use}")
        fig.update_layout(height=380, yaxis_title="SLA %")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Wait Time vs Agents")
        fig2 = px.line(stable, x="Agents", y="Wq_sec",
                       markers=True,
                       title="Avg Wait Time (seconds) vs Agents",
                       color_discrete_sequence=[CLR["orange"]])
        fig2.add_hline(y=sla_sec, line_dash="dash", line_color=CLR["success"],
                       annotation_text=f"SLA={sla_sec}s")
        fig2.add_vline(x=S_use, line_dash="dot", line_color=CLR["primary"])
        fig2.update_layout(height=380, yaxis_title="Avg Wait (seconds)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Full Capacity Planning Table")
    st.dataframe(cap_df, use_container_width=True)

    # Find minimum agents for 80% SLA
    sla_80 = stable[stable["SLA_Approx"] >= 80]["Agents"].min()
    sla_90 = stable[stable["SLA_Approx"] >= 90]["Agents"].min()
    if pd.notna(sla_80):
        insight(f"Minimum agents for 80% SLA: {int(sla_80)} agents")
    if pd.notna(sla_90):
        insight(f"Minimum agents for 90% SLA: {int(sla_90)} agents")
    if pd.notna(sla_80) and int(sla_80) > S_use:
        warn(f"Current {S_use} agents INSUFFICIENT for 80% SLA. Need {int(sla_80)}.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — SLA & ABANDONMENT ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("📉 Tab 6 — SLA & Abandonment Analysis ★")
    info("SLA and abandonment are the two most important call center KPIs. "
         "They are inversely related — more agents = higher SLA = lower abandonment.")

    col1, col2, col3 = st.columns(3)
    col1.metric("SLA Achievement",   f"{sla_pct:.1f}%",
                delta=f"{sla_pct-80:.1f}pp vs 80% target")
    col2.metric("Abandonment Rate",  f"{abn_pct:.1f}%",
                delta=f"{abn_pct-5:.1f}pp vs 5% target",
                delta_color="inverse")
    col3.metric("Total Abandoned",   f"{df['abandoned'].sum():,} calls")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 SLA by Call Type")
        if "call_type" in df.columns and "sla_met" in df.columns:
            sla_type = df.groupby("call_type")["sla_met"].mean() * 100
            sla_type = sla_type.sort_values(ascending=False)
            fig = px.bar(x=sla_type.values, y=sla_type.index, orientation="h",
                         color=sla_type.values,
                         color_continuous_scale=["#c62828","#f57f17","#2e7d32"],
                         title="SLA % by Call Type",
                         text=sla_type.values.round(1))
            fig.add_vline(x=80, line_dash="dash", line_color=CLR["danger"])
            fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Abandonment by Call Type")
        if "call_type" in df.columns and "abandoned" in df.columns:
            abn_type = df.groupby("call_type")["abandoned"].mean() * 100
            abn_type = abn_type.sort_values(ascending=False)
            fig2 = px.bar(x=abn_type.values, y=abn_type.index, orientation="h",
                          color=abn_type.values,
                          color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                          title="Abandonment Rate % by Call Type",
                          text=abn_type.values.round(1))
            fig2.add_vline(x=5, line_dash="dash", line_color=CLR["success"],
                           annotation_text="5% target")
            fig2.update_traces(textposition="outside", texttemplate="%{text:.1f}%")
            fig2.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 SLA & Abandonment Trend — Daily")
    if "day" in df.columns:
        daily = df.groupby("day").agg(
            SLA_Pct=("sla_met","mean"), Abn_Pct=("abandoned","mean")
        ).reset_index() * 100
        daily["day"] = df.groupby("day").size().reset_index()["day"]
        daily_raw = df.groupby("day").agg(
            SLA_Pct=("sla_met","mean"), Abn_Pct=("abandoned","mean")
        ).reset_index()
        daily_raw["SLA_Pct"] *= 100; daily_raw["Abn_Pct"] *= 100

        fig3 = go.Figure()
        fig3.add_scatter(x=daily_raw["day"], y=daily_raw["SLA_Pct"].rolling(5).mean(),
                         name="SLA % (5-day MA)", mode="lines",
                         line=dict(color=CLR["success"],width=2.5))
        fig3.add_scatter(x=daily_raw["day"], y=daily_raw["Abn_Pct"].rolling(5).mean(),
                         name="Abandonment % (5-day MA)", mode="lines",
                         line=dict(color=CLR["danger"],width=2.5))
        fig3.add_hline(y=80, line_dash="dash", line_color=CLR["success"],
                       annotation_text="SLA 80%")
        fig3.add_hline(y=5, line_dash="dash", line_color=CLR["danger"],
                       annotation_text="Abn 5%")
        fig3.update_layout(height=380, title="SLA & Abandonment Rate Trend")
        st.plotly_chart(fig3, use_container_width=True)

    if sla_pct < 80:
        warn(f"SLA {sla_pct:.1f}% is BELOW 80% target. "
             f"Add {max(0, int(sla_80)-S_use if pd.notna(sla_80) else 1)} agent(s).")
    else:
        insight(f"SLA {sla_pct:.1f}% meets the 80% target. ✅")

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
            wq_l.append(m["Wq"]*3600 if m and m["Wq"]*3600 < 300 else None)
        fig, ax = plt.subplots(figsize=(7,4))
        valid = [(l,w) for l,w in zip(lam_r,wq_l) if w is not None]
        if valid:
            lx,wx = zip(*valid)
            ax.plot(lx, wx, color=CLR["orange"], lw=2.5, marker="o", ms=4)
            ax.axvline(lam_use, color=CLR["danger"], lw=2, ls="--",
                       label=f"Current λ={lam_use}")
            ax.axhline(sla_sec, color=CLR["success"], lw=1.5, ls=":",
                       label=f"SLA={sla_sec}s")
            ax.set_xlabel("Arrival Rate λ (calls/hr)")
            ax.set_ylabel("Avg Wait (seconds)")
            ax.set_title("Sensitivity to Call Volume", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Wait Time vs Handle Rate μ")
        mu_r  = np.linspace(mu_use*0.5, mu_use*2.0, 20)
        wq_m  = []
        for m_val in mu_r:
            m = MMS(lam_use, m_val, S_use) if ENGINE_OK else None
            wq_m.append(m["Wq"]*3600 if m and m["Wq"]*3600 < 300 else None)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        valid2 = [(m,w) for m,w in zip(mu_r,wq_m) if w is not None]
        if valid2:
            mx,wx2 = zip(*valid2)
            ax2.plot(mx, wx2, color=CLR["success"], lw=2.5, marker="s", ms=4)
            ax2.axvline(mu_use, color=CLR["danger"], lw=2, ls="--",
                        label=f"Current μ={mu_use}")
            ax2.axhline(sla_sec, color=CLR["warning"], lw=1.5, ls=":",
                        label=f"SLA={sla_sec}s")
            ax2.set_xlabel("Handle Rate μ (calls/agent/hr)")
            ax2.set_ylabel("Avg Wait (seconds)")
            ax2.set_title("Sensitivity to AHT Reduction", fontweight="bold")
            ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Heatmap: Wait (seconds) vs λ × Agents")
    lam_vals = np.linspace(lam_use*0.6, lam_use*1.4, 7)
    s_vals   = range(max(1,S_use-2), S_use+5)
    heat_data = {}
    for s in s_vals:
        col_data = []
        for l in lam_vals:
            m = MMS(l, mu_use, s) if ENGINE_OK else None
            val = round(m["Wq"]*3600,0) if m and m["Wq"]*3600 < 300 else 300.0
            col_data.append(val)
        heat_data[f"S={s}"] = col_data
    heat_df = pd.DataFrame(heat_data, index=[f"λ={l:.0f}" for l in lam_vals])
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.heatmap(heat_df, annot=True, fmt=".0f", cmap="RdYlGn_r",
                ax=ax3, linewidths=0.5, annot_kws={"size":9})
    ax3.set_title("Avg Wait (seconds) — Call Volume × Agents",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Reducing AHT by 10% (training/tools) has same effect as adding ~0.5 agents.")
    insight("Call center performance is extremely non-linear near ρ=1 — small volume spikes cause large wait increases.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Business KPIs ★")

    working_days     = 250
    daily_calls      = lam_use * hrs_per_day
    annual_calls     = daily_calls * working_days
    annual_abandoned = int(annual_calls * abn_pct/100)

    agent_cost_day   = S_use * cost_per_agent * hrs_per_day
    agent_cost_ann   = agent_cost_day * working_days
    abandon_cost_ann = annual_abandoned * cost_per_abandon
    wait_cost_ann    = daily_calls*(1-abn_pct/100)*metrics["Wq"]*60*cost_per_wait_min*working_days
    total_cost_ann   = agent_cost_ann + abandon_cost_ann + wait_cost_ann

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Calls",       f"{annual_calls:,.0f}")
    c2.metric("Annual Abandoned",   f"{annual_abandoned:,}")
    c3.metric("Abandon Cost/Year",  f"${abandon_cost_ann:,.0f}")
    c4.metric("Total Cost/Year",    f"${total_cost_ann:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Annual Cost vs Agents")
        opt_res = []
        for s in range(1, 16):
            m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
            if m:
                ac = s * cost_per_agent * hrs_per_day * working_days
                # Abandonment decreases with more agents
                est_abn = max(0, abn_pct - (s-S_use)*5) / 100
                abn_c  = annual_calls * est_abn * cost_per_abandon
                wc     = daily_calls*(1-est_abn)*m["Wq"]*60*cost_per_wait_min*working_days
                opt_res.append({"S":s,"Agent_Cost":ac,"Abandon_Cost":abn_c,
                                 "Wait_Cost":wc,"Total":ac+abn_c+wc})
        opt_df = pd.DataFrame(opt_res)
        best_s = opt_df.loc[opt_df["Total"].idxmin()]

        fig = go.Figure()
        fig.add_bar(name="Agent Cost", x=opt_df["S"],
                    y=opt_df["Agent_Cost"], marker_color=CLR["primary"])
        fig.add_bar(name="Abandon Cost", x=opt_df["S"],
                    y=opt_df["Abandon_Cost"], marker_color=CLR["danger"])
        fig.add_bar(name="Wait Cost", x=opt_df["S"],
                    y=opt_df["Wait_Cost"], marker_color=CLR["warning"])
        fig.add_scatter(x=opt_df["S"], y=opt_df["Total"],
                        name="Total", mode="lines+markers",
                        line=dict(color=CLR["dark"],width=3))
        fig.add_vline(x=best_s["S"], line_dash="dash",
                      line_color=CLR["success"],
                      annotation_text=f"Optimal S={int(best_s['S'])}")
        fig.update_layout(barmode="stack", height=400,
                          title="Annual Cost Breakdown vs Agents")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 KPI Summary Table")
        biz_df = pd.DataFrame({
            "KPI": ["Daily calls","Annual calls","Annual abandoned",
                    "Agent cost/year","Abandon cost/year","Wait cost/year",
                    "Total cost/year","Optimal agents","SLA achievement"],
            "Value": [f"{daily_calls:.0f}",f"{annual_calls:,.0f}",
                      f"{annual_abandoned:,}",f"${agent_cost_ann:,.0f}",
                      f"${abandon_cost_ann:,.0f}",f"${wait_cost_ann:,.0f}",
                      f"${total_cost_ann:,.0f}",
                      f"{int(best_s['S'])} agents",f"{sla_pct:.1f}%"],
        })
        st.dataframe(biz_df, use_container_width=True)

    savings = total_cost_ann - best_s["Total"]
    if savings > 0 and int(best_s["S"]) != S_use:
        insight(f"Optimal {int(best_s['S'])} agents saves ${savings:,.0f}/year vs current {S_use}.")
    else:
        insight(f"Current {S_use} agents is near cost-optimal. ✅")
    insight(f"Each abandoned call costs ${cost_per_abandon} — {annual_abandoned:,} abandoned × "
            f"${cost_per_abandon} = ${abandon_cost_ann:,.0f}/year in lost business.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Findings & Recommendations")
    st.markdown(f"### 📞 Call Center Analytics — Summary Report")
    st.markdown(f"**λ={lam_use}/hr · AHT={60/mu_use:.1f}min · S={S_use} agents · "
                f"ρ={metrics['rho']*100:.1f}% · SLA={sla_sec}s target**")
    st.markdown("---")

    sec("1️⃣ Current Performance")
    insight(f"SLA: {sla_pct:.1f}% of calls answered within {sla_sec}s "
            f"({'✅ Meets' if sla_pct>=80 else '⚠️ Below'} 80% standard)")
    insight(f"Abandonment: {abn_pct:.1f}% "
            f"({'✅ Acceptable' if abn_pct<=5 else '⚠️ HIGH — above 5% standard'})")
    insight(f"Agent utilization: {metrics['rho']*100:.1f}% "
            f"({'✅ Sustainable' if metrics['rho']<0.85 else '⚠️ Overloaded'})")

    sec("2️⃣ Operational Recommendations")
    recs = [
        ("👥 Right-Size Staffing",
         f"Deploy {int(best_s['S'])} agents during peak hours. "
         f"This minimises total cost at ${best_s['Total']:,.0f}/year."),
        ("⏱ Reduce AHT",
         f"Current AHT={60/mu_use:.1f}min. Reducing to {60/mu_use*0.9:.1f}min (10% reduction) "
         f"via agent training and better CRM tools is equivalent to adding 0.5 agents."),
        ("📞 Call Routing",
         "Route Complaint calls (highest AHT) to senior/specialist agents. "
         "Fast-track Billing and Account queries to IVR self-service."),
        ("📊 Real-Time Monitoring",
         f"Alert supervisor when ρ > 85% or SLA drops below 75%. "
         f"Bring in 1 overflow agent immediately."),
        ("🔄 Shift Scheduling",
         "Use peak-hour call volume data to build 30-minute interval staffing schedule. "
         "Erlang C calculation per interval gives precise agent requirements."),
        ("💬 IVR Deflection",
         "Route 20-30% of General and Billing calls to IVR/chatbot. "
         "Reduces effective λ and dramatically improves SLA at zero staffing cost."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""CALL CENTER ANALYTICS — M/M/S REPORT
M3 · M.Sc. Queueing Engine · Call Center Service Template

MODEL: M/M/{S_use} (with abandonment)
  λ (arrival rate)   : {lam_use} calls/hour
  μ (service rate)   : {mu_use} calls/agent/hour (AHT={60/mu_use:.1f}min)
  S (agents)         : {S_use}
  ρ (utilization)    : {metrics['rho']*100:.2f}%
  SLA target         : {sla_sec} seconds

PERFORMANCE:
  SLA achievement    : {sla_pct:.1f}%
  Abandonment rate   : {abn_pct:.1f}%
  Avg wait (Wq)      : {metrics['Wq']*3600:.1f} seconds
  Avg handle time    : {60/mu_use:.1f} minutes

BUSINESS KPIs:
  Annual calls       : {annual_calls:,.0f}
  Annual abandoned   : {annual_abandoned:,}
  Agent cost/year    : ${agent_cost_ann:,.0f}
  Abandon cost/year  : ${abandon_cost_ann:,.0f}
  Total cost/year    : ${total_cost_ann:,.0f}
  Optimal agents     : {int(best_s['S'])}

RECOMMENDATIONS:
  1. Staff {int(best_s['S'])} agents during peak hours
  2. Reduce AHT by 10% via training/tools
  3. Route complaints to senior agents
  4. Alert at ρ>85% → bring overflow agent
  5. Deflect 20-30% to IVR/chatbot
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="CallCenter_Queue_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Capacity Plan (.csv)",
                           cap_df.to_csv(index=False),
                           file_name="CallCenter_Capacity_Plan_M3.csv",
                           mime="text/csv", use_container_width=True)
