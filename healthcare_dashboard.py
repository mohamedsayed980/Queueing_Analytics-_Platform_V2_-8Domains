"""
Healthcare Queue Template — dashboard.py
M.Sc. Queueing Engine Extended to Service Domain: Healthcare
Author : Mohamed · M3

Architecture:
  Synthetic/Real Data → Distribution Fitting → M/M/S Engine
  → Capacity Planning → Interactive Dashboard

Tabs:
  1. Queue Profile & Data Overview
  2. Arrival Pattern Analysis ★
  3. Consultation Time Analysis ★
  4. M/M/S Analytical Results ★
  5. Capacity Planning ★
  6. Simulation Validation ★
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

# ── ENGINE PATH ───────────────────────────────────────────────
ENGINE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ENGINE_DIR))
try:
    from queue_engine import MMS, queue_metrics
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False

st.set_page_config(page_title="Healthcare Queue Analytics · M3",
                   page_icon="🏥", layout="wide")

ACCENT = "#00695c"
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","amber":"#f57f17",
       "grey":"#546e7a","dark":"#1a237e","purple":"#6a1b9a"}

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#e0f2f1;
  border-left:4px solid {ACCENT};border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},#1565c0);
  color:#fff !important;padding:10px 18px;border-radius:8px;
  font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e0f2f1;border-left:4px solid {ACCENT};
  padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#00332e !important;margin:0;font-size:0.93rem;line-height:1.6;}}
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
    st.markdown("## 🏥 Healthcare Queue Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload queue data CSV", type=["csv"],
                            key="health_upload")
    st.divider()
    st.markdown("### ⚙️ Queue Parameters")
    S_override   = st.slider("Doctors (S)", 1, 15, 4)
    lam_override = st.number_input("Arrival rate λ (patients/hr)", 1.0, 60.0, 12.0, 1.0)
    mu_override  = st.number_input("Service rate μ (patients/doctor/hr)", 1.0, 20.0, 4.0, 0.5)
    st.divider()
    st.markdown("### 💰 Business Parameters")
    cost_per_doctor  = st.number_input("Doctor cost ($/hr)", 50, 500, 150, 10)
    cost_per_wait_min= st.number_input("Cost per patient wait-min ($)", 0.1, 10.0, 1.0, 0.1)
    target_wait_min  = st.number_input("Target max wait (min)", 5.0, 120.0, 30.0, 5.0)
    hrs_per_day      = st.slider("Operating hours/day", 6, 24, 10)

    if not ENGINE_OK:
        st.warning("⚠️ queue_engine.py not found\nPlace it in same folder.")

# ── LOAD DATA ─────────────────────────────────────────────────
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "healthcare_queue_data.csv")

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

if _up is not None:
    df = load_data(file_bytes=_up.read())
elif os.path.exists(_data_path):
    df = load_data()
else:
    st.error("❌ No data found.")
    st.info("Run healthcare_generate_data.py in Jupyter → copy to data/ folder.")
    st.stop()

if df.empty:
    st.error("❌ Dataset is empty."); st.stop()

df_sorted = df.sort_values("arrival_time").reset_index(drop=True)
df_sorted["iat"] = df_sorted["arrival_time"].diff().fillna(
    df_sorted["arrival_time"].iloc[0])

n_patients   = len(df_sorted)
total_hrs    = df_sorted["arrival_time"].max() - df_sorted["arrival_time"].min()

lam_use = lam_override
mu_use  = mu_override
S_use   = S_override

# ── ANALYTICAL RESULTS ───────────────────────────────────────
if ENGINE_OK:
    metrics = MMS(lam_use, mu_use, S_use)
else:
    from math import factorial
    def _mms_manual(lam, mu, S):
        a = lam/mu; rho = lam/(S*mu)
        if rho >= 1: return None
        s1 = sum((a**n)/factorial(n) for n in range(S))
        s2 = (a**S)/(factorial(S)*(1-rho))
        P0 = 1/(s1+s2)
        Lq = P0*(a**S)*rho/(factorial(S)*(1-rho)**2)
        Wq = Lq/lam; Ws = Wq+1/mu; Ls = lam*Ws
        return {"rho":rho,"Lq":Lq,"Wq":Wq,"Ws":Ws,"Ls":Ls,"P0":P0}
    metrics = _mms_manual(lam_use, mu_use, S_use)

if metrics is None:
    st.error(f"❌ System UNSTABLE: ρ={lam_use/(S_use*mu_use):.3f} ≥ 1. Add doctors.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#004d40,#00695c);
     padding:36px 40px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#fff;margin:0 0 8px 0;font-size:2.2rem;font-weight:800;'>
    🏥 Healthcare Queue Analytics Platform</h1>
  <p style='color:#b2dfdb;margin:0;font-size:1rem;'>
    M.Sc. Queueing Engine Extended to Healthcare · M/M/S Model ·
    {n_patients:,} Patient Records</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpi(f"{n_patients:,}",           "Patients",          c1)
kpi(f"{S_use}",                  "Doctors",           c2)
kpi(f"{metrics['rho']*100:.1f}%","Utilization ρ",     c3)
kpi(f"{metrics['Wq']*60:.1f}min","Avg Wait Wq",       c4)
kpi(f"{metrics['Lq']:.2f}",      "Queue Length Lq",   c5)
kpi(f"{metrics['Ws']*60:.1f}min","Total System Time",  c6)

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Queue Profile",
    "2 · Arrival Patterns ★",
    "3 · Consultation Analysis ★",
    "4 · M/M/S Results ★",
    "5 · Capacity Planning ★",
    "6 · Simulation Validation ★",
    "7 · Sensitivity Analysis ★",
    "8 · Business KPIs ★",
    "9 · Recommendations",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — QUEUE PROFILE
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Queue Profile & Data Overview")
    info(f"M/M/{S_use} model · λ={lam_use}/hr · μ={mu_use}/hr · "
         f"ρ={metrics['rho']:.3f} · Healthcare Outpatient/ED Setting")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Patient Wait Time Distribution")
        if "wait_time_min" in df_sorted.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            ax.hist(df_sorted["wait_time_min"].clip(0,120), bins=60,
                    color=CLR["teal"], edgecolor="white", alpha=0.85, density=True)
            ax.axvline(df_sorted["wait_time_min"].mean(), color=CLR["danger"],
                       lw=2.5, ls="--",
                       label=f"Mean={df_sorted['wait_time_min'].mean():.1f}min")
            ax.axvline(target_wait_min, color=CLR["warning"],
                       lw=2, ls=":", label=f"Target={target_wait_min:.0f}min")
            ax.set_xlabel("Wait Time (minutes)"); ax.set_ylabel("Density")
            ax.set_title("Patient Wait Time Distribution", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Acuity Mix & Wait by Acuity")
        if "acuity" in df_sorted.columns:
            acuity_wait = df_sorted.groupby("acuity")["wait_time_min"].mean()
            acuity_cnt  = df_sorted["acuity"].value_counts()
            fig2, axes  = plt.subplots(1, 2, figsize=(7,4))
            colors_a = [CLR["success"],CLR["warning"],CLR["danger"]]
            axes[0].pie(acuity_cnt.values, labels=acuity_cnt.index,
                        colors=colors_a, autopct="%1.1f%%",
                        startangle=90)
            axes[0].set_title("Acuity Distribution", fontweight="bold")
            axes[1].bar(acuity_wait.index, acuity_wait.values,
                        color=colors_a, edgecolor="white")
            axes[1].set_ylabel("Avg Wait (min)")
            axes[1].set_title("Avg Wait by Acuity", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📄 Sample Patient Records")
        st.dataframe(df_sorted.head(10), use_container_width=True)
    with col4:
        sec("📊 Key Statistics")
        stats_df = df_sorted[["wait_time_min","consultation_min",
                               "total_time_min"]].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
        breach_pct = df_sorted["long_wait"].mean()*100 \
                     if "long_wait" in df_sorted.columns else \
                     (df_sorted["wait_time_min"]>target_wait_min).mean()*100
        st.metric("Breach Rate (>target)", f"{breach_pct:.1f}%")

    if metrics["rho"] > 0.85:
        warn(f"ρ={metrics['rho']:.3f} > 85% — system overloaded. Add doctors immediately.")
    else:
        insight(f"ρ={metrics['rho']:.3f} — system operating in safe range.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ARRIVAL PATTERNS ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Patient Arrival Patterns ★")
    info("When do patients arrive? Peak hours drive staffing decisions.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Hourly Arrival Rate")
        if "hour" in df_sorted.columns:
            hourly = df_sorted.groupby("hour").size() / df_sorted["day"].max()
            fig = px.bar(x=hourly.index, y=hourly.values,
                         color=hourly.values,
                         color_continuous_scale=["#e0f2f1","#00695c","#c62828"],
                         title="Average Patients per Hour",
                         labels={"x":"Hour of Day","y":"Patients/Day"},
                         text=hourly.values.round(1))
            fig.update_traces(textposition="outside")
            fig.add_hline(y=hourly.mean(), line_dash="dash",
                          line_color=CLR["danger"],
                          annotation_text=f"Avg={hourly.mean():.1f}")
            fig.update_layout(height=380, showlegend=False,
                              xaxis=dict(tickmode="linear"))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Inter-Arrival Time — Poisson Test")
        fig2, ax = plt.subplots(figsize=(7,4))
        iat_min = df_sorted["iat"].clip(0, 0.5) * 60
        ax.hist(iat_min, bins=50, color=CLR["teal"],
                edgecolor="white", alpha=0.85, density=True, label="Empirical")
        x_exp = np.linspace(0, iat_min.max(), 200)
        y_exp = stats.expon.pdf(x_exp, scale=60/lam_use)
        ax.plot(x_exp, y_exp, color=CLR["danger"], lw=2.5, ls="--",
                label=f"Exp(λ={lam_use}/hr)")
        ax.set_xlabel("Inter-Arrival Time (min)"); ax.set_ylabel("Density")
        ax.set_title("Inter-Arrival: Empirical vs Poisson", fontweight="bold")
        ax.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Daily Patient Volume — 300 Days")
    if "day" in df_sorted.columns:
        daily = df_sorted.groupby("day").size().reset_index(name="patients")
        fig3 = px.line(daily, x="day", y="patients",
                       title="Daily Patient Volume",
                       color_discrete_sequence=[CLR["teal"]])
        roll = daily["patients"].rolling(7, center=True).mean()
        fig3.add_scatter(x=daily["day"], y=roll, name="7-day MA",
                         line=dict(color=CLR["danger"], width=2.5))
        fig3.add_hline(y=daily["patients"].mean(), line_dash="dot",
                       line_color=CLR["grey"],
                       annotation_text=f"Avg={daily['patients'].mean():.0f}/day")
        fig3.update_layout(height=360)
        st.plotly_chart(fig3, use_container_width=True)

    insight(f"λ={lam_use}/hr → {lam_use*hrs_per_day:.0f} patients/day over {hrs_per_day}hr operation.")
    if "rush_hour" in df_sorted.columns:
        rush = df_sorted[df_sorted["rush_hour"]==1]["wait_time_min"].mean()
        normal = df_sorted[df_sorted["rush_hour"]==0]["wait_time_min"].mean()
        insight(f"Rush hour avg wait: {rush:.1f} min vs off-peak: {normal:.1f} min.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — CONSULTATION ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("⏱ Tab 3 — Consultation Time Analysis ★")
    info("Service rate μ = patients per doctor per hour. "
         "Clinical mix (routine vs complex) drives this.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Consultation Time Distribution")
        if "consultation_min" in df_sorted.columns:
            fig, ax = plt.subplots(figsize=(7,4))
            svc = df_sorted["consultation_min"].clip(0,60)
            ax.hist(svc, bins=60, color=CLR["success"],
                    edgecolor="white", alpha=0.85, density=True, label="Empirical")
            x_exp = np.linspace(0, svc.max(), 300)
            y_exp = stats.expon.pdf(x_exp, scale=60/mu_use)
            ax.plot(x_exp, y_exp, color=CLR["danger"], lw=2, ls="--",
                    label=f"Exp(μ={mu_use}/hr)")
            ax.set_xlabel("Consultation Time (min)"); ax.set_ylabel("Density")
            ax.set_title("Consultation: Empirical vs Fitted", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Consultation Time by Acuity")
        if "acuity" in df_sorted.columns and "consultation_min" in df_sorted.columns:
            fig2, ax2 = plt.subplots(figsize=(7,4))
            for acuity, color in [("Low",CLR["success"]),
                                   ("Medium",CLR["warning"]),
                                   ("High",CLR["danger"])]:
                data = df_sorted[df_sorted["acuity"]==acuity]["consultation_min"]
                if len(data) > 0:
                    data.clip(0,60).plot.kde(ax=ax2, color=color,
                                             lw=2.5, label=f"{acuity} (n={len(data):,})")
            ax2.set_xlabel("Consultation Time (min)"); ax2.set_ylabel("Density")
            ax2.set_title("Consultation by Acuity Level", fontweight="bold")
            ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Doctor Workload Distribution")
    if "doctor_id" in df_sorted.columns:
        doc_load = df_sorted.groupby("doctor_id").agg(
            Patients=("patient_id","count"),
            Avg_Consult=("consultation_min","mean"),
            Avg_Wait=("wait_time_min","mean")
        ).reset_index()
        fig3 = px.bar(doc_load, x="doctor_id", y="Patients",
                      color="Avg_Wait",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Patient Load & Avg Wait per Doctor",
                      labels={"doctor_id":"Doctor ID","Patients":"Total Patients"},
                      text="Patients")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=370)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(doc_load.round(2), use_container_width=True)

    if "consultation_min" in df_sorted.columns:
        mu_emp = 60/df_sorted["consultation_min"].mean()
        insight(f"Empirical μ={mu_emp:.2f} patients/hr · Mean consultation: "
                f"{df_sorted['consultation_min'].mean():.1f} min.")
    insight(f"Mixed acuity distribution creates non-exponential service times — "
            f"M/M/S is a conservative approximation.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — M/M/S ANALYTICAL RESULTS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S Analytical Results ★")
    info(f"M/M/{S_use} model · λ={lam_use} patients/hr · "
         f"μ={mu_use} patients/doctor/hr · Exact solution.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Utilization ρ",        f"{metrics['rho']*100:.2f}%")
    c2.metric("Queue Length Lq",      f"{metrics['Lq']:.4f} patients")
    c3.metric("System Length Ls",     f"{metrics['Ls']:.4f} patients")
    c4,c5,c6 = st.columns(3)
    c4.metric("Wait in Queue Wq",     f"{metrics['Wq']*60:.3f} min")
    c5.metric("Time in System Ws",    f"{metrics['Ws']*60:.3f} min")
    c6.metric("P(doctor idle) P₀",    f"{metrics['P0']*100:.2f}%")

    st.markdown("---")
    sec("📊 Full Performance Measures Table")
    full_df = pd.DataFrame({
        "Symbol": ["λ","μ","S","a=λ/μ","ρ","P₀","Lq","Ls","Wq","Ws"],
        "Name":   ["Arrival rate","Service rate","Doctors","Traffic intensity",
                   "Utilization","Idle probability","Queue length","System length",
                   "Queue waiting time","System time"],
        "Value":  [f"{lam_use} patients/hr",
                   f"{mu_use} patients/doctor/hr ({60/mu_use:.0f} min/patient)",
                   f"{S_use} doctors",
                   f"{lam_use/mu_use:.4f}",
                   f"{metrics['rho']:.4f} ({metrics['rho']*100:.1f}%)",
                   f"{metrics['P0']:.4f} ({metrics['P0']*100:.1f}%)",
                   f"{metrics['Lq']:.4f} patients",
                   f"{metrics['Ls']:.4f} patients",
                   f"{metrics['Wq']*60:.4f} minutes",
                   f"{metrics['Ws']*60:.4f} minutes"],
        "Clinical Meaning": [
            "Patient demand rate","Doctor throughput","Staff count",
            "Load per doctor","How busy doctors are",
            "% time all doctors idle","Avg patients in waiting room",
            "Avg patients in dept","Avg time sitting in waiting room",
            "Avg total time from arrival to leaving"],
    })
    st.dataframe(full_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Analytical vs Empirical Comparison")
    if "wait_time_min" in df_sorted.columns:
        emp_wq = df_sorted["wait_time_min"].mean()
        emp_ws = (df_sorted["wait_time_min"] + df_sorted["consultation_min"]).mean() \
                 if "consultation_min" in df_sorted.columns else None
        comp_df = pd.DataFrame({
            "Metric":     ["Wq (min)","Ws (min)","ρ","Lq"],
            "Analytical": [round(metrics["Wq"]*60,3),
                           round(metrics["Ws"]*60,3),
                           round(metrics["rho"],4),
                           round(metrics["Lq"],4)],
            "Empirical":  [round(emp_wq,3),
                           round(emp_ws,3) if emp_ws else "N/A",
                           round(lam_use/(S_use*mu_use),4),
                           round(df_sorted["queue_at_arrival"].mean(),4)
                           if "queue_at_arrival" in df_sorted.columns else "N/A"],
        })
        st.dataframe(comp_df, use_container_width=True)

    if metrics["Wq"]*60 > target_wait_min:
        warn(f"Wq={metrics['Wq']*60:.1f} min EXCEEDS target {target_wait_min:.0f} min. "
             f"Add doctors to comply with healthcare standards.")
    else:
        insight(f"Wq={metrics['Wq']*60:.1f} min is within target {target_wait_min:.0f} min. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 5 — CAPACITY PLANNING ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Capacity Planning ★")
    info("How many doctors do we need to meet clinical wait-time standards?")

    results = []
    for s in range(1, 16):
        m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
        if m:
            results.append({
                "Doctors": s, "ρ": round(m["rho"],4),
                "Lq": round(m["Lq"],4), "Wq_min": round(m["Wq"]*60,3),
                "Ws_min": round(m["Ws"]*60,3), "P0": round(m["P0"],4),
                "Stable": m["rho"] < 1.0,
                "Meets_Target": m["Wq"]*60 <= target_wait_min,
            })
        else:
            results.append({"Doctors":s,"ρ":"∞","Lq":"∞","Wq_min":"∞",
                             "Ws_min":"∞","P0":"-","Stable":False,"Meets_Target":False})

    cap_df = pd.DataFrame(results)
    stable  = cap_df[cap_df["Stable"]]

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait Time vs Doctors")
        fig = px.line(stable, x="Doctors", y="Wq_min",
                      markers=True, title="Avg Wait Time vs Number of Doctors",
                      color_discrete_sequence=[CLR["teal"]])
        fig.add_hline(y=target_wait_min, line_dash="dash",
                      line_color=CLR["danger"],
                      annotation_text=f"Target={target_wait_min:.0f}min")
        fig.add_vline(x=S_use, line_dash="dot", line_color=CLR["primary"],
                      annotation_text=f"Current S={S_use}")
        fig.update_layout(height=380, yaxis_title="Avg Wait (min)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Utilization vs Doctors")
        fig2 = px.bar(stable, x="Doctors", y="ρ",
                      color="ρ",
                      color_continuous_scale=["#2e7d32","#f57f17","#c62828"],
                      title="Utilization ρ vs Doctors",
                      text=stable["ρ"].apply(lambda x: f"{x:.3f}"))
        fig2.add_hline(y=0.85, line_dash="dash", line_color=CLR["warning"],
                       annotation_text="85% threshold")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Full Capacity Planning Table")
    cap_display = cap_df.copy()
    st.dataframe(cap_display, use_container_width=True)

    min_doc = stable[stable["Meets_Target"]]["Doctors"].min()
    if pd.notna(min_doc):
        insight(f"Minimum doctors to meet {target_wait_min:.0f}min target: "
                f"{int(min_doc)} doctors")
        if int(min_doc) > S_use:
            warn(f"Current {S_use} doctors is INSUFFICIENT. "
                 f"Need {int(min_doc)} to meet clinical target.")
        else:
            insight(f"Current {S_use} doctors meets the {target_wait_min:.0f}min target. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 6 — SIMULATION VALIDATION ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🔬 Tab 6 — Simulation vs Analytical Validation ★")
    info("Validates M/M/S assumptions against empirical patient data.")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Analytical Formula Results")
        ana_df = pd.DataFrame({
            "Metric":  ["ρ","Wq","Ws","Lq","Ls","P₀"],
            "Value":   [f"{metrics['rho']:.4f}",
                        f"{metrics['Wq']*60:.3f} min",
                        f"{metrics['Ws']*60:.3f} min",
                        f"{metrics['Lq']:.4f} patients",
                        f"{metrics['Ls']:.4f} patients",
                        f"{metrics['P0']*100:.2f}%"],
        })
        st.dataframe(ana_df, use_container_width=True)

    with col2:
        sec("📊 Empirical Data Results")
        if "wait_time_min" in df_sorted.columns:
            p_long = (df_sorted["wait_time_min"] > target_wait_min).mean()*100
            emp_df = pd.DataFrame({
                "Metric": ["Wq empirical","Ws empirical","Breach rate",
                           "P(wait=0)","Max wait"],
                "Value":  [f"{df_sorted['wait_time_min'].mean():.3f} min",
                           f"{(df_sorted['wait_time_min']+df_sorted['consultation_min']).mean():.3f} min"
                           if "consultation_min" in df_sorted.columns else "N/A",
                           f"{p_long:.1f}% > {target_wait_min:.0f}min",
                           f"{(df_sorted['wait_time_min']==0).mean()*100:.1f}%",
                           f"{df_sorted['wait_time_min'].max():.1f} min"],
            })
            st.dataframe(emp_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Empirical CDF vs Analytical")
    if "wait_time_min" in df_sorted.columns:
        fig, ax = plt.subplots(figsize=(10,5))
        sorted_w = np.sort(df_sorted["wait_time_min"])
        cdf_emp  = np.arange(1, len(sorted_w)+1) / len(sorted_w)
        ax.plot(sorted_w, cdf_emp, color=CLR["teal"], lw=2.5,
                label="Empirical CDF", alpha=0.85)
        # P(Wq ≤ t)
        wq_mean = metrics["Wq"]*60
        if wq_mean > 0:
            x_range = np.linspace(0, min(sorted_w.max(), wq_mean*8), 400)
            P_wait = metrics["Lq"] * mu_use / lam_use if lam_use > 0 else 0
            rate   = (S_use * mu_use - lam_use) / 60  # per minute
            cdf_ana = np.where(x_range==0, 1-P_wait,
                               1 - P_wait * np.exp(-rate*x_range))
            cdf_ana = np.clip(cdf_ana, 0, 1)
            ax.plot(x_range, cdf_ana, color=CLR["danger"], lw=2.5, ls="--",
                    label="Analytical (M/M/S approx)")
        ax.axvline(target_wait_min, color=CLR["warning"], lw=2, ls=":",
                   label=f"Target={target_wait_min:.0f}min")
        ax.axhline(0.95, color=CLR["grey"], lw=1, ls=":",
                   label="95th percentile")
        ax.set_xlabel("Wait Time (minutes)"); ax.set_ylabel("CDF P(Wq ≤ t)")
        ax.set_title("Wait Time CDF: Empirical vs Analytical",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(0, min(90, sorted_w.max()))
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    p95 = df_sorted["wait_time_min"].quantile(0.95) \
          if "wait_time_min" in df_sorted.columns else "N/A"
    insight(f"95th percentile wait: {p95:.1f} min — "
            f"95% of patients wait less than this.")
    info("M/M/S assumes exponential service — mixed acuity creates slight deviation. "
         "Model remains conservative (overestimates wait for routine patients).")

# ══════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITY ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — Sensitivity Analysis ★")
    info("How sensitive is patient wait time to changes in demand, "
         "consultation speed, or staffing?")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Wait vs Arrival Rate λ")
        lam_r = np.linspace(lam_use*0.5, lam_use*1.5, 20)
        wq_l  = []
        for l in lam_r:
            m = MMS(l, mu_use, S_use) if ENGINE_OK else None
            wq_l.append(m["Wq"]*60 if m and m["Wq"]*60 < 300 else None)
        fig, ax = plt.subplots(figsize=(7,4))
        valid = [(l,w) for l,w in zip(lam_r,wq_l) if w is not None]
        if valid:
            lx,wx = zip(*valid)
            ax.plot(lx, wx, color=CLR["teal"], lw=2.5, marker="o", ms=4)
            ax.axvline(lam_use, color=CLR["danger"], lw=2, ls="--",
                       label=f"Current λ={lam_use}")
            ax.axhline(target_wait_min, color=CLR["warning"], lw=1.5, ls=":",
                       label=f"Target={target_wait_min:.0f}min")
            ax.set_xlabel("Arrival Rate λ (patients/hr)")
            ax.set_ylabel("Avg Wait Wq (min)")
            ax.set_title("Sensitivity to Arrival Rate", fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Wait vs Service Rate μ")
        mu_r  = np.linspace(mu_use*0.5, mu_use*2.0, 20)
        wq_m  = []
        for m_val in mu_r:
            m = MMS(lam_use, m_val, S_use) if ENGINE_OK else None
            wq_m.append(m["Wq"]*60 if m and m["Wq"]*60 < 300 else None)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        valid2 = [(m,w) for m,w in zip(mu_r,wq_m) if w is not None]
        if valid2:
            mx,wx2 = zip(*valid2)
            ax2.plot(mx, wx2, color=CLR["success"], lw=2.5, marker="s", ms=4)
            ax2.axvline(mu_use, color=CLR["danger"], lw=2, ls="--",
                        label=f"Current μ={mu_use}")
            ax2.axhline(target_wait_min, color=CLR["warning"], lw=1.5, ls=":",
                        label=f"Target={target_wait_min:.0f}min")
            ax2.set_xlabel("Service Rate μ (patients/doctor/hr)")
            ax2.set_ylabel("Avg Wait Wq (min)")
            ax2.set_title("Sensitivity to Consultation Speed", fontweight="bold")
            ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Heatmap: Wait Time vs λ × Doctors")
    lam_vals = np.linspace(lam_use*0.6, lam_use*1.4, 8)
    s_vals   = range(max(1,S_use-2), S_use+5)
    heat_data = {}
    for s in s_vals:
        col_data = []
        for l in lam_vals:
            m = MMS(l, mu_use, s) if ENGINE_OK else None
            val = round(m["Wq"]*60,1) if m and m["Wq"]*60 < 120 else 120.0
            col_data.append(val)
        heat_data[f"S={s}"] = col_data
    heat_df = pd.DataFrame(heat_data, index=[f"λ={l:.0f}" for l in lam_vals])
    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.heatmap(heat_df, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax3, linewidths=0.5, annot_kws={"size":9})
    ax3.set_title("Patient Wait Time (min) — Arrival Rate × Doctors",
                  fontsize=12, fontweight="bold")
    ax3.set_ylabel("Arrival Rate λ"); ax3.set_xlabel("Number of Doctors S")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Small reductions in consultation time (faster μ) reduce wait dramatically near ρ=1.")
    insight("Adding 1 doctor at high load reduces wait exponentially — non-linear effect.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Healthcare Business KPIs ★")
    info("Translate queueing metrics into clinical and financial impact.")

    working_days     = 300
    daily_patients   = lam_use * hrs_per_day
    annual_patients  = daily_patients * working_days

    doc_cost_daily   = S_use * cost_per_doctor * hrs_per_day
    doc_cost_annual  = doc_cost_daily * working_days
    wait_cost_daily  = daily_patients * metrics["Wq"]*60 * cost_per_wait_min
    wait_cost_annual = wait_cost_daily * working_days
    total_cost_ann   = doc_cost_annual + wait_cost_annual

    breach_pct = (df_sorted["wait_time_min"] > target_wait_min).mean()*100 \
                 if "wait_time_min" in df_sorted.columns else \
                 (metrics["Wq"]*60 > target_wait_min)*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Patients",     f"{annual_patients:,.0f}")
    c2.metric("Doctor Cost/Year",    f"${doc_cost_annual:,.0f}")
    c3.metric("Wait Cost/Year",      f"${wait_cost_annual:,.0f}")
    c4.metric("Breach Rate",         f"{breach_pct:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Annual Cost vs Number of Doctors")
        opt_results = []
        for s in range(1, 16):
            m = MMS(lam_use, mu_use, s) if ENGINE_OK else None
            if m:
                dc = s * cost_per_doctor * hrs_per_day * working_days
                wc = daily_patients * m["Wq"]*60 * cost_per_wait_min * working_days
                opt_results.append({"S":s,"Doctor_Cost":dc,"Wait_Cost":wc,
                                    "Total":dc+wc,"Wq_min":m["Wq"]*60,
                                    "Meets_Target":m["Wq"]*60<=target_wait_min})

        opt_df = pd.DataFrame(opt_results)
        best_s = opt_df.loc[opt_df["Total"].idxmin()]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Doctor Cost", x=opt_df["S"],
                             y=opt_df["Doctor_Cost"], marker_color=CLR["teal"]))
        fig.add_trace(go.Bar(name="Wait Cost", x=opt_df["S"],
                             y=opt_df["Wait_Cost"], marker_color=CLR["danger"]))
        fig.add_scatter(x=opt_df["S"], y=opt_df["Total"],
                        name="Total Cost", mode="lines+markers",
                        line=dict(color=CLR["dark"],width=3))
        fig.add_vline(x=best_s["S"], line_dash="dash",
                      line_color=CLR["success"],
                      annotation_text=f"Optimal S={int(best_s['S'])}",
                      annotation_position="top right")
        fig.update_layout(barmode="stack", height=400,
                          title="Annual Cost: Doctor vs Patient Wait",
                          yaxis_title="Annual Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 Clinical KPI Summary")
        p95 = df_sorted["wait_time_min"].quantile(0.95) \
              if "wait_time_min" in df_sorted.columns else metrics["Wq"]*60*2
        clin_df = pd.DataFrame({
            "KPI": ["Daily patients","Annual patients",
                    "Avg wait (Wq)","95th pctile wait",
                    "Breach rate (>target)","Doctor cost/year",
                    "Wait cost/year","Total cost/year",
                    "Optimal doctors","Optimal total cost"],
            "Value": [f"{daily_patients:.0f}",f"{annual_patients:,.0f}",
                      f"{metrics['Wq']*60:.2f} min",f"{p95:.1f} min",
                      f"{breach_pct:.1f}%",f"${doc_cost_annual:,.0f}",
                      f"${wait_cost_annual:,.0f}",f"${total_cost_ann:,.0f}",
                      f"{int(best_s['S'])} doctors",f"${best_s['Total']:,.0f}"],
        })
        st.dataframe(clin_df, use_container_width=True)

    savings = total_cost_ann - best_s["Total"]
    if int(best_s["S"]) != S_use and savings > 0:
        insight(f"Optimal: {int(best_s['S'])} doctors saves ${savings:,.0f}/year.")
    elif int(best_s["S"]) != S_use:
        warn(f"Optimal config is {int(best_s['S'])} doctors (${best_s['Total']:,.0f}/yr).")
    else:
        insight(f"Current {S_use} doctors is cost-optimal. ✅")

# ══════════════════════════════════════════════════════════════
# TAB 9 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Clinical Findings & Recommendations")

    st.markdown(f"### 🏥 Healthcare Queue Analytics — Summary Report")
    st.markdown(f"**λ={lam_use}/hr · μ={mu_use}/hr · S={S_use} doctors · "
                f"ρ={metrics['rho']*100:.1f}% · M/M/{S_use} Model**")
    st.markdown("---")

    sec("1️⃣ Current Clinical Performance")
    insight(f"Average patient wait: {metrics['Wq']*60:.1f} min "
            f"({'✅ Within' if metrics['Wq']*60<=target_wait_min else '⚠️ Exceeds'} "
            f"{target_wait_min:.0f}min target)")
    insight(f"Doctor utilization: {metrics['rho']*100:.1f}% — "
            f"{'✅ Sustainable' if metrics['rho']<0.85 else '⚠️ Overloaded'}")
    if "wait_time_min" in df_sorted.columns:
        breach = (df_sorted["wait_time_min"]>target_wait_min).mean()*100
        warn(f"{breach:.1f}% of patients wait longer than {target_wait_min:.0f}min target.") \
            if breach > 5 else \
            insight(f"Only {breach:.1f}% breach the {target_wait_min:.0f}min target. ✅")

    sec("2️⃣ Staffing Recommendations")
    recs = [
        ("👨‍⚕️ Optimal Staffing",
         f"Deploy {int(best_s['S'])} doctors during peak hours (λ={lam_use}/hr). "
         f"Cost-optimal at ${best_s['Total']:,.0f}/year."),
        ("⏰ Dynamic Scheduling",
         "Morning (9-11AM) and afternoon (2-4PM) peaks require maximum staffing. "
         "Reduce to 2-3 doctors during mid-morning and late afternoon lulls."),
        ("🎯 Consultation Efficiency",
         f"Current avg consultation: {df_sorted['consultation_min'].mean():.1f} min. "
         "Standardised protocols for routine cases can reduce this by 15-20%."),
        ("📊 Utilization Control",
         f"Keep ρ < 85%. Current: {metrics['rho']*100:.1f}%. "
         "Trigger on-call doctor when ρ > 80% for 30+ consecutive minutes."),
        ("🏥 Triage Optimisation",
         "Fast-track low-acuity patients to nurse practitioners — "
         "reduces doctor load for complex cases and improves μ."),
        ("📱 Appointment Scheduling",
         "Introduce appointment slots to smooth λ. "
         "Even 30% pre-booked reduces peak arrival variance significantly."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""HEALTHCARE QUEUE ANALYTICS — M/M/S REPORT
M3 · M.Sc. Queueing Engine · Healthcare Service Template

MODEL: M/M/{S_use}
  λ (arrival rate)   : {lam_use} patients/hour
  μ (service rate)   : {mu_use} patients/doctor/hour ({60/mu_use:.0f} min/patient)
  S (doctors)        : {S_use}
  ρ (utilization)    : {metrics['rho']*100:.2f}%
  Operating hours    : {hrs_per_day} hrs/day · {working_days} days/year

PERFORMANCE MEASURES:
  Wq (avg wait)      : {metrics['Wq']*60:.3f} min
  Ws (total time)    : {metrics['Ws']*60:.3f} min
  Lq (queue length)  : {metrics['Lq']:.4f} patients
  P0 (idle prob)     : {metrics['P0']*100:.2f}%

CLINICAL KPIs:
  Daily patients     : {daily_patients:.0f}
  Annual patients    : {annual_patients:,.0f}
  Breach rate        : {breach_pct:.1f}%
  Doctor cost/year   : ${doc_cost_annual:,.0f}
  Wait cost/year     : ${wait_cost_annual:,.0f}
  Optimal doctors    : {int(best_s['S'])}

RECOMMENDATIONS:
  1. Deploy {int(best_s['S'])} doctors during peak hours
  2. Dynamic scheduling: reduce off-peak, surge at peak
  3. Fast-track low-acuity to nurse practitioners
  4. Introduce 30% appointment slots to smooth arrivals
  5. Monitor ρ — add on-call when ρ > 80%
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Clinical Report (.txt)", report_txt,
                           file_name="Healthcare_Queue_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        cap_csv = cap_df.to_csv(index=False) if 'cap_df' in dir() else ""
        st.download_button("📥 Download Capacity Plan (.csv)", cap_csv,
                           file_name="Healthcare_Capacity_Plan_M3.csv",
                           mime="text/csv", use_container_width=True)
