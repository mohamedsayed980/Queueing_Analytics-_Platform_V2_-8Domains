"""
Port / Logistics Queue Template - port_dashboard.py
M.Sc. Queueing Engine extended to a container terminal.   Author: Mohamed · M3

MODEL   ships (customers) wait at the ANCHORAGE and are served at S identical BERTHS.
        Rates are per DAY (ships/day), waiting times are shown in HOURS.

RULES FOLLOWED (lessons from the other templates)
  * every KPI card / warning is ANALYTIC (changes with the sliders); the CSV is used
    only for charts, distributions and history
  * Plotly figures are named pfigN, no Matplotlib in this page
  * no constants copied from the generator script
  * Erlang-C is computed in a function that cannot divide by zero

TABS  1 Terminal Overview · 2 Arrival Patterns · 3 Vessels & Cranes · 4 M/M/S Results ·
      5 Berth Planning · 6 Delay & Demurrage · 7 Sensitivity · 8 Business KPIs · 9 Recommendations
"""
import io
import re
import sys
import pathlib
import warnings
from math import factorial, exp, ceil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Port & Logistics Queue Analytics · M3",
                   page_icon="🚢", layout="wide")

# ── where am I? works from the repo root AND from pages/ ─────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name.lower() == "pages" else _HERE
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

H = 24.0                      # hours per day
ACCENT = "#00838f"
CLR = {"primary": "#1565c0", "success": "#2e7d32", "warning": "#e65100", "danger": "#c62828",
       "teal": ACCENT, "amber": "#f57f17", "grey": "#546e7a", "dark": "#006064"}
CLASS_ORDER = ["Feeder", "Panamax", "Post-Panamax"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _stretch():
    """width kwargs that work on old AND new Streamlit versions."""
    try:
        import inspect
        p = inspect.signature(st.plotly_chart).parameters
        if "use_container_width" not in p and "width" in p:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch()

st.markdown(f"""
<style>
[data-testid="stSidebar"]{{background:#0f1923;}}
[data-testid="stSidebar"] *{{color:#e0e8f0 !important;}}
.main{{background:#f4f7fb;}}
div[data-testid="metric-container"]{{background:#e0f7fa;border-left:4px solid {ACCENT};
  border-radius:6px;padding:10px 14px;}}
.sec-header{{background:linear-gradient(90deg,{ACCENT},{CLR['dark']});color:#fff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}}
.insight-box{{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}}
.insight-box p{{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.warn-box{{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}}
.warn-box p{{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.info-box{{background:#e0f7fa;border-left:4px solid {ACCENT};padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}}
.info-box p{{color:#00363a !important;margin:0;font-size:0.93rem;line-height:1.6;}}
.kpi-card{{background:#fff;border-radius:10px;padding:16px;text-align:center;
  box-shadow:0 2px 10px rgba(0,0,0,0.08);border-bottom:3px solid {ACCENT};}}
.kpi-num{{font-size:1.7rem;font-weight:800;color:{ACCENT};}}
.kpi-lbl{{font-size:0.8rem;color:#546e7a;margin-top:4px;}}
</style>""", unsafe_allow_html=True)


def sec(t):     st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)


def kpi(num, lbl, col):
    col.markdown(f'<div class="kpi-card"><div class="kpi-num">{num}</div>'
                 f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# QUEUEING MATHS - own verified M/M/S (Erlang-B recursion), rates per DAY
# ═════════════════════════════════════════════════════════════════════════════
# Set USE_ENGINE = True only after check_engine.py reports ALL PASS for your queue_engine.py.
USE_ENGINE = False


def mms(lam, mu, S):
    """M/M/S metrics (same time unit as 1/mu). None when the system is unstable."""
    try:
        S = int(S)
        if lam <= 0 or mu <= 0 or S < 1:
            return None
        a, rho = lam / mu, lam / (S * mu)
        if rho >= 1:
            return None
        if USE_ENGINE:
            from queue_engine import MMS as _MMS
            e = _MMS(lam, mu, S)
            if e is None:
                return None
        B = 1.0
        for k in range(1, S + 1):
            B = a * B / (k + a * B)
        C = B / (1 - rho * (1 - B))                    # Erlang C = P(wait > 0)
        Lq = C * rho / (1 - rho)
        if USE_ENGINE:
            Lq = float(e["Lq"])
        Wq = Lq / lam
        P0 = 1 / (sum(a ** n / factorial(n) for n in range(S)) + a ** S / (factorial(S) * (1 - rho)))
        return dict(lam=lam, mu=mu, S=S, a=a, rho=rho, C=C, P0=P0, Lq=Lq, Wq=Wq, Ws=Wq + 1 / mu, Ls=Lq + a)
    except Exception:
        return None


def p_exceed(m, t_days):
    """P(wait in the queue > t) for t in days - exact for M/M/S."""
    theta = m["S"] * m["mu"] - m["lam"]
    return float(min(1.0, max(0.0, m["C"] * exp(-theta * t_days))))


def wait_quantile(m, q):
    """wait (days) that a fraction q of ships does not exceed."""
    theta = m["S"] * m["mu"] - m["lam"]
    if m["C"] <= 1 - q:
        return 0.0
    return float(np.log(m["C"] / (1 - q)) / theta)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚢 Port & Logistics Queue Analytics")
    st.markdown("**M.Sc. Queueing Engine · Service Template**")
    st.divider()
    st.markdown("### 📂 Data Source")
    _up = st.file_uploader("Upload ship-call CSV", type=["csv"], key="port_upload")
    st.divider()
    st.markdown("### ⚙️ Terminal Parameters")
    S_use = st.slider("Berths (S)", 1, 10, 3, key="port_S")
    lam_use = st.number_input("Ship arrivals λ (ships/day)", 0.5, 40.0, 8.0, 0.5, key="port_lam")
    svc_hr = st.number_input("Mean berth time (hours)", 1.0, 72.0, 7.5, 0.5, key="port_svc")
    sla_hr = st.number_input("Anchorage wait target (hours)", 0.5, 48.0, 6.0, 0.5, key="port_sla")
    sla_pct = st.slider("Required compliance (% of ships)", 50, 99, 80, key="port_slapct")
    st.divider()
    st.markdown("### 💰 Business Parameters (illustrative)")
    rate_f = st.number_input("Demurrage Feeder ($/ship-hour)", 100, 5000, 600, 50, key="port_rf")
    rate_p = st.number_input("Demurrage Panamax ($/ship-hour)", 100, 5000, 1000, 50, key="port_rp")
    rate_pp = st.number_input("Demurrage Post-Panamax ($/ship-hour)", 100, 5000, 1500, 50, key="port_rpp")
    berth_cost_day = st.number_input("Berth operating cost ($/berth/day)", 1000, 50000, 8000, 500, key="port_bc")
    rev_teu = st.number_input("Revenue per TEU handled ($)", 20, 500, 180, 10, key="port_rev")
    days_yr = st.slider("Operating days per year", 200, 365, 365, key="port_days")

mu_use = H / svc_hr                                    # ships/day/berth
RATES = {"Feeder": rate_f, "Panamax": rate_p, "Post-Panamax": rate_pp}

# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
REQUIRED = ["arrival_time", "wait_time_hr", "service_time_hr", "turnaround_hr", "vessel_class",
            "teu_exchanged", "cranes_assigned", "handling_hr", "overhead_hr", "berth_id",
            "service_end", "queue_at_arrival", "day", "hour"]


def _load_port_df(up):
    if up is not None:
        d = pd.read_csv(io.BytesIO(up.getvalue()))
        d.columns = d.columns.str.strip()
        return d, up.name
    try:                                               # shared loader if the repo has it
        import data_loader
        d = data_loader.load_domain_data("port_logistics")
        if d is not None:
            return d, "data/ folder"
    except ImportError:
        pass
    for folder in (_ROOT / "data", _HERE / "data"):    # self-contained fallback
        if folder.exists():
            hits = sorted(f for f in folder.iterdir() if f.name.lower().endswith(".csv")
                          and "portlogistics" in re.sub(r"[\s_\-]+", "", f.name.lower()))
            if hits:
                f = min(hits, key=lambda x: len(x.name))
                d = pd.read_csv(f, encoding="utf-8-sig")
                d.columns = d.columns.str.strip()
                return d, f.name
    return None, None


df, src_name = _load_port_df(_up)
if df is None or df.empty:
    st.error("No ship-call data found. Upload port_logistics_queue_data.csv in the sidebar, "
             "or place it in the repo's data/ folder.")
    _d = _ROOT / "data"
    st.code(f"looked in: {_d}\nfiles there: "
            + (", ".join(sorted(p.name for p in _d.iterdir())) if _d.exists() else "(folder missing)"))
    st.info("Create the file with port_generate_data.py.")
    st.stop()

_missing = [c for c in REQUIRED if c not in df.columns]
if _missing:
    st.error(f"The data file is missing columns: {', '.join(_missing)}")
    st.info("Use the file produced by port_generate_data.py.")
    st.stop()

df = df.sort_values("arrival_time").reset_index(drop=True)

# ── facts taken from the data (history, not sliders) ─────────────────────────
n_ships = len(df)
n_days_data = int(df["day"].max())
lam_emp = n_ships / n_days_data
svc_emp = float(df["service_time_hr"].mean())
horizon = float(df["service_end"].max())
S_data = int(df["berth_id"].nunique())
iat = df["arrival_time"].diff().dropna()
ca2 = float((iat.std() / iat.mean()) ** 2)                       # arrival variability
cs2 = float((df["service_time_hr"].std() / svc_emp) ** 2)        # service variability
mix = df["vessel_class"].value_counts(normalize=True)
classes = [c for c in CLASS_ORDER if c in mix.index] + [c for c in mix.index if c not in CLASS_ORDER]
avg_rate = float(sum(mix[c] * RATES.get(c, np.mean(list(RATES.values()))) for c in mix.index))
avg_teu = float(df["teu_exchanged"].mean())
handling_share = float(df["handling_hr"].mean() / (df["handling_hr"].mean() + df["overhead_hr"].mean()))

# ═════════════════════════════════════════════════════════════════════════════
# ANALYTIC RESULTS FOR THE CURRENT SLIDERS
# ═════════════════════════════════════════════════════════════════════════════
m = mms(lam_use, mu_use, S_use)
if m is None:
    cap = S_use * mu_use
    need = int(lam_use / mu_use) + 1
    st.error(f"UNSTABLE: arrivals λ = {lam_use:.1f} ships/day exceed berth capacity "
             f"S·μ = {cap:.1f} ships/day (ρ = {lam_use / cap:.2f}). The anchorage queue grows without limit.")
    st.info(f"At least {need} berths are needed, or a shorter berth time. "
            f"Change the sidebar parameters to continue.")
    st.stop()

wq_h = m["Wq"] * H
tur_h = wq_h + svc_hr
comp = (1 - p_exceed(m, sla_hr / H)) * 100                       # % of ships within the wait target
ships_yr = lam_use * days_yr

st.markdown(f"""
<div style='background:linear-gradient(135deg,{CLR['dark']},{ACCENT});padding:34px 40px;
border-radius:14px;margin-bottom:22px;'>
<h1 style='color:#fff;margin:0 0 8px 0;font-size:2.1rem;font-weight:800;'>
🚢 Port & Logistics Queue Analytics Platform</h1>
<p style='color:#b2ebf2;margin:0;font-size:1rem;'>
M.Sc. Queueing Engine extended to a container terminal · M/M/{S_use} berths ·
{n_ships:,} ship calls · {n_days_data} days · data: {src_name}</p></div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(f"{n_ships:,}", "Ship calls analysed", k1)
kpi(f"{S_use}", "Berths", k2)
kpi(f"{m['rho'] * 100:.1f}%", "Berth utilisation ρ", k3)
kpi(f"{wq_h:.1f} h", "Avg anchorage wait", k4)
kpi(f"{comp:.0f}%", f"Waiting ≤ {sla_hr:g} h", k5)
kpi(f"{tur_h:.1f} h", "Avg turnaround", k6)

tabs = st.tabs(["1 · Terminal Overview", "2 · Arrival Patterns ★", "3 · Vessels & Cranes ★",
                "4 · M/M/S Results ★", "5 · Berth Planning ★", "6 · Delay & Demurrage ★",
                "7 · Sensitivity ★", "8 · Business KPIs ★", "9 · Recommendations"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 - OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🚢 Tab 1 — Terminal Overview")
    info(f"M/M/{S_use} · λ = {lam_use:g} ships/day · μ = {mu_use:.2f} ships/day/berth "
         f"(mean berth time {svc_hr:g} h) · ρ = {m['rho']:.3f}. Cards use the formulas; charts use the data.")
    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Anchorage wait — distribution in the data")
        pfig1 = px.histogram(df, x="wait_time_hr", nbins=50, color_discrete_sequence=[ACCENT])
        pfig1.add_vline(x=float(df["wait_time_hr"].mean()), line_dash="dash", line_color=CLR["danger"],
                        annotation_text=f"data mean {df['wait_time_hr'].mean():.1f} h")
        pfig1.add_vline(x=wq_h, line_dash="dot", line_color=CLR["dark"],
                        annotation_text=f"M/M/S {wq_h:.1f} h", annotation_position="top left")
        pfig1.add_vline(x=sla_hr, line_dash="dashdot", line_color=CLR["warning"],
                        annotation_text=f"target {sla_hr:g} h", annotation_position="bottom right")
        pfig1.update_layout(height=380, xaxis_title="Wait before berthing (hours)",
                            yaxis_title="Ships", showlegend=False)
        st.plotly_chart(pfig1, **STRETCH)
    with c2:
        sec("📊 Vessel class mix")
        pfig2 = px.pie(values=[mix[c] for c in classes], names=classes, hole=0.45,
                       color_discrete_sequence=[CLR["primary"], CLR["amber"], CLR["danger"], CLR["grey"]])
        pfig2.update_traces(textinfo="percent+label")
        pfig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(pfig2, **STRETCH)

    c3, c4 = st.columns(2)
    with c3:
        sec("📊 Utilisation per berth — data")
        util = (df.groupby("berth_id")["service_time_hr"].sum() / horizon * 100).round(1)
        pfig3 = px.bar(x=[f"Berth {int(i)}" for i in util.index], y=util.values, text=util.values,
                       color_discrete_sequence=[ACCENT])
        pfig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        pfig3.update_layout(height=340, yaxis_title="% of time occupied", xaxis_title="", yaxis_range=[0, 105])
        st.plotly_chart(pfig3, **STRETCH)
    with c4:
        sec("📋 Key statistics — data")
        stats_df = pd.DataFrame({
            "Statistic": ["Ship calls", "Ships per day", "Mean berth time", "Mean anchorage wait",
                          "Ships that waited", "Longest wait", "Mean turnaround", "Containers (TEU)",
                          "TEU per ship"],
            "Value": [f"{n_ships:,}", f"{lam_emp:.2f}", f"{svc_emp:.2f} h",
                      f"{df['wait_time_hr'].mean():.2f} h",
                      f"{(df['wait_time_hr'] > 0.01).mean() * 100:.1f}%",
                      f"{df['wait_time_hr'].max():.1f} h", f"{df['turnaround_hr'].mean():.2f} h",
                      f"{int(df['teu_exchanged'].sum()):,}", f"{avg_teu:.0f}"]})
        st.dataframe(stats_df, hide_index=True, **STRETCH)

    if m["rho"] > 0.85:
        warn(f"ρ = {m['rho']:.3f} is above 85%: ships queue at the anchorage most of the time. "
             f"Add a berth or shorten berth time.")
    else:
        insight(f"ρ = {m['rho']:.3f} — berth load is in a workable range.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 - ARRIVALS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Arrival Patterns ★")
    info("Ships follow a published schedule, so arrivals are less random than customers at a bank: "
         "the weekday matters, the hour of day does not (ports run 24/7).")
    daily = df.groupby("day").size().reindex(range(1, n_days_data + 1), fill_value=0)
    wk = daily.groupby((daily.index - 1) % 7).mean().reindex(range(7))
    hr = df.groupby("hour").size().reindex(range(24), fill_value=0) / n_days_data

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Ships per day by weekday")
        pfig4 = px.bar(x=WEEKDAYS, y=wk.values, text=wk.values, color_discrete_sequence=[ACCENT])
        pfig4.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        pfig4.add_hline(y=lam_emp, line_dash="dash", line_color=CLR["danger"],
                        annotation_text=f"average {lam_emp:.1f}")
        pfig4.update_layout(height=360, yaxis_title="ships / day", xaxis_title="")
        st.plotly_chart(pfig4, **STRETCH)
    with c2:
        sec("📊 Arrivals by hour of day")
        pfig5 = px.bar(x=list(range(24)), y=hr.values, color_discrete_sequence=[CLR["primary"]])
        pfig5.update_layout(height=360, yaxis_title="ships / day", xaxis_title="hour of day")
        st.plotly_chart(pfig5, **STRETCH)

    sec("📊 Daily ship arrivals over the year")
    pfig6 = go.Figure()
    pfig6.add_scatter(x=daily.index, y=daily.values, mode="lines", name="ships/day",
                      line=dict(color=CLR["grey"], width=1), opacity=0.5)
    pfig6.add_scatter(x=daily.index, y=daily.rolling(7, center=True).mean(), mode="lines",
                      name="7-day average", line=dict(color=ACCENT, width=3))
    pfig6.update_layout(height=330, xaxis_title="day of study", yaxis_title="ships")
    st.plotly_chart(pfig6, **STRETCH)

    c3, c4 = st.columns(2)
    with c3:
        sec("📊 Time between arrivals vs exponential")
        mean_iat = float(iat.mean())
        xs = np.linspace(0, float(iat.quantile(0.995)), 200)
        pfig7 = go.Figure()
        pfig7.add_histogram(x=iat, nbinsx=60, histnorm="probability density", name="data",
                            marker_color=ACCENT, opacity=0.75)
        pfig7.add_scatter(x=xs, y=np.exp(-xs / mean_iat) / mean_iat, mode="lines",
                          name="exponential (Poisson arrivals)", line=dict(color=CLR["danger"], width=3))
        pfig7.update_layout(height=340, xaxis_title="hours between ships", yaxis_title="density")
        st.plotly_chart(pfig7, **STRETCH)
    with c4:
        sec("📊 Ships already waiting when a ship arrives")
        qa = df["queue_at_arrival"].clip(upper=8).value_counts(normalize=True).sort_index() * 100
        pfig8 = px.bar(x=[("8+" if i >= 8 else str(int(i))) for i in qa.index], y=qa.values,
                       color_discrete_sequence=[CLR["amber"]])
        pfig8.update_layout(height=340, xaxis_title="ships in the anchorage queue", yaxis_title="% of arrivals")
        st.plotly_chart(pfig8, **STRETCH)

    cv_a = float(np.sqrt(ca2))
    insight(f"Arrival rate in the data: {lam_emp:.2f} ships/day · coefficient of variation of the gaps "
            f"{cv_a:.2f} (1.00 = pure Poisson).")
    if wk.max() > 1.05 * lam_emp:
        insight(f"The busiest weekday carries {wk.max():.1f} ships/day against {wk.min():.1f} on the quietest — "
                f"berth windows should follow the weekly cycle.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 - VESSELS & CRANES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🏗️ Tab 3 — Vessels & Crane Analysis ★")
    info("Berth time = handling (containers ÷ cranes × crane speed) + overhead (pilot, mooring, unmooring). "
         "Bigger ships get more cranes, so berth time grows more slowly than ship size.")
    g = df.groupby("vessel_class")
    tbl = pd.DataFrame({
        "Ships": g.size(),
        "Share of ships %": (g.size() / n_ships * 100).round(1),
        "Avg TEU": g["teu_exchanged"].mean().round(0),
        "Cranes": g["cranes_assigned"].agg(lambda s: int(s.mode().iloc[0])),
        "Handling (h)": g["handling_hr"].mean().round(2),
        "Overhead (h)": g["overhead_hr"].mean().round(2),
        "Berth time (h)": g["service_time_hr"].mean().round(2),
        "Share of berth-hours %": (g["service_time_hr"].sum() / df["service_time_hr"].sum() * 100).round(1),
    }).reindex(classes).reset_index().rename(columns={"vessel_class": "Vessel class"})
    st.dataframe(tbl, hide_index=True, **STRETCH)

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Berth time by vessel class")
        pfig9 = px.box(df, x="vessel_class", y="service_time_hr", color="vessel_class",
                       category_orders={"vessel_class": classes},
                       color_discrete_sequence=[CLR["primary"], CLR["amber"], CLR["danger"], CLR["grey"]])
        pfig9.update_layout(height=370, showlegend=False, xaxis_title="", yaxis_title="berth time (hours)")
        st.plotly_chart(pfig9, **STRETCH)
    with c2:
        sec("📊 Containers vs handling time")
        samp = df.sample(min(1500, n_ships), random_state=1)
        pfig10 = px.scatter(samp, x="teu_exchanged", y="handling_hr", color="vessel_class", opacity=0.55,
                            category_orders={"vessel_class": classes},
                            color_discrete_sequence=[CLR["primary"], CLR["amber"], CLR["danger"], CLR["grey"]])
        pfig10.update_layout(height=370, xaxis_title="TEU exchanged", yaxis_title="handling time (hours)")
        st.plotly_chart(pfig10, **STRETCH)

    c3, c4 = st.columns(2)
    with c3:
        sec("📊 Berth productivity (TEU per berth-hour)")
        prod = (df["teu_exchanged"] / df["service_time_hr"]).groupby(df["vessel_class"]).mean().reindex(classes)
        pfig11 = px.bar(x=prod.index, y=prod.values, text=prod.values, color_discrete_sequence=[ACCENT])
        pfig11.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        pfig11.update_layout(height=330, xaxis_title="", yaxis_title="TEU per berth-hour")
        st.plotly_chart(pfig11, **STRETCH)
    with c4:
        sec("📊 Where the berth hours go")
        pfig12 = go.Figure()
        pfig12.add_bar(x=classes, y=[tbl.set_index("Vessel class").loc[c, "Handling (h)"] for c in classes],
                       name="handling", marker_color=ACCENT)
        pfig12.add_bar(x=classes, y=[tbl.set_index("Vessel class").loc[c, "Overhead (h)"] for c in classes],
                       name="overhead", marker_color=CLR["amber"])
        pfig12.update_layout(barmode="stack", height=330, yaxis_title="hours per call")
        st.plotly_chart(pfig12, **STRETCH)

    top = tbl.sort_values("Share of berth-hours %", ascending=False).iloc[0]
    big = tbl.iloc[-1]
    ov_share = (1 - handling_share) * 100
    insight(f"{top['Vessel class']} ships take {top['Share of berth-hours %']:.0f}% of all berth-hours; "
            f"{big['Vessel class']} is {big['Share of ships %']:.0f}% of calls but "
            f"{big['Share of berth-hours %']:.0f}% of berth-hours.")
    insight(f"Overhead (pilot, mooring, unmooring) is {ov_share:.0f}% of berth time — "
            f"the part that does not depend on cranes.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 - M/M/S RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔢 Tab 4 — M/M/S Analytical Results ★")
    info(f"M/M/{S_use}: exact formulas for λ = {lam_use:g} ships/day, μ = {mu_use:.2f} ships/day/berth.")
    a1, a2, a3 = st.columns(3)
    a1.metric("Berth utilisation ρ", f"{m['rho'] * 100:.2f}%")
    a2.metric("Ships waiting Lq", f"{m['Lq']:.3f}")
    a3.metric("Ships in port Ls", f"{m['Ls']:.3f}")
    b1, b2, b3 = st.columns(3)
    b1.metric("Avg anchorage wait Wq", f"{wq_h:.2f} h")
    b2.metric("Avg turnaround Ws", f"{tur_h:.2f} h")
    b3.metric("Ships that must wait", f"{m['C'] * 100:.1f}%")

    measures = pd.DataFrame({
        "Symbol": ["λ", "μ", "S", "a = λ/μ", "ρ", "P₀", "Erlang C", "Lq", "Ls", "Wq", "Ws"],
        "Value": [f"{lam_use:g} ships/day", f"{mu_use:.3f} ships/day/berth ({svc_hr:g} h)", f"{S_use} berths",
                  f"{m['a']:.3f}", f"{m['rho']:.4f}", f"{m['P0']:.4f}", f"{m['C']:.4f}",
                  f"{m['Lq']:.3f} ships", f"{m['Ls']:.3f} ships", f"{wq_h:.3f} h", f"{tur_h:.3f} h"],
        "Port meaning": ["Ship arrivals per day", "Ships one berth can finish per day", "Open berths",
                         "Berths' worth of work offered", "Share of berth time occupied",
                         "Probability all berths are idle", "Probability a ship has to wait",
                         "Average ships at anchorage", "Average ships at anchorage + at berth",
                         "Average wait before berthing", "Average arrival-to-departure time"]})
    st.dataframe(measures, hide_index=True, **STRETCH)

    sec("📊 Formula vs variability-adjusted vs data")
    ac = (ca2 + cs2) / 2
    emp_rho = float(df["service_time_hr"].sum() / (S_data * horizon))
    comp_df = pd.DataFrame({
        "Measure": ["Berth utilisation ρ", "Avg anchorage wait (h)", "Avg turnaround (h)",
                    "Ships waiting Lq (average)"],
        f"M/M/S at sidebar values": [f"{m['rho']:.3f}", f"{wq_h:.2f}", f"{tur_h:.2f}", f"{m['Lq']:.2f}"],
        "Variability-adjusted (Allen–Cunneen)": [f"{m['rho']:.3f}", f"{wq_h * ac:.2f}",
                                                 f"{wq_h * ac + svc_hr:.2f}", f"{m['Lq'] * ac:.2f}"],
        f"Data ({S_data} berths, {lam_emp:.1f}/day)": [f"{emp_rho:.3f}", f"{df['wait_time_hr'].mean():.2f}",
                                                     f"{df['turnaround_hr'].mean():.2f}",
                                                     f"{df['queue_at_arrival'].mean():.2f}"]})
    st.dataframe(comp_df, hide_index=True, **STRETCH)
    info(f"Variability in the data: arrivals c² = {ca2:.2f}, berth time c² = {cs2:.2f} (exponential = 1.00). "
         f"Berth times are much more regular than M/M/S assumes, so the plain formula is a conservative "
         f"(upper) estimate. The adjusted column scales the wait by (c²ₐ + c²ₛ)/2 — an approximation "
         f"that assumes the same variability as the data. The data column is directly comparable only "
         f"when the sidebar matches the data settings.")
    if wq_h > sla_hr:
        warn(f"Average wait {wq_h:.1f} h exceeds the {sla_hr:g} h target.")
    else:
        insight(f"Average wait {wq_h:.1f} h is inside the {sla_hr:g} h target.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 - BERTH PLANNING
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    sec("📐 Tab 5 — Berth Capacity Planning ★")
    info(f"How many berths keep {sla_pct}% of ships waiting no more than {sla_hr:g} h?")
    rows = []
    for s in range(1, 13):
        ms = mms(lam_use, mu_use, s)
        if ms is None:
            rows.append({"Berths": s, "ρ": round(lam_use / (s * mu_use), 3), "Status": "❌ unstable",
                         "Wait (h)": np.nan, "Turnaround (h)": np.nan, "Compliance %": np.nan,
                         "Meets target": "❌"})
        else:
            cs_ = (1 - p_exceed(ms, sla_hr / H)) * 100
            rows.append({"Berths": s, "ρ": round(ms["rho"], 3), "Status": "stable",
                         "Wait (h)": round(ms["Wq"] * H, 2), "Turnaround (h)": round(ms["Wq"] * H + svc_hr, 2),
                         "Compliance %": round(cs_, 1), "Meets target": "✅" if cs_ >= sla_pct else "❌"})
    cap_df = pd.DataFrame(rows)
    stable = cap_df[cap_df["Status"] == "stable"]

    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Anchorage wait vs berths")
        pfig13 = px.line(stable, x="Berths", y="Wait (h)", markers=True, color_discrete_sequence=[ACCENT])
        pfig13.add_hline(y=sla_hr, line_dash="dash", line_color=CLR["danger"],
                         annotation_text=f"target {sla_hr:g} h")
        pfig13.add_vline(x=S_use, line_dash="dot", line_color=CLR["grey"], annotation_text=f"now S={S_use}")
        pfig13.update_layout(height=370, yaxis_title="average wait (hours)")
        st.plotly_chart(pfig13, **STRETCH)
    with c2:
        sec("📊 Compliance vs berths")
        pfig14 = px.bar(stable, x="Berths", y="Compliance %", text="Compliance %",
                        color_discrete_sequence=[CLR["success"]])
        pfig14.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        pfig14.add_hline(y=sla_pct, line_dash="dash", line_color=CLR["danger"],
                         annotation_text=f"required {sla_pct}%")
        pfig14.update_layout(height=370, yaxis_range=[0, 110], yaxis_title=f"% of ships waiting ≤ {sla_hr:g} h")
        st.plotly_chart(pfig14, **STRETCH)

    st.dataframe(cap_df, hide_index=True, **STRETCH)
    ok_rows = stable[stable["Meets target"] == "✅"]
    s_min = int(ok_rows["Berths"].min()) if not ok_rows.empty else None
    if s_min is None:
        warn("No berth count up to 12 meets the target at these settings.")
    elif s_min > S_use:
        warn(f"{S_use} berths do not meet the target — {s_min} berths are needed "
             f"({comp:.0f}% compliance now, {sla_pct}% required).")
    else:
        insight(f"{S_use} berths meet the target ({comp:.0f}% ≥ {sla_pct}%). Minimum needed: {s_min}.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 - DELAY & DEMURRAGE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🚨 Tab 6 — Delay & Demurrage Risk ★")
    info("Demurrage is paid per hour a ship waits at the anchorage. The wait is long-tailed: "
         "the average hides the ships that wait a day or more.")
    ts_h = np.linspace(0, 48, 97)
    pfig15 = go.Figure()
    for s, colr in ((S_use - 1, CLR["danger"]), (S_use, ACCENT), (S_use + 1, CLR["success"])):
        ms = mms(lam_use, mu_use, s) if s >= 1 else None
        if ms is None:
            continue
        pfig15.add_scatter(x=ts_h, y=[p_exceed(ms, t / H) * 100 for t in ts_h], mode="lines",
                           name=f"{s} berths" + (" (now)" if s == S_use else ""),
                           line=dict(color=colr, width=4 if s == S_use else 2))
    pfig15.add_vline(x=sla_hr, line_dash="dash", line_color=CLR["warning"], annotation_text=f"target {sla_hr:g} h")
    pfig15.add_hline(y=100 - sla_pct, line_dash="dot", line_color=CLR["grey"],
                     annotation_text=f"{100 - sla_pct}% allowed to exceed")
    pfig15.update_layout(height=400, xaxis_title="wait t (hours)", yaxis_title="% of ships waiting MORE than t")
    st.plotly_chart(pfig15, **STRETCH)

    c1, c2 = st.columns(2)
    with c1:
        sec("📋 Wait percentiles")
        qs = [0.50, 0.75, 0.90, 0.95, 0.99]
        pct_df = pd.DataFrame({
            "Share of ships": [f"{int(q * 100)}%" for q in qs],
            "Wait ≤ (h) — M/M/S": [round(wait_quantile(m, q) * H, 1) for q in qs],
            "Wait ≤ (h) — data": [round(float(df["wait_time_hr"].quantile(q)), 1) for q in qs]})
        st.dataframe(pct_df, hide_index=True, **STRETCH)
        st.caption("M/M/S follows the sidebar; the data column is fixed history and is lower "
                   "because berth times are more regular than exponential.")
    with c2:
        sec("💰 Demurrage by vessel class")
        dm = pd.DataFrame({
            "Vessel class": classes,
            "Ships / year": [round(ships_yr * mix[c]) for c in classes],
            "Rate ($/h)": [RATES.get(c, avg_rate) for c in classes],
            "Cost per ship ($)": [round(wq_h * RATES.get(c, avg_rate)) for c in classes]})
        dm["Annual cost ($)"] = dm["Ships / year"] * dm["Cost per ship ($)"]
        st.dataframe(dm, hide_index=True, **STRETCH)
        st.metric("Total expected demurrage / year", f"${dm['Annual cost ($)'].sum():,.0f}")

    p90 = wait_quantile(m, 0.90) * H
    warn(f"Average wait {wq_h:.1f} h, but 1 ship in 10 waits {p90:.1f} h or more (M/M/S estimate).") \
        if p90 > 2 * max(wq_h, 0.1) else \
        insight(f"Average wait {wq_h:.1f} h; 90% of ships wait {p90:.1f} h or less.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 - SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    sec("📉 Tab 7 — Sensitivity Analysis ★")
    c1, c2 = st.columns(2)
    with c1:
        sec("📊 Wait vs ship arrivals λ")
        lam_max = S_use * mu_use
        lam_grid = np.linspace(0.3 * lam_use, min(1.6 * lam_use, 0.985 * lam_max), 30)
        w1 = [(x, mms(x, mu_use, S_use)) for x in lam_grid]
        w1 = [(x, r["Wq"] * H) for x, r in w1 if r is not None]
        pfig16 = px.line(x=[a for a, _ in w1], y=[b for _, b in w1], color_discrete_sequence=[ACCENT])
        pfig16.add_vline(x=lam_use, line_dash="dash", line_color=CLR["danger"], annotation_text="now")
        pfig16.add_hline(y=sla_hr, line_dash="dot", line_color=CLR["warning"], annotation_text="target")
        pfig16.update_layout(height=360, xaxis_title="arrivals (ships/day)", yaxis_title="average wait (h)")
        st.plotly_chart(pfig16, **STRETCH)
        st.caption(f"Capacity limit: S·μ = {lam_max:.1f} ships/day — the wait explodes as λ approaches it.")
    with c2:
        sec("📊 Wait vs mean berth time")
        sv_grid = np.linspace(3.0, max(14.0, svc_hr * 1.3), 30)
        w2 = [(x, mms(lam_use, H / x, S_use)) for x in sv_grid]
        w2 = [(x, r["Wq"] * H) for x, r in w2 if r is not None]
        pfig17 = px.line(x=[a for a, _ in w2], y=[b for _, b in w2], color_discrete_sequence=[CLR["success"]])
        pfig17.add_vline(x=svc_hr, line_dash="dash", line_color=CLR["danger"], annotation_text="now")
        pfig17.add_hline(y=sla_hr, line_dash="dot", line_color=CLR["warning"], annotation_text="target")
        pfig17.update_layout(height=360, xaxis_title="mean berth time (hours)", yaxis_title="average wait (h)")
        st.plotly_chart(pfig17, **STRETCH)

    sec("📊 Heatmap: average wait (hours) — arrivals × berths")
    lg = np.round(np.linspace(max(1.0, lam_use * 0.5), lam_use * 1.5, 7), 1)
    sg = list(range(1, 9))
    z = []
    for x in lg:
        z.append([min(48.0, (mms(x, mu_use, s)["Wq"] * H) if mms(x, mu_use, s) else 48.0) for s in sg])
    pfig18 = px.imshow(np.array(z), x=[str(s) for s in sg], y=[f"{x:.1f}" for x in lg], text_auto=".1f",
                       color_continuous_scale="RdYlGn_r", aspect="auto",
                       labels=dict(x="berths", y="arrivals (ships/day)", color="wait (h)"))
    pfig18.update_layout(height=380)
    st.plotly_chart(pfig18, **STRETCH)
    st.caption("Values are capped at 48 h; a cell showing 48.0 is unstable or worse than two days.")

    sec("🏗️ Crane-handling lever: what if handling were faster?")
    lever = []
    for pct in (0, 5, 10, 15, 20):
        svc_new = svc_hr * (1 - handling_share * pct / 100)
        mn = mms(lam_use, H / svc_new, S_use)
        lever.append({"Handling time saved": f"{pct}%", "Berth time (h)": round(svc_new, 2),
                      "Utilisation ρ": round(mn["rho"], 3) if mn else np.nan,
                      "Avg wait (h)": round(mn["Wq"] * H, 2) if mn else np.nan})
    st.dataframe(pd.DataFrame(lever), hide_index=True, **STRETCH)
    l10 = svc_hr * (1 - handling_share * 0.10)
    m10 = mms(lam_use, H / l10, S_use)
    if m10 is not None:
        insight(f"10% faster handling cuts berth time from {svc_hr:.1f} h to {l10:.1f} h and the average "
                f"wait from {wq_h:.1f} h to {m10['Wq'] * H:.1f} h — a berth-sized gain without building a berth.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 - BUSINESS KPIs
# ═════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    sec("💼 Tab 8 — Business KPIs ★")
    info("Illustrative economics: berth operating cost against demurrage paid for waiting ships. "
         "Change the rates in the sidebar to match your terminal.")
    teu_yr = ships_yr * avg_teu
    revenue = teu_yr * rev_teu
    berth_cost_yr = S_use * berth_cost_day * days_yr
    dem_yr = ships_yr * wq_h * avg_rate
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Ships per year", f"{ships_yr:,.0f}")
    d2.metric("Containers per year", f"{teu_yr / 1e3:,.0f}k TEU")
    d3.metric("Demurrage per year", f"${dem_yr / 1e6:,.2f}M")
    d4.metric("Berth capacity headroom", f"{(1 - m['rho']) * 100:.0f}%")

    opt = []
    for s in range(1, 11):
        ms = mms(lam_use, mu_use, s)
        if ms is None:
            continue
        bc = s * berth_cost_day * days_yr
        dc = ships_yr * ms["Wq"] * H * avg_rate
        opt.append({"Berths": s, "Berth cost $M": bc / 1e6, "Demurrage $M": dc / 1e6,
                    "Total $M": (bc + dc) / 1e6, "Wait (h)": ms["Wq"] * H})
    opt_df = pd.DataFrame(opt)
    if opt_df.empty:
        warn("No stable berth count up to 10 at these settings.")
    else:
        best = opt_df.loc[opt_df["Total $M"].idxmin()]
        c1, c2 = st.columns(2)
        with c1:
            sec("📊 Annual cost vs berths")
            pfig19 = go.Figure()
            pfig19.add_bar(x=opt_df["Berths"], y=opt_df["Berth cost $M"], name="berth operating",
                           marker_color=ACCENT)
            pfig19.add_bar(x=opt_df["Berths"], y=opt_df["Demurrage $M"], name="demurrage",
                           marker_color=CLR["danger"])
            pfig19.add_scatter(x=opt_df["Berths"], y=opt_df["Total $M"], name="total", mode="lines+markers",
                               line=dict(color=CLR["dark"], width=3))
            pfig19.add_vline(x=int(best["Berths"]), line_dash="dash", line_color=CLR["success"],
                             annotation_text=f"lowest cost: {int(best['Berths'])}")
            pfig19.update_layout(barmode="stack", height=400, yaxis_title="$ million per year",
                                 xaxis_title="berths")
            st.plotly_chart(pfig19, **STRETCH)
        with c2:
            sec("📋 KPI summary")
            now_total = (berth_cost_yr + dem_yr) / 1e6
            kp = pd.DataFrame({
                "KPI": ["Revenue per year", "Berth operating cost", "Expected demurrage",
                        "Cost per TEU (berth + demurrage)", "Lowest-cost berth count",
                        "Total cost at that count", "Saving vs current berths"],
                "Value": [f"${revenue / 1e6:,.1f}M", f"${berth_cost_yr / 1e6:,.1f}M", f"${dem_yr / 1e6:,.2f}M",
                          f"${(berth_cost_yr + dem_yr) / teu_yr:,.1f}", f"{int(best['Berths'])}",
                          f"${best['Total $M']:,.1f}M", f"${now_total - best['Total $M']:,.2f}M"]})
            st.dataframe(kp, hide_index=True, **STRETCH)
        if int(best["Berths"]) == S_use:
            insight(f"{S_use} berths is the lowest-cost configuration at these rates.")
        else:
            gap = now_total - best["Total $M"]
            (insight if gap > 0 else warn)(
                f"Lowest total cost is at {int(best['Berths'])} berths "
                f"(${best['Total $M']:,.1f}M vs ${now_total:,.1f}M with {S_use}).")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 9 - RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💡 Tab 9 — Findings & Recommendations")
    st.markdown(f"**λ = {lam_use:g} ships/day · {svc_hr:g} h per berth · {S_use} berths · "
                f"ρ = {m['rho'] * 100:.1f}% · M/M/{S_use}**")
    insight(f"Average anchorage wait {wq_h:.1f} h; {comp:.0f}% of ships wait ≤ {sla_hr:g} h "
            f"({'meets' if comp >= sla_pct else 'misses'} the {sla_pct}% requirement).")
    insight(f"Average turnaround {tur_h:.1f} h, of which {svc_hr:g} h at the berth.")

    l10 = svc_hr * (1 - handling_share * 0.10)
    m10 = mms(lam_use, H / l10, S_use)
    w10 = f"{m10['Wq'] * H:.1f}" if m10 else "n/a"
    best_txt = f"{int(best['Berths'])}" if not opt_df.empty else "n/a"
    recs = [
        ("🛳️ Berth count", f"Minimum berths for the {sla_pct}%/{sla_hr:g} h target: "
                           f"{s_min if s_min else 'more than 12'}. Lowest total cost (berth + demurrage): {best_txt}."),
        ("🏗️ Crane productivity", f"10% faster handling takes the average wait from {wq_h:.1f} h to {w10} h. "
                                  f"Add a crane to Panamax and Post-Panamax calls first — they hold the most berth-hours."),
        ("⚓ Pre-berthing overhead", f"Pilotage, mooring and unmooring are {(1 - handling_share) * 100:.0f}% of berth time. "
                                     f"Pre-cleared documents and standing pilot slots shorten every call."),
        ("📅 Weekly berth windows", "Monday–Wednesday arrivals run above average. Offer booked berth windows and "
                                    "steer flexible calls to the quieter weekend days."),
        ("📡 Anchorage information", f"1 ship in 10 waits {wait_quantile(m, 0.9) * H:.0f} h or more. Publish expected "
                                     f"berthing times so carriers can slow-steam instead of waiting at anchor."),
        ("💰 Demurrage terms", f"Expected demurrage ≈ ${dem_yr / 1e6:,.2f}M per year at these rates — "
                               f"the figure to weigh against another berth."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>', unsafe_allow_html=True)

    report = f"""PORT & LOGISTICS QUEUE ANALYTICS - M/M/S REPORT
M3 · M.Sc. Queueing Engine · Port Service Template

MODEL M/M/{S_use}
  arrivals          : {lam_use:g} ships/day
  mean berth time   : {svc_hr:g} h  (mu = {mu_use:.3f} ships/day/berth)
  berths            : {S_use}
  utilisation       : {m['rho'] * 100:.2f}%

PERFORMANCE
  average wait      : {wq_h:.2f} h
  average turnaround: {tur_h:.2f} h
  ships that wait   : {m['C'] * 100:.1f}%
  within {sla_hr:g} h target : {comp:.1f}%  (required {sla_pct}%)

BUSINESS (illustrative rates)
  ships per year    : {ships_yr:,.0f}
  expected demurrage: ${dem_yr:,.0f} per year
  berth cost        : ${berth_cost_yr:,.0f} per year
  minimum berths for target : {s_min if s_min else '>12'}
  lowest-cost berths: {best_txt}
"""
    r1, r2 = st.columns(2)
    with r1:
        st.download_button("📥 Download report (.txt)", report, file_name="Port_Queue_Report_M3.txt",
                           mime="text/plain", key="port_dl_report")
    with r2:
        st.download_button("📥 Download berth plan (.csv)", cap_df.to_csv(index=False),
                           file_name="Port_Berth_Plan_M3.csv", mime="text/csv", key="port_dl_plan")
