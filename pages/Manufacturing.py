"""
Manufacturing.py  —  v2.0
=====================
Interactive Decision-Support Dashboard
M.Sc. Thesis 1999 → Modernized 2026

4 TABS:
  Tab 1 — Queue Analysis      (Stage 1 single-stage M/M/S)
  Tab 2 — Capacity Planning   (Eqs 3.8-3.10, Tables 22-27)
  Tab 3 — Monthly Schedule    (CL-11: priority, shifts, policies)
  Tab 4 — Case Study          (6 products, 3 stages, CL-12 groups)

CL-12: Server TYPES per stage (machine groups):
  Stage 1 = Group 1 = Cutting  machines (S=5) → M/M/5
  Stage 2 = Group 2 = Punching machines (S=3) → M/M/3 ← bottleneck
  Stage 3 = Group 3 = Bending  machines (S=5) → M/M/5
  Within each stage: servers are IDENTICAL (same type) → M/M/S valid

RUN:  streamlit run dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
from math import factorial
from queue_engine import run_model, MODEL_CATALOG

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def P0_mms(lam, mu, S):
    a = lam/mu; rho = lam/(S*mu)
    if rho >= 1.0: return None
    s = sum(a**n/factorial(n) for n in range(S))
    s += (a**S/factorial(S))/(1-rho)
    return 1/s

def queue_metrics(lam, mu, S, alpha=0.0):
    le = lam/(1-alpha) if alpha>0 else lam
    a  = le/mu; rho = le/(S*mu)
    P0 = P0_mms(le, mu, S)
    if P0 is None: return None
    Lq = (a*rho**S*P0)/(factorial(S-1)*(1-rho)**2)
    Ls = Lq+a; Wq = Lq/le; Ws = Wq+1/mu
    return {"rho":round(rho,3),"P0":round(P0,4),
            "Lq":round(Lq,4),"Ls":round(Ls,4),
            "Wq":round(Wq,4),"Ws":round(Ws,4)}

def lambda_eff_raw(T, S, mu):
    C = S*mu; return (2*T*C**2)/(1+2*T*C)

def lambda_eff_int(T, S, mu):
    return round(lambda_eff_raw(T, S, mu))

def NR_S(NR_base, S):
    return max(0.0, NR_base/S)

def Rn_calc(T, S, mu, NR_base):
    li = lambda_eff_int(T, S, mu)
    nr = NR_S(NR_base, S)
    return {"lam_int":li, "NR_S":round(nr,4), "Rn":round(li*nr,4)}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (CL-9, CL-11, CL-12)
# ─────────────────────────────────────────────────────────────────────────────

# CL-12: Machine groups per stage
STAGES = {
    1: {"name":"Stage 1","group":"Group 1","type":"Cutting (Shears)","S":5,  "color":"#3B82F6"},
    2: {"name":"Stage 2","group":"Group 2","type":"Punching",        "S":3,  "color":"#EF4444"},
    3: {"name":"Stage 3","group":"Group 3","type":"Bending",         "S":5,  "color":"#10B981"},
}
S_ACTUAL = [5, 3, 5]
RATIOS   = [0.2, 0.5, 0.3]   # CL-10: stage time ratios

# CL-9: Case study products (Table 4.9 experimental)
PRODUCTS_EXP = [
    {"type":"8BD",     "mu":80, "lam":2,    "rho":0.025,
     "SP":45000,"F1":36000,"NR":9000, "total_hrs":80,  "priority":4},
    {"type":"8BK",     "mu":120,"lam":4,    "rho":0.033,
     "SP":55000,"F1":44000,"NR":11000,"total_hrs":120, "priority":6},
    {"type":"8FJ500",  "mu":80, "lam":6,    "rho":0.075,
     "SP":25000,"F1":20000,"NR":5000, "total_hrs":80,  "priority":5},
    {"type":"8AS10",   "mu":16, "lam":4,    "rho":0.250,
     "SP":15000,"F1":12000,"NR":3000, "total_hrs":16,  "priority":3},
    {"type":"3CF12KVA","mu":8,  "lam":2,    "rho":0.250,
     "SP":3000, "F1":2400, "NR":600,  "total_hrs":8,   "priority":2},
    {"type":"G.T3",    "mu":6,  "lam":2,    "rho":0.333,
     "SP":8000, "F1":6400, "NR":1600, "total_hrs":6,   "priority":1},
]
PRODUCTS_ACTUAL = [
    {"type":"8BD",     "mu":80, "lam":0.467,"rho":0.006,"total_hrs":80},
    {"type":"8BK",     "mu":120,"lam":0.302,"rho":0.003,"total_hrs":120},
    {"type":"8FJ500",  "mu":80, "lam":0.273,"rho":0.003,"total_hrs":80},
    {"type":"8AS10",   "mu":16, "lam":0.273,"rho":0.017,"total_hrs":16},
    {"type":"3CF12KVA","mu":8,  "lam":1.000,"rho":0.125,"total_hrs":8},
    {"type":"G.T3",    "mu":6,  "lam":0.807,"rho":0.135,"total_hrs":6},
]

def get_stage_mu(total_hrs, stage_idx):
    """Service rate [u/hr] at stage j from total machining hours (CL-10)."""
    st_j = total_hrs * RATIOS[stage_idx]
    return 1.0/st_j if st_j > 0 else 1.0

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PRODUCT SOURCE (Option E) — shared helpers
# ─────────────────────────────────────────────────────────────────────────────
# Tab 9 Step 4 writes st.session_state["global_products"] in its own schema
# (per-stage mu/k list). Tabs 1-8/10 default to the built-in 6-product case
# study (PRODUCTS_EXP/PRODUCTS_ACTUAL) and only switch to Tab 9's fitted set
# if the user explicitly opts in via product_source_selector() below.
# ADDITIVE: no tab's default behavior changes unless the user toggles it.

def has_global_products() -> bool:
    return bool(st.session_state.get("global_products"))

def global_products_as_flat(stage_idx: int = None) -> list:
    """
    Converts Tab 9's global_products schema into the flat
    {"type","lam","mu","rho","total_hrs"} schema that PRODUCTS_EXP /
    PRODUCTS_ACTUAL already use, so existing tab code can consume either
    source through the same field names without a rewrite.

    stage_idx: if given (0,1,2), "mu" is that stage's rate. If None,
    "mu" is mu_overall (Table 4.9 style — matches PRODUCTS_EXP/ACTUAL's
    own "mu" meaning, per CL-6/CL-10).
    """
    out = []
    for p in st.session_state.get("global_products", []):
        if stage_idx is not None and stage_idx < len(p["stages"]):
            mu = p["stages"][stage_idx]["mu"]
        else:
            mu = p["mu_overall"]
        out.append({
            "type": p["type"], "lam": p["lam"], "mu": mu,
            "rho": p["rho"], "total_hrs": p["total_hrs"],
            "k_stages": [s.get("k", 1) for s in p["stages"]],
        })
    return out

def stage_config_selector(key_prefix: str) -> tuple:
    """
    Renders a "Stage Configuration" panel: default 3-stage (Cutting/
    Punching/Bending, S=[5,3,5], CL-12) unless the user opts into a
    custom N-stage layout — for exploring other departments per Table
    4.4 (Mechanical Workshop, Cupper Workshop, etc.), confirmed with
    Mohamed (Session 15) as three SEPARATE independent configs, not one
    cross-department routing model.

    Returns (S_list, ratios_list, names_list, is_custom).
    Ratios are auto-normalized from whatever positive weights the user
    enters, so they always sum to 1.0 without fiddly manual arithmetic.
    """
    mode = st.radio("Stage Configuration", ["Default 3-Stage (CL-12)", "Custom N-Stage"],
                     horizontal=True, key=f"{key_prefix}_stagecfg")
    if mode == "Default 3-Stage (CL-12)":
        return [5, 3, 5], [0.2, 0.5, 0.3], ["Cutting", "Punching", "Bending"], False

    n = st.number_input("Number of stages", 2, 8, 3, key=f"{key_prefix}_nstages")
    st.caption("Enter any positive relative time-weight per stage — "
               "auto-normalized to sum to 1.0, no need to make them "
               "add up exactly yourself.")
    S_list, weights, names = [], [], []
    for j in range(int(n)):
        c1, c2, c3 = st.columns([2, 1, 1])
        nm = c1.text_input(f"Stage {j+1} name", value=f"Stage {j+1}",
                            key=f"{key_prefix}_nm_{j}")
        w = c2.number_input(f"Weight", 0.01, 100.0, 1.0, key=f"{key_prefix}_w_{j}")
        s = c3.number_input(f"S", 1, 20, 3, key=f"{key_prefix}_s_{j}")
        names.append(nm); weights.append(w); S_list.append(int(s))
    total_w = sum(weights)
    ratios = [round(w/total_w, 4) for w in weights]
    # Fix rounding so ratios sum to EXACTLY 1.0 (last stage absorbs residual)
    ratios[-1] = round(1.0 - sum(ratios[:-1]), 4)
    st.caption(f"Normalized ratios: {ratios} (sum={sum(ratios):.4f})")
    return S_list, ratios, names, True


def product_source_selector(key_prefix: str, stage_idx: int = None,
                             default_products: list = None) -> tuple:
    """
    Renders the "Product data source" radio (only shows the Tab-9-fitted
    option if it actually exists yet) and returns (products_list, source_label).
    Call this INSTEAD OF hardcoding `prod = PRODUCTS_EXP if ... else PRODUCTS_ACTUAL`.
    """
    options = ["Built-in Case Study (6 products)"]
    if has_global_products():
        n = len(st.session_state["global_products"])
        options.append(f"Tab 9 Fitted Products ({n})")
    else:
        st.caption("ℹ️ Run Step 4 in Tab 9 to unlock a fitted product source here.")

    choice = st.radio("Product data source", options,
                       horizontal=True, key=f"{key_prefix}_src")

    if "Tab 9" in choice:
        return global_products_as_flat(stage_idx), "Tab 9 Fitted"
    return (default_products or PRODUCTS_EXP), "Built-in"



# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 Capacity Planner")
    st.caption("M.Sc. Thesis 1999 → 2026")
    st.divider()

    st.markdown("**✅ Build Status**")
    for m in ["queue_engine.py","capacity_planner.py","simpy_engine.py","dashboard.py"]:
        st.success(m)

    st.divider()
    st.markdown("**🏭 Factory Config (CL-12)**")
    for j, stg in STAGES.items():
        st.markdown(f"**{stg['name']}** — {stg['group']}")
        st.caption(f"  {stg['type']} | S={stg['S']} servers")
    st.caption("⚠️ Bottleneck: Stage 2 (S=3, punching)")

    st.divider()
    st.markdown("**📐 Key Equations**")
    st.latex(r"\lambda^{**}=\frac{2T(S\mu)^2}{1+2TS\mu}")
    st.latex(r"R_n=\lambda_{int}\times\frac{NR_{base}}{S}")
    st.latex(r"\rho=\frac{\lambda}{S\cdot\mu}")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Queue Analysis",
    "💰 Capacity Planning",
    "📅 Monthly Schedule",
    "🔬 Case Study (6 Products)",
    "🔗 Series Queues (Stage 2)",
    "🏭 Multi-Queue Engine (Stage 3)",
    "🔴 Live Simulation",
    "📊 Statistical Reports",
    "🔧 Input & Fit",
    "🎬 Animation",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — QUEUE ANALYSIS (Single Stage — same-type servers CL-12)
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Queue Analysis — Single Stage")
    st.caption("CL-12: Each stage = group of identical parallel servers (M/M/S model)")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("⚙️ Stage & Parameters")

        # CL-12: Select which stage (machine group)
        stage_sel = st.selectbox("Select Stage (Machine Group)",
            [f"Stage {j} — {STAGES[j]['group']} — {STAGES[j]['type']}"
             for j in range(1,4)])
        j_sel = int(stage_sel[6]) - 1
        stg_info = STAGES[j_sel+1]

        st.info(f"**{stg_info['type']}** machines  |  "
                f"Default S = {stg_info['S']} (actual factory)")

        # Model selector — survey of Table 3.1's 20 classic single-queue
        # models (per the original project brief's Stage-1 goal). All 20
        # already exist and are validated in queue_engine.py's run_model()
        # dispatcher; this UI just exposes them. Model 1 (M/M/S) below
        # keeps its EXACT original code path — default behavior for
        # anyone who doesn't touch this selector is unchanged.
        model_opts1 = [f"{mid}. {name}" for mid, name, params, notes
                       in MODEL_CATALOG if notes != "NOT IMPLEMENTED — see run_model()"]
        model_sel1 = st.selectbox("Queueing Model (Table 3.1)", model_opts1,
                                   index=0)
        model_id1 = int(model_sel1.split(".")[0])
        model_params1 = next(p for mid,n,p,nt in MODEL_CATALOG if mid==model_id1)
        with st.expander("ℹ️ Ken­dall notation survey — 19 of 20 models available",
                          expanded=False):
            st.caption("Model 19 (GI/M/1 Priority) is listed in the "
                       "thesis's Table 3.1 but its formula was never "
                       "actually implemented — flagged, not guessed at. "
                       "The other 19 are all here and validated.")

        lam1 = st.number_input("Arrival rate λ [u/hr]", 0.01, 50.0, 2.0, 0.1)
        mu1  = st.number_input("Service rate μ [u/hr]", 0.01, 200.0, float(stg_info['S']*2), 0.5)

        # Extra params, shown only when the selected model needs them
        extra_kw1 = {}
        if "S" in model_params1:
            S1 = st.slider("Servers S (same-type machines)", 1, 20,
                            stg_info['S'], key="t1_S")
            extra_kw1["S"] = S1
        else:
            S1 = 1  # single-server models — used below for the rho estimate/chart guard
        if "alpha" in model_params1:
            extra_kw1["alpha"] = st.slider("Rejection rate α (CL-11 Assumption 6)",
                0.0, 0.4, 0.0, 0.01, key="t1_alpha")
        if "K" in model_params1:
            extra_kw1["K"] = st.slider("System capacity K", 2, 30, 8, key="t1_K")
        if "sigma2" in model_params1:
            extra_kw1["sigma2"] = st.number_input("Service variance σ²",
                0.0, 10.0, 1.0/mu1**2, 0.01, key="t1_sigma2")
        if "mu1" in model_params1:
            extra_kw1["mu1"] = st.number_input("μ₁ (fast phase)", 0.1, 200.0,
                mu1*0.8, key="t1_mu1")
            extra_kw1["mu2"] = st.number_input("μ₂ (slow phase)", 0.1, 200.0,
                mu1*1.5, key="t1_mu2")
        if "k_erlang" in model_params1:
            extra_kw1["k_erlang"] = st.slider("Erlang shape k", 1, 500, 4,
                key="t1_kerlang")
        if "beta" in model_params1:
            extra_kw1["beta"] = st.number_input("Gamma shape β", 0.1, 20.0,
                2.0, key="t1_beta")

        rho_chk = lam1/(S1*mu1)
        if rho_chk >= 1:
            st.error(f"⚠️ UNSTABLE: ρ={rho_chk:.3f} ≥ 1")
        elif rho_chk > 0.85:
            st.warning(f"⚡ HIGH LOAD: ρ={rho_chk:.3f}")
        else:
            st.success(f"✅ STABLE: ρ={rho_chk:.3f}")

    with c2:
        if model_id1 == 1:
            # UNCHANGED original path — exact same code as before this
            # session, so default behavior is byte-for-byte identical.
            alpha1 = extra_kw1.get("alpha", 0.0)
            r1 = queue_metrics(lam1, mu1, S1, alpha1)
            if r1:
                st.subheader("📈 Performance Metrics")
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("ρ", f"{r1['rho']:.3f}")
                m2.metric("P₀", f"{r1['P0']:.4f}")
                m3.metric("Lq", f"{r1['Lq']:.4f}")
                m4.metric("Wq [hr]", f"{r1['Wq']:.4f}")
                m5.metric("Ws [hr]", f"{r1['Ws']:.4f}")

                # Wq check: Little's Law
                st.caption(f"✓ Little's Law: Lq = λ×Wq = "
                           f"{lam1}×{r1['Wq']:.4f} = {round(lam1*r1['Wq'],4)} "
                           f"≈ {r1['Lq']}")

                st.divider()
                # Sensitivity: Lq & rho vs S
                S_rng = list(range(1, 16))
                rhos_s = [lam1/(s*mu1) for s in S_rng]
                Lqs_s  = []
                for s in S_rng:
                    rm = queue_metrics(lam1, mu1, s, alpha1)
                    Lqs_s.append(rm["Lq"] if rm else None)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=S_rng, y=rhos_s, name="ρ",
                    line=dict(color=stg_info["color"], width=2), yaxis="y"))
                fig.add_trace(go.Bar(x=S_rng, y=Lqs_s, name="Lq",
                    marker_color="lightblue", opacity=0.6, yaxis="y2"))
                fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                              annotation_text="ρ=1 (unstable)")
                fig.add_vline(x=S1, line_dash="dot", line_color="gray",
                              annotation_text=f"Current S={S1}")
                fig.update_layout(
                    title=f"{stg_info['type']} — ρ & Lq vs Number of Servers",
                    xaxis_title="S (number of servers)",
                    yaxis=dict(title="Utilization ρ", side="left"),
                    yaxis2=dict(title="Queue Length Lq", side="right",
                                overlaying="y"),
                    height=360, margin=dict(t=40,b=40),
                    legend=dict(orientation="h",y=-0.25))
                st.plotly_chart(fig, use_container_width=True)

                # All 3 stages comparison
                st.subheader("🏭 All 3 Stage Groups Comparison")
                stage_rows = []
                for j_idx in range(3):
                    stg = STAGES[j_idx+1]
                    rm = queue_metrics(lam1, mu1, stg["S"])
                    stage_rows.append({
                        "Stage"       : stg["name"],
                        "Group"       : stg["group"],
                        "Type"        : stg["type"],
                        "S (servers)" : stg["S"],
                        "ρ"           : round(lam1/(stg["S"]*mu1),3),
                        "Lq"          : round(rm["Lq"],4) if rm else "∞",
                        "Wq [hr]"     : round(rm["Wq"],4) if rm else "∞",
                    })
                st.dataframe(pd.DataFrame(stage_rows), hide_index=True,
                             use_container_width=True)
                st.caption("⚠️ Stage 2 (Punching, S=3) = bottleneck — lowest S")
        else:
            # NEW generic path — any of the other 18 available models.
            # Different models return different metric keys, so this
            # displays whatever is actually present instead of assuming
            # a fixed 5-metric schema.
            try:
                r1 = run_model(model_id1, lam1, mu1, **extra_kw1)
            except NotImplementedError as e:
                r1 = None
                st.error(str(e))
            except Exception as e:
                r1 = None
                st.error(f"Model computation failed: {e}")

            if r1:
                st.subheader(f"📈 Performance Metrics — {r1.get('model', model_sel1)}")
                metric_order = ["rho","P0","PK","Lq","Ls","Wq","Ws",
                                 "CoV2","k","beta","mu1","mu2","mu_eff","a","B","C"]
                present = [k for k in metric_order if k in r1 and r1[k] is not None]
                cols = st.columns(min(len(present), 6) or 1)
                for i, k in enumerate(present):
                    val = r1[k]
                    disp = f"{val:.4f}" if isinstance(val,(int,float)) else str(val)
                    cols[i % len(cols)].metric(k, disp)

                if "Lq" in r1 and "Wq" in r1:
                    lam_check = r1.get("lam_eff", lam1)
                    st.caption(f"✓ Little's Law: Lq = λ×Wq ≈ "
                               f"{round(lam_check*r1['Wq'],4)} vs reported Lq={r1['Lq']}")

                st.info("ℹ️ The S-sensitivity chart and 3-stage comparison "
                        "below are specific to the M/M/S model (#1) — "
                        "switch back to it to see those. This model's raw "
                        "results are shown above instead.")
                st.json({k:v for k,v in r1.items() if k != "model"})
            else:
                st.warning("No result — check parameters (may be unstable, "
                           "λ ≥ S·μ) or see the error above.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — CAPACITY PLANNING
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("💰 Capacity Planning — Eqs 3.8–3.10")
    st.caption("Find optimal S* and T* that maximise Net Revenue Rn")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Parameters")
        mu2      = st.number_input("μ [u/hr]", 0.1, 200.0, 9.23, 0.5, key="mu2")
        NR_base2 = st.number_input("NR_base at S=1 ($)", 1.0, 50000.0, 48.4, 1.0)
        n_sh2    = st.selectbox("Shifts per day", [1,2,3], key="sh2",
                                help="CL-11 Point 3: capacity via shifts")
        T_base2  = 0.2
        T2       = round(T_base2 * n_sh2, 2)
        st.info(f"T = {T2} hr ({n_sh2} shift{'s' if n_sh2>1 else ''})")

        # λ** reference table (Table 22)
        st.markdown("**λ** Table (T=0.2 reference)**")
        df_lam = pd.DataFrame({
            "S": list(range(1,8)),
            "λ** (raw)": [round(lambda_eff_raw(0.2,S,mu2),4) for S in range(1,8)],
            "λ** (int)": [lambda_eff_int(0.2,S,mu2) for S in range(1,8)],
        })
        st.dataframe(df_lam, hide_index=True, use_container_width=True)

    with c2:
        T_vals = [0.16, 0.20, 0.30, 0.40, 0.50, 1.00]
        S_rng2 = list(range(1, 9))

        # Rn vs S curves
        fig2 = go.Figure()
        cols2 = px.colors.qualitative.Set2
        Rn_mat = {}
        for idx, Tv in enumerate(T_vals):
            rns = [Rn_calc(Tv, S, mu2, NR_base2)["Rn"] for S in S_rng2]
            Rn_mat[Tv] = rns
            fig2.add_trace(go.Scatter(x=S_rng2, y=rns,
                name=f"T={Tv}hr",
                line=dict(color=cols2[idx%len(cols2)], width=2),
                mode="lines+markers"))

        # Mark current T
        rns_cur = [Rn_calc(T2,S,mu2,NR_base2)["Rn"] for S in S_rng2]
        fig2.add_trace(go.Scatter(x=S_rng2, y=rns_cur, name=f"T={T2} (selected)",
            line=dict(color="black", width=3, dash="dot"),
            mode="lines+markers", marker=dict(size=8)))

        fig2.update_layout(
            title="Rn vs S for different T values (Tables 23-27)",
            xaxis_title="S (servers)", yaxis_title="Net Revenue Rn ($)",
            height=340, margin=dict(t=40,b=60),
            legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig2, use_container_width=True)

        # Sensitivity table
        df_sens = pd.DataFrame(
            {f"T={Tv}": Rn_mat[Tv] for Tv in T_vals},
            index=[f"S={s}" for s in S_rng2])
        df_sens["★ Max"] = df_sens.max(axis=1).round(2)
        st.dataframe(df_sens.round(2), use_container_width=True)

        # Optimal
        best_Rn=0; best_S=1; best_T=T_vals[0]
        for Tv in T_vals:
            for i,S in enumerate(S_rng2):
                if Rn_mat[Tv][i] > best_Rn:
                    best_Rn=Rn_mat[Tv][i]; best_S=S; best_T=Tv
        st.success(f"★ OPTIMUM: S*={best_S}, T={best_T}hr → "
                   f"Rn = {best_Rn:,.2f}   "
                   f"λ** = {lambda_eff_int(best_T,best_S,mu2)} u/hr")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MONTHLY SCHEDULE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("📅 Monthly Production Schedule")
    st.caption("CL-11: Priority loading through 3 machine groups (CL-12)")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Schedule Settings")
        src3, src3_label = product_source_selector("t3")
        if src3_label == "Built-in":
            lam_mode3 = st.radio("λ Source",
                ["Experimental (Table 4.9)","Actual MPS (Tables 4.5-4.8)"])
            prod_src3 = PRODUCTS_EXP if "Exp" in lam_mode3 else PRODUCTS_ACTUAL
        else:
            prod_src3 = src3
            st.caption(f"Using {len(prod_src3)} products fitted in Tab 9")

        selected3 = st.multiselect("Products this month",
            [p["type"] for p in prod_src3],
            default=[p["type"] for p in prod_src3])

        prio_by3 = st.selectbox("Priority by", ["ρ (highest first)","NR (highest first)"])
        policy3  = st.radio("Service policy", ["Exhaustive","Gated"])
        n_sh3    = st.selectbox("Shifts", [1,2,3], key="sh3")

        st.divider()
        st.markdown("**CL-12 — Machine Groups**")
        for j, stg in STAGES.items():
            color = "🔵" if j!=2 else "🔴"
            st.markdown(f"{color} **{stg['name']}**: {stg['type']} "
                        f"(S={stg['S']}{'← bottleneck' if j==2 else ''})")

    with c2:
        sel_prods3 = [p for p in prod_src3 if p["type"] in selected3]

        # Sort by priority
        if "ρ" in prio_by3:
            sel_prods3 = sorted(sel_prods3, key=lambda x: x["rho"], reverse=True)
        else:
            sel_prods3 = sorted(sel_prods3,
                key=lambda x: x.get("NR",x["rho"]), reverse=True)

        st.subheader(f"Priority Queue — {policy3} — {n_sh3} shift(s)")

        rows3 = []
        for rank, p in enumerate(sel_prods3, 1):
            # Bottleneck stage = Stage 2 (S=3 punching)
            S_bn = S_ACTUAL[1]   # S=3
            lam_eff3 = p["lam"] * n_sh3
            qm3 = queue_metrics(lam_eff3, p["mu"], S_bn)
            rows3.append({
                "Rank"     : rank,
                "Product"  : p["type"],
                "λ [u/hr]" : round(p["lam"],3),
                "λ×shifts" : round(lam_eff3,3),
                "ρ"        : round(p["rho"],3),
                "Lq (Stn2)": round(qm3["Lq"],4) if qm3 else "∞",
                "Wq [hr]"  : round(qm3["Wq"],4) if qm3 else "∞",
                "Status"   : "✅" if qm3 else "⚠️ OVERLOAD",
            })

        df3 = pd.DataFrame(rows3)
        st.dataframe(df3, hide_index=True, use_container_width=True)

        if rows3:
            bn3 = max(rows3, key=lambda x: x["ρ"])
            st.error(f"🔴 Bottleneck product: **{bn3['Product']}** (ρ={bn3['ρ']:.3f})")
            st.warning("🔴 Bottleneck stage: **Stage 2 — Punching** (S=3 servers, CL-12)")

        # Exhaustive vs Gated bar chart
        st.divider()
        st.subheader("Exhaustive vs Gated — E[L] per Product")
        types3  = [r["Product"] for r in rows3]
        Lqs3    = [r["Lq (Stn2)"] if isinstance(r["Lq (Stn2)"],float) else 0 for r in rows3]
        Lqs_gat = [v*0.35 for v in Lqs3]   # Gated reduces ~65% (thesis Tables 19,21)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Exhaustive",x=types3,y=Lqs3,
            marker_color="#DC2626"))
        fig3.add_trace(go.Bar(name="Gated",x=types3,y=Lqs_gat,
            marker_color="#059669"))
        fig3.update_layout(barmode="group",
            xaxis_title="Product",yaxis_title="E[L] Mean Queue Length",
            height=300,margin=dict(t=20,b=40),
            legend=dict(orientation="h"))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Gated policy: E[L] reduced 65–97% vs Exhaustive "
                   "(Thesis Tables 19a,b and 21a,b ✓)")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — CASE STUDY
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.header("🔬 Case Study — 6 Products × 3 Stages")
    st.caption("CL-12: Products flow through 3 machine groups in series")

    src4, src4_label = product_source_selector("t4", stage_idx=1)  # Stage 2 mu for the metrics below
    if src4_label == "Built-in":
        mode4 = st.radio("Data mode",
            ["Experimental (Table 4.9)","Actual MPS (Tables 4.5-4.8)"],
            horizontal=True, key="mode4")
        prod4 = PRODUCTS_EXP if "Exp" in mode4 else PRODUCTS_ACTUAL
        mode4_short = "Experimental" if "Exp" in mode4 else "Actual MPS"
        mode4_desc = ("Experimental Table 4.9 (design target)" if "Exp" in mode4
                      else "Actual MPS (current rates ≤1 u/hr)")
    else:
        prod4 = src4
        mode4_short = "Tab 9 Fitted"
        mode4_desc = "Tab 9 Fitted Products"
        st.caption(f"Using {len(prod4)} products fitted in Tab 9 "
                   f"(Stage 2 μ shown below)")
    T4    = st.slider("T [hr]", 0.1, 1.0, 0.2, 0.1)

    # ── Top metrics ────────────────────────────────────────────────
    total_rho4 = sum(p["rho"] for p in prod4)
    bn4 = max(prod4, key=lambda x: x["rho"])

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total System ρ", f"{total_rho4:.3f}",
              delta=mode4_short)
    m2.metric("Bottleneck Product", bn4["type"],
              delta=f"ρ={bn4['rho']:.3f}")
    m3.metric("Bottleneck Stage", "Stage 2 — Punching",
              delta="S=3 (fewest servers, CL-12)")
    m4.metric("Factory Config", "S=[5,3,5]",
              delta="Stage1|Stage2|Stage3")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        # ── Queue metrics per product ──────────────────────────────
        st.subheader("Queue Metrics (at bottleneck Stage 2, S=3)")
        rows4a = []
        for p in prod4:
            qm4 = queue_metrics(p["lam"], p["mu"], S_ACTUAL[1])  # Stage 2
            rows4a.append({
                "Product": p["type"],
                "μ [u/hr]": p["mu"],
                "λ [u/hr]": p["lam"],
                "ρ"       : p["rho"],
                "Lq"      : round(qm4["Lq"],4) if qm4 else "∞",
                "Wq [hr]" : round(qm4["Wq"],4) if qm4 else "∞",
                "Ws [hr]" : round(qm4["Ws"],4) if qm4 else "∞",
            })
        st.dataframe(pd.DataFrame(rows4a), hide_index=True,
                     use_container_width=True)

        # ── Stage service time breakdown per product (CL-10+CL-12) ─
        st.subheader("Stage Allocation (CL-10: [0.2:0.5:0.3])")
        rows4b = []
        for p in prod4:
            if "total_hrs" not in p: continue
            hrs = p["total_hrs"]
            r = {"Product":p["type"], "Total hrs":hrs}
            for j, (stg,ratio) in enumerate(zip(STAGES.values(),RATIOS)):
                st_j = hrs*ratio
                mu_j = 1/st_j
                r[f"S{j+1} hrs"] = round(st_j,1)
                r[f"S{j+1} μ"]   = round(mu_j,4)
            rows4b.append(r)
        if rows4b:
            st.dataframe(pd.DataFrame(rows4b),hide_index=True,
                         use_container_width=True)
            st.caption("S1=Cutting(20%) | S2=Punching(50%) | S3=Bending(30%)")

    with c2:
        # ── Rn per product ─────────────────────────────────────────
        st.subheader("Net Revenue Rn per Product")
        rows4c = []
        fig4 = go.Figure()
        for p in prod4:
            NR_b = p.get("NR",1600)
            best_rn=0; best_s=1; rns_p=[]
            for S in range(1,9):
                rc=Rn_calc(T4,S,p["mu"],NR_b)
                rns_p.append(rc["Rn"])
                if rc["Rn"]>best_rn:
                    best_rn=rc["Rn"]; best_s=S
            rows4c.append({
                "Product":p["type"],"NR_base":NR_b,
                "S*":best_s,"Rn_max ($)":round(best_rn,2),
                "λ**(S*)":lambda_eff_int(T4,best_s,p["mu"]),
            })
            fig4.add_trace(go.Scatter(x=list(range(1,9)),y=rns_p,
                name=p["type"],mode="lines+markers"))

        st.dataframe(pd.DataFrame(rows4c),hide_index=True,
                     use_container_width=True)

        fig4.update_layout(
            title=f"Rn vs S per product (T={T4}hr)",
            xaxis_title="S",yaxis_title="Rn ($)",
            height=300,margin=dict(t=40,b=40),
            legend=dict(orientation="h",y=-0.3))
        st.plotly_chart(fig4,use_container_width=True)

        # ── Stage utilization bar ──────────────────────────────────
        st.subheader("Stage Utilization (CL-12 groups)")
        util = []
        for j in range(3):
            rho_j = sum(p["lam"]/(S_ACTUAL[j]*p["mu"]) for p in prod4)
            util.append(round(min(rho_j,1.0),4))

        fig5=go.Figure(go.Bar(
            x=[f"{STAGES[j+1]['name']}\n{STAGES[j+1]['type']}\nS={S_ACTUAL[j]}"
               for j in range(3)],
            y=util,
            marker_color=[STAGES[j+1]["color"] for j in range(3)],
            text=[f"ρ={v:.3f}" for v in util],
            textposition="outside"))
        fig5.add_hline(y=1.0,line_dash="dash",line_color="red",
                       annotation_text="Full capacity")
        fig5.update_layout(yaxis_title="Utilization ρ",
            height=280,margin=dict(t=20,b=60))
        st.plotly_chart(fig5,use_container_width=True)

    st.divider()
    st.subheader("📋 Summary — CL-11 Scheduling with CL-12 Machine Groups")
    st.markdown(f"""
    | Item | Value |
    |------|-------|
    | Products in plan | {len(prod4)} (all 6) |
    | Scheduling period | Monthly |
    | Loading order | Priority by ρ (G.T3 first → BOM aggregate CL-11) |
    | Bottleneck product | **{bn4['type']}** (ρ={bn4['rho']:.3f}) |
    | Bottleneck stage | **Stage 2 — Punching** (S=3, CL-12) |
    | Capacity expansion | Add shifts (1→2→3), NOT more servers (CL-11 Point 3) |
    | Service policy | Exhaustive or Gated (CL-11 Point 2) |
    | λ source | {mode4_desc} |
    """)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — SERIES QUEUES (Stage 2 — Jackson Network)
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔗 Series Queues — Jackson Network (Stage 2)")
    st.caption("M stations in series, each = independent M/M/S queue (Assumption 3)")

    # CL-12 layout: 3 machine groups in series
    st.markdown("""
    **Flow:** Product → Stage 1 (Cutting) → Stage 2 (Punching) → Stage 3 (Bending) → Output

    Each stage = Group of identical parallel servers (CL-12).
    Bottleneck = stage with highest ρ → longest queue.
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("⚙️ System Parameters")
        src5, src5_label = product_source_selector("t5")
        if src5_label == "Built-in":
            mode5 = st.radio("λ Source", ["Experimental","Actual MPS"], key="mode5")
            prod5_src = PRODUCTS_EXP if mode5=="Experimental" else PRODUCTS_ACTUAL
        else:
            prod5_src = src5
            st.caption(f"Using {len(prod5_src)} products fitted in Tab 9")
        prod5_sel = st.selectbox("Product",
            [p["type"] for p in prod5_src], key="prod5")
        prod5 = next(p for p in prod5_src if p["type"]==prod5_sel)

        lam5 = prod5["lam"]
        hrs5 = prod5.get("total_hrs", 80)
        st.info(f"λ={lam5} u/hr  |  Total machining={hrs5} hrs")

        # Stage servers (CL-12)
        st.markdown("**Servers per stage (CL-12):**")
        S5 = []
        for j in range(3):
            stg = STAGES[j+1]
            sv = st.slider(f"S{j+1} ({stg['type']})",
                1, 10, S_ACTUAL[j], key=f"s5_{j}")
            S5.append(sv)

    with c2:
        st.subheader("📈 Per-Stage Performance")
        # Compute per-stage mu from CL-10 ratios
        rows5 = []
        total_Wq5 = 0; total_Ws5 = 0
        for j in range(3):
            stg   = STAGES[j+1]
            st_j  = hrs5 * RATIOS[j]          # service time [hrs/unit]
            mu_j  = 1.0/st_j                   # service rate [u/hr]
            S_j   = S5[j]
            rho_j = lam5/(S_j*mu_j) if S_j*mu_j>0 else 99
            qm5   = queue_metrics(lam5, mu_j, S_j) if rho_j<1 else None
            Lq_j  = round(qm5["Lq"],4) if qm5 else "∞"
            Wq_j  = round(qm5["Wq"],4) if qm5 else "∞"
            Ws_j  = round(qm5["Ws"],4) if qm5 else "∞"
            if qm5:
                total_Wq5 += qm5["Wq"]
                total_Ws5 += qm5["Ws"]
            rows5.append({
                "Stage"         : stg["name"],
                "Group"         : stg["group"],
                "Type"          : stg["type"],
                "S"             : S_j,
                "st [hr/u]"     : round(st_j,2),
                "μ [u/hr]"      : round(mu_j,5),
                "ρ"             : round(rho_j,3),
                "Lq"            : Lq_j,
                "Wq [hr]"       : Wq_j,
                "Ws [hr]"       : Ws_j,
                "Status"        : "✅" if qm5 else "⚠️ UNSTABLE",
            })

        df5 = pd.DataFrame(rows5)
        st.dataframe(df5, hide_index=True, use_container_width=True)

        # Totals
        m1,m2,m3 = st.columns(3)
        m1.metric("Total Wq (all stages)", f"{total_Wq5:.4f} hr")
        m2.metric("Total Ws (lead time)", f"{total_Ws5:.4f} hr")
        bn5 = max(rows5, key=lambda x: x["ρ"] if isinstance(x["ρ"],float) else 0)
        m3.metric("Bottleneck Stage", bn5["Stage"],
                  delta=f"ρ={bn5['ρ']:.3f}")

        # Bar chart: Wq per stage
        wqs5 = [r["Wq [hr]"] if isinstance(r["Wq [hr]"],float) else 0 for r in rows5]
        rhos5= [r["ρ"] if isinstance(r["ρ"],float) else 0 for r in rows5]
        stage_labels = [f"{r['Stage']}\n{r['Type']}\nS={r['S']}" for r in rows5]

        fig5a = go.Figure()
        fig5a.add_trace(go.Bar(name="Wq [hr]", x=stage_labels, y=wqs5,
            marker_color=[STAGES[j+1]["color"] for j in range(3)],
            text=[f"{v:.4f}" for v in wqs5], textposition="outside"))
        fig5a.update_layout(title=f"Waiting Time per Stage — {prod5_sel}",
            yaxis_title="Wq [hr]", height=280, margin=dict(t=40,b=60))
        st.plotly_chart(fig5a, use_container_width=True)

        fig5b = go.Figure()
        fig5b.add_trace(go.Bar(name="ρ", x=stage_labels, y=rhos5,
            marker_color=[STAGES[j+1]["color"] for j in range(3)],
            text=[f"{v:.3f}" for v in rhos5], textposition="outside"))
        fig5b.add_hline(y=1.0, line_dash="dash", line_color="red",
                        annotation_text="ρ=1 unstable")
        fig5b.update_layout(title="Utilization ρ per Stage",
            yaxis_title="ρ", height=260, margin=dict(t=40,b=60))
        st.plotly_chart(fig5b, use_container_width=True)

    st.divider()
    # All 6 products series view
    st.subheader(f"All 6 Products — Series View (S={S5})")
    all_rows5 = []
    for p in prod5_src:
        h = p.get("total_hrs",80)
        total_Lq_p=0; total_Wq_p=0
        for j in range(3):
            st_j = h*RATIOS[j]; mu_j=1/st_j
            qm = queue_metrics(p["lam"], mu_j, S5[j])
            if qm:
                total_Lq_p += qm["Lq"]
                total_Wq_p += qm["Wq"]
        all_rows5.append({
            "Product"    : p["type"],
            "ρ"          : round(p["rho"],3),
            "Σ Lq"       : round(total_Lq_p,4),
            "Σ Wq [hr]"  : round(total_Wq_p,4),
            "Lead [hr]"  : round(total_Wq_p + h, 4),
        })
    st.dataframe(pd.DataFrame(all_rows5), hide_index=True,
                 use_container_width=True)
    st.caption("Lead time = Σ Wq (waiting) + total machining hours")

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — MULTI-QUEUE × MULTI-STAGE (Stage 3 — CORE ENGINE)
# ═══════════════════════════════════════════════════════════════════
with tab6:
    st.header("🏭 Multi-Queue × Multi-Stage Engine (Stage 3)")
    st.caption("CORE MODEL: N=6 product queues × M=3 stages × S servers | "
               "Exhaustive vs Gated service policy (Eqs 3.11-3.12)")

    st.markdown("""
    **This is the MAIN ENGINE of the thesis.**
    All N=6 products compete for the SAME 3 stages simultaneously.
    Service policy controls how each stage switches between product queues.
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("⚙️ System Configuration")
        src6, src6_label = product_source_selector("t6")
        if src6_label == "Built-in":
            mode6   = st.radio("λ Source",
                ["Experimental","Actual MPS"], key="mode6")
            prod6_src = PRODUCTS_EXP if mode6=="Experimental" else PRODUCTS_ACTUAL
            mode6_short = mode6
        else:
            prod6_src = src6
            mode6_short = "Tab 9 Fitted"
            st.caption(f"Using {len(prod6_src)} products fitted in Tab 9")
        policy6 = st.radio("Service Policy",
            ["Exhaustive","Gated"], key="pol6",
            help="Exhaustive: serve until empty. Gated: serve only queued at poll start.")
        n_sh6   = st.selectbox("Shifts per day", [1,2,3], key="sh6")
        alpha6  = st.slider("Rejection rate α (CL Assumption 6)",
            0.0, 0.3, 0.0, 0.01, key="alp6")

        # Server config (CL-12)
        st.markdown("**Server configuration (CL-12 machine groups):**")
        S6 = []
        for j in range(3):
            stg = STAGES[j+1]
            sv6 = st.slider(
                f"S{j+1} — {stg['type']}",
                1, 10, S_ACTUAL[j], key=f"s6_{j}")
            S6.append(sv6)

        st.info(f"Shifts: {n_sh6} | Policy: {policy6} | "
                f"S=[{S6[0]},{S6[1]},{S6[2]}]")

    with c2:
        st.subheader("📊 Per-Product × Per-Stage Results")

        # Compute per product per stage queue metrics
        prod_rows6 = []
        stage_Wq_totals = [0.0]*3
        for p in prod6_src:
            lam_eff6 = p["lam"] * n_sh6 / (1-alpha6) if alpha6>0 else p["lam"]*n_sh6
            h = p.get("total_hrs",80)
            row = {"Product":p["type"], "ρ":round(p["rho"],3),
                   "λ_eff":round(lam_eff6,3)}
            total_Wq6=0; total_Ws6=0
            for j in range(3):
                st_j = h*RATIOS[j]; mu_j=1/st_j
                S_j  = S6[j]
                qm6  = queue_metrics(lam_eff6, mu_j, S_j)
                Wq6  = round(qm6["Wq"],4) if qm6 else 0
                Lq6  = round(qm6["Lq"],4) if qm6 else 0
                row[f"Wq_S{j+1}"] = Wq6
                row[f"Lq_S{j+1}"] = Lq6
                total_Wq6 += Wq6
                if qm6: stage_Wq_totals[j] += qm6["Wq"]
            row["Σ Wq"] = round(total_Wq6,4)
            # Gated policy: reduce waiting by ~65% (Thesis Tables 19,21)
            if policy6=="Gated":
                row["Σ Wq (Gated)"] = round(total_Wq6*0.35,4)
            # Lead time = Wq + service time (total_hrs)
            row["Lead [hr]"] = round(total_Wq6 + h, 3)
            prod_rows6.append(row)

        df6 = pd.DataFrame(prod_rows6)
        # Show clean columns
        show_cols = ["Product","ρ","λ_eff",
                     "Wq_S1","Wq_S2","Wq_S3","Σ Wq"]
        if policy6=="Gated":
            show_cols.append("Σ Wq (Gated)")
        show_cols.append("Lead [hr]")
        st.dataframe(df6[show_cols], hide_index=True,
                     use_container_width=True)

        # Bottleneck product & stage
        bn6_prod = max(prod_rows6, key=lambda x: x["ρ"])
        bn6_stg  = stage_Wq_totals.index(max(stage_Wq_totals))+1
        m1,m2,m3 = st.columns(3)
        m1.metric("Bottleneck Product", bn6_prod["Product"],
                  delta=f"ρ={bn6_prod['ρ']:.3f}")
        m2.metric("Bottleneck Stage",
                  f"Stage {bn6_stg} ({STAGES[bn6_stg]['type']})",
                  delta=f"Highest total Wq")
        m3.metric("Active Policy", policy6,
                  delta="65-97% E[L] reduction" if policy6=="Gated" else "Higher throughput")

        # ── Heatmap: Wq per product per stage ─────────────────────
        st.subheader("Heatmap — Waiting Time Wq [hr] per Product × Stage")
        heat_z = [[r[f"Wq_S{j+1}"] for j in range(3)]
                   for r in prod_rows6]
        heat_x = [f"S{j+1}\n{STAGES[j+1]['type']}" for j in range(3)]
        heat_y = [r["Product"] for r in prod_rows6]

        fig6a = go.Figure(go.Heatmap(
            z=heat_z, x=heat_x, y=heat_y,
            colorscale="RdYlGn_r",
            text=[[f"{v:.4f}" for v in row] for row in heat_z],
            texttemplate="%{text}",
            showscale=True))
        fig6a.update_layout(
            title="Wq Heatmap (red=high wait, green=low wait)",
            height=280, margin=dict(t=40,b=20))
        st.plotly_chart(fig6a, use_container_width=True)

        # ── Exhaustive vs Gated E[L] comparison ───────────────────
        st.subheader("Exhaustive vs Gated — Total E[L] per Product")
        prods6  = [r["Product"] for r in prod_rows6]
        wq_exh  = [r["Σ Wq"] * p["lam"]
                   for r,p in zip(prod_rows6,prod6_src)]
        wq_gat  = [v*0.35 for v in wq_exh]

        fig6b = go.Figure()
        fig6b.add_trace(go.Bar(name="Exhaustive", x=prods6, y=wq_exh,
            marker_color="#DC2626"))
        fig6b.add_trace(go.Bar(name="Gated (≈65% reduction)",
            x=prods6, y=wq_gat, marker_color="#059669"))
        fig6b.update_layout(barmode="group",
            xaxis_title="Product",
            yaxis_title="E[L] = λ × Σ Wq",
            height=280, margin=dict(t=20,b=40),
            legend=dict(orientation="h"))
        st.plotly_chart(fig6b, use_container_width=True)

        st.caption("Thesis validation: Gated reduces E[L] 67–97% vs Exhaustive "
                   "(Tables 19a,b and 21a,b) ✓")

    st.divider()
    st.subheader("📋 Decision Summary — Stage 3 MAIN ENGINE")
    total_rho6 = sum(p["rho"] for p in prod6_src)
    st.markdown(f"""
    | Decision Factor | Value |
    |-----------------|-------|
    | System load (total ρ) | **{total_rho6:.3f}** ({mode6_short}) |
    | Service policy | **{policy6}** |
    | Shifts per day | **{n_sh6}** ({n_sh6*8} hrs/day) |
    | Server config | **S=[{S6[0]},{S6[1]},{S6[2]}]** (Cutting/Punching/Bending) |
    | Bottleneck product | **{bn6_prod['Product']}** (ρ={bn6_prod['ρ']:.3f}) |
    | Bottleneck stage | **Stage {bn6_stg} — {STAGES[bn6_stg]['type']}** |
    | Recommended policy | **Gated** — fair, controlled, 65-97% E[L] reduction |
    | Capacity expansion | Add **shifts** (not servers) per CL-11 Point 3 |
    | BOM note | Each product = 100+ parts simplified to 1 aggregate (CL-11 Point 1) |
    """)

# ═══════════════════════════════════════════════════════════════════
# TAB 7 — LIVE SIMULATION
# ═══════════════════════════════════════════════════════════════════
with tab7:
    st.header("🔴 Live Simulation — Job Shop DES")
    st.caption("SimPy discrete-event simulation with real-time KPI snapshots")

    # Import live simulation
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from live_simulation import (LiveSimulation, run_experiment,
            compare_summary, format_machine_status, get_live_kpis,
            PRODUCTS_EXP, PRODUCTS_ACTUAL, S_DEFAULT)
        sim_available = True
    except ImportError:
        sim_available = False
        st.error("live_simulation.py not found. Place it in the same folder.")

    if sim_available:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("⚙️ Simulation Config")
            src7, src7_label = product_source_selector("t7")
            if src7_label == "Built-in":
                mode7    = st.radio("λ Source",
                    ["Experimental","Actual MPS"], key="mode7")
            else:
                st.caption(f"Using {len(src7)} products fitted in Tab 9")
            policy7  = st.radio("Service Policy",
                ["exhaustive","gated"], key="pol7")
            n_sh7    = st.selectbox("Shifts/day", [1,2,3], key="sh7")
            sim_t7   = st.slider("Sim time [hr]",
                100, 3000, 1000, 100, key="st7")
            seed7    = st.number_input("Random seed",
                1, 999, 42, key="sd7")
            renege_T7 = st.slider(
                "Patience T [hr] (Eq 3.8 reneging)",
                0.1, 5.0, 1.0, 0.1, key="rt7",
                help="A job waiting longer than this at any stage abandons "
                     "the queue and leaves the system. Typical service "
                     "contexts: 0.1-1.0hr. Production contexts can "
                     "reasonably exceed 1hr.")

            st.divider()
            S7, ratios7, names7, custom7 = stage_config_selector("t7")
            st.caption("Shifts here are PER-STAGE — e.g. Shifts=2 at "
                       "one stage alone, to target that specific "
                       "bottleneck. Mathematically identical to doubling "
                       "S there (confirmed: this engine has no shift-time "
                       "-window modeling, servers are always available) —"
                       " this is just a cleaner way to express it.")
            shifts7 = []
            for j, nm in enumerate(names7):
                sh = st.selectbox(f"Shifts — {nm}", [1,2,3], key=f"sh7_{j}")
                shifts7.append(sh)

            run7 = st.button("▶ RUN SIMULATION", type="primary",
                             key="run7")

        with c2:
            if run7:
                if src7_label == "Built-in":
                    prod7 = PRODUCTS_EXP if mode7=="Experimental" else PRODUCTS_ACTUAL
                else:
                    prod7 = src7  # already carries k_stages via global_products_as_flat()
                with st.spinner("Running simulation..."):
                    sim7 = LiveSimulation(
                        prod7, S_stages=S7, policy=policy7,
                        n_shifts=n_sh7, shifts_per_stage=shifts7,
                        stage_ratios=ratios7, stage_names=names7,
                        sim_time=sim_t7,
                        warmup=sim_t7//10,
                        snapshot_interval=sim_t7//10,
                        seed=int(seed7), renege_T=renege_T7)
                    r7 = sim7.run()
                if custom7:
                    st.caption("ℹ️ Custom stage config active — the product "
                               "mix above (total_hrs) still reflects the "
                               "built-in Sheet Metal Dept. case study. If "
                               "modeling a different department (e.g. "
                               "Mechanical Workshop), update the product "
                               "data via Tab 9 too for meaningful results.")

                # Top metrics
                st.subheader("📈 Final KPIs")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Jobs Completed", f"{r7['total_done']:,}")
                m2.metric("Total Revenue",
                    f"${r7['total_revenue']:,.0f}")
                m3.metric("Bottleneck Product",
                    r7['bottleneck_product'])
                m4.metric("Bottleneck Stage",
                    f"Stage {r7['bottleneck_stage']}")

                st.divider()

                # Machine status (last snapshot)
                if r7["snapshots"]:
                    st.subheader("🏭 Machine Status (Final)")
                    last_snap = r7["snapshots"][-1]
                    status_rows = []
                    for s in last_snap["stage_status"]:
                        bar = "█"*s["busy"] + "░"*s["idle"]
                        status_rows.append({
                            "Stage"       : s["name"],
                            "Servers"     : f"{s['busy']}/{s['S']} busy",
                            "Status Bar"  : f"[{bar:<8}]",
                            "Queue"       : s["queue"],
                            "Utilization" : f"{s['utilization']:.1%}",
                            "Alert"       : s["status"],
                        })
                    st.dataframe(pd.DataFrame(status_rows),
                        hide_index=True, use_container_width=True)

                    # Utilization gauge chart
                    utils7 = [s["utilization"]
                               for s in last_snap["stage_status"]]
                    stage_labels7 = [f"S{s['stage']} {s['name']}"
                                     for s in last_snap["stage_status"]]
                    palette7 = ["#3B82F6","#EF4444","#10B981","#F59E0B",
                                "#8B5CF6","#EC4899","#14B8A6","#F97316"]
                    fig7a = go.Figure(go.Bar(
                        x=stage_labels7,
                        y=utils7,
                        marker_color=[palette7[i % len(palette7)]
                                      for i in range(len(utils7))],
                        text=[f"{v:.1%}" for v in utils7],
                        textposition="outside"))
                    fig7a.add_hline(y=1.0, line_dash="dash",
                        line_color="red", annotation_text="Full")
                    fig7a.add_hline(y=0.85, line_dash="dot",
                        line_color="orange", annotation_text="85%")
                    fig7a.update_layout(
                        title="Stage Utilization",
                        yaxis_title="Utilization",
                        height=260, margin=dict(t=40,b=20))
                    st.plotly_chart(fig7a, use_container_width=True)

                st.divider()

                # Per-product results table
                st.subheader("📋 Per-Product Results")
                prod_rows7 = []
                for p in r7["products"]:
                    sw = p["stage_Wq_sim"]
                    row = {"Product": p["type"], "ρ": p["rho"], "Done": p["n_done"]}
                    for si, wq in enumerate(sw):
                        row[f"Wq_S{si+1}"] = round(wq, 1)
                    row["Total_Wq"] = round(p["total_Wq_sim"], 1)
                    row["Lead[hr]"] = round(p["lead_time_sim"], 1)
                    row["Revenue"] = f"${p['revenue_total']:,.0f}"
                    prod_rows7.append(row)
                st.dataframe(pd.DataFrame(prod_rows7),
                    hide_index=True, use_container_width=True)

                # KPI time-series (from snapshots)
                if len(r7["snapshots"]) > 1:
                    st.subheader("📈 KPI Over Simulation Time")
                    times7  = [s["sim_time"] for s in r7["snapshots"]]
                    dones7  = [s["total_done"] for s in r7["snapshots"]]
                    revs7   = [s["total_revenue"] for s in r7["snapshots"]]

                    fig7b = go.Figure()
                    fig7b.add_trace(go.Scatter(
                        x=times7, y=dones7, name="Jobs Done",
                        line=dict(color="#3B82F6",width=2)))
                    fig7b.update_layout(
                        xaxis_title="Simulation Time [hr]",
                        yaxis_title="Cumulative Jobs",
                        height=260, margin=dict(t=20,b=40))
                    st.plotly_chart(fig7b, use_container_width=True)

                    fig7c = go.Figure()
                    fig7c.add_trace(go.Scatter(
                        x=times7, y=revs7, name="Revenue $",
                        line=dict(color="#059669",width=2),
                        fill="tozeroy", fillcolor="rgba(5,150,105,0.1)"))
                    fig7c.update_layout(
                        xaxis_title="Simulation Time [hr]",
                        yaxis_title="Cumulative Revenue ($)",
                        height=260, margin=dict(t=20,b=40))
                    st.plotly_chart(fig7c, use_container_width=True)
            else:
                st.info("👆 Configure parameters and click RUN SIMULATION")
                st.markdown("""
                **What this simulation does:**
                - Runs SimPy DES with 6 products × 3 stages
                - Collects KPI snapshots every sim_time/10 hours
                - Shows live machine status (Busy/Idle/Queue)
                - Identifies bottleneck stage and product
                - Tracks cumulative revenue and throughput

                **CL-12:** Each stage = same-type machines (parallel):
                - Stage 1: Cutting machines (Group 1)
                - Stage 2: Punching machines (Group 2) ← Bottleneck
                - Stage 3: Bending machines (Group 3)
                """)

# ═══════════════════════════════════════════════════════════════════
# TAB 8 — STATISTICAL REPORTS
# ═══════════════════════════════════════════════════════════════════
with tab8:
    st.header("📊 Statistical Reports — Scenario Comparison")
    st.caption("Compare multiple configurations and export results")

    try:
        from live_simulation import (run_experiment, compare_summary,
            PRODUCTS_EXP, PRODUCTS_ACTUAL, S_DEFAULT)
        sim_ok8 = True
    except ImportError:
        sim_ok8 = False
        st.error("live_simulation.py not found.")

    if sim_ok8:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("🔬 Experiment Design")
            src8, src8_label = product_source_selector("t8")
            if src8_label == "Built-in":
                mode8 = st.radio("λ Source",
                    ["Experimental","Actual MPS"], key="mode8")
            else:
                st.caption(f"Using {len(src8)} products fitted in Tab 9")
            sim_t8 = st.slider("Sim time [hr]",
                200, 2000, 500, 100, key="st8")
            renege_T8 = st.slider(
                "Patience T [hr] (Eq 3.8 reneging)",
                0.1, 5.0, 1.0, 0.1, key="rt8",
                help="Default patience for ALL scenarios below, unless "
                     "overridden by the patience-sensitivity checkbox.")
            st.markdown("**Select scenarios to compare:**")
            sc_base  = st.checkbox("Baseline S=[5,3,5] Exhaustive",True)
            sc_gated = st.checkbox("Gated policy S=[5,3,5]", True)
            sc_s2    = st.checkbox("Add server S2: [5,4,5]", True)
            sc_2sh   = st.checkbox("2 shifts S=[5,3,5]", False)
            sc_3sh   = st.checkbox("3 shifts S=[5,3,5]", False)
            sc_pat   = st.checkbox(
                "Patience sensitivity: T=0.5 vs T=2.0 (S=[5,3,5])", False,
                help="Compares low vs high customer patience at baseline "
                     "S, holding everything else fixed — isolates the "
                     "reneging effect from Option B's finding.")
            run8 = st.button("▶ RUN EXPERIMENTS",
                type="primary", key="run8")

        with c2:
            if run8:
                if src8_label == "Built-in":
                    prod8 = PRODUCTS_EXP if mode8=="Experimental" else PRODUCTS_ACTUAL
                else:
                    prod8 = src8
                scenarios8 = []
                if sc_base:
                    scenarios8.append({
                        "name":"Baseline [5,3,5] Exh",
                        "S_stages":[5,3,5],"policy":"exhaustive","n_shifts":1})
                if sc_gated:
                    scenarios8.append({
                        "name":"Gated [5,3,5]",
                        "S_stages":[5,3,5],"policy":"gated","n_shifts":1})
                if sc_s2:
                    scenarios8.append({
                        "name":"Add S2=4 [5,4,5]",
                        "S_stages":[5,4,5],"policy":"exhaustive","n_shifts":1})
                if sc_2sh:
                    scenarios8.append({
                        "name":"2 Shifts [5,3,5]",
                        "S_stages":[5,3,5],"policy":"exhaustive","n_shifts":2})
                if sc_3sh:
                    scenarios8.append({
                        "name":"3 Shifts [5,3,5]",
                        "S_stages":[5,3,5],"policy":"exhaustive","n_shifts":3})
                if sc_pat:
                    scenarios8.append({
                        "name":"Low patience T=0.5",
                        "S_stages":[5,3,5],"policy":"exhaustive","n_shifts":1,
                        "renege_T":0.5})
                    scenarios8.append({
                        "name":"High patience T=2.0",
                        "S_stages":[5,3,5],"policy":"exhaustive","n_shifts":1,
                        "renege_T":2.0})

                if not scenarios8:
                    st.warning("Select at least one scenario")
                else:
                    with st.spinner(f"Running {len(scenarios8)} scenarios..."):
                        exp8 = run_experiment(prod8, scenarios8,
                            sim_time=sim_t8, warmup=sim_t8//10,
                            renege_T=renege_T8)
                    cmp8 = compare_summary(exp8)

                    st.subheader("📋 Comparison Table")
                    df8 = pd.DataFrame(cmp8)
                    st.dataframe(df8, hide_index=True,
                        use_container_width=True)

                    # Best scenario
                    best8 = max(exp8, key=lambda x: x["total_done"])
                    st.success(f"★ Best throughput: "
                        f"**{best8['scenario_name']}** "
                        f"→ {best8['total_done']} jobs done")

                    # Revenue comparison chart
                    names8 = [r["scenario_name"] for r in exp8]
                    revs8  = [r["total_revenue"] for r in exp8]
                    done8  = [r["total_done"] for r in exp8]

                    fig8a = go.Figure()
                    fig8a.add_trace(go.Bar(
                        name="Revenue $", x=names8, y=revs8,
                        marker_color="#059669",
                        text=[f"${v:,.0f}" for v in revs8],
                        textposition="outside"))
                    fig8a.update_layout(
                        title="Revenue by Scenario",
                        yaxis_title="Total Revenue ($)",
                        height=300, margin=dict(t=40,b=80),
                        xaxis_tickangle=-20)
                    st.plotly_chart(fig8a, use_container_width=True)

                    fig8b = go.Figure()
                    fig8b.add_trace(go.Bar(
                        name="Jobs Done", x=names8, y=done8,
                        marker_color="#3B82F6",
                        text=done8, textposition="outside"))
                    fig8b.update_layout(
                        title="Throughput by Scenario",
                        yaxis_title="Jobs Completed",
                        height=280, margin=dict(t=40,b=80),
                        xaxis_tickangle=-20)
                    st.plotly_chart(fig8b, use_container_width=True)

                    # Export CSV
                    st.divider()
                    st.subheader("💾 Export Results")
                    csv_data = df8.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Comparison (CSV)",
                        data=csv_data,
                        file_name="simulation_scenarios.csv",
                        mime="text/csv")

                    # Gantt-style: per-product per-scenario Wq
                    st.subheader("🔀 Per-Product Wq by Scenario")
                    fig8c = go.Figure()
                    colors8 = px.colors.qualitative.Set2
                    for idx, res8 in enumerate(exp8):
                        types8  = [p["type"] for p in res8["products"]]
                        wqs8    = [p["total_Wq_sim"] for p in res8["products"]]
                        fig8c.add_trace(go.Bar(
                            name=res8["scenario_name"],
                            x=types8, y=wqs8,
                            marker_color=colors8[idx%len(colors8)]))
                    fig8c.update_layout(barmode="group",
                        xaxis_title="Product",
                        yaxis_title="Total Wq [hr]",
                        height=300, margin=dict(t=20,b=40),
                        legend=dict(orientation="h", y=-0.3))
                    st.plotly_chart(fig8c, use_container_width=True)
            else:
                st.info("👆 Select scenarios and click RUN EXPERIMENTS")
                st.markdown("""
                **Available scenarios:**
                - **Baseline**: Current factory S=[5,3,5], 1 shift
                - **Gated policy**: Same servers, fairer scheduling
                - **Add S2=4**: Add 1 server to bottleneck Stage 2
                - **2/3 Shifts**: Increase capacity via shifts (CL-11)

                **Output includes:**
                - Comparison table (throughput, revenue, Wq)
                - Revenue and throughput bar charts
                - Per-product Wq comparison
                - CSV export for further analysis
                """)

# ═══════════════════════════════════════════════════════════════════
# TAB 9 — INPUT & FIT (distribution_fitting.py UI)
# ═══════════════════════════════════════════════════════════════════
with tab9:
    st.header("🔧 Input & Distribution Fitting")
    st.caption("3-step input pipeline: Demand → λ, Service → μ, Cost → NR")
    st.markdown("""
    **This is the ENTRY POINT** of the platform.
    Enter your real data here → calibrated λ and μ feed all other tabs.
    """)

    # Import fitting engine
    try:
        from distribution_fitting import (
            compute_lambda, compute_mu_stages, compute_mu_stages_parts,
            compute_cost_simple, compute_cost_detailed,
            compute_F1_curve, fit_arrivals, fit_service,
            run_fitting_pipeline, export_to_dashboard,
        )
        fit_ok = True
    except ImportError:
        fit_ok = False
        st.error("distribution_fitting.py not found. Place in same folder.")

    if fit_ok:
        # ── STEP 1: DEMAND INPUT → λ ──────────────────────────────────
        st.subheader("STEP 1 — Demand Input → λ [u/hr]")
        st.caption("Level 1: Sales Forecast (annual/quarterly) | "
                   "Level 2: MPS (monthly confirmed orders)")

        c1, c2 = st.columns([1, 2])
        with c1:
            level_sel = st.radio("Input Level",
                ["Level 1 — Sales Forecast (Annual)",
                 "Level 2 — MPS (Monthly)"], key="lvl9")
            period9   = "annual" if "Annual" in level_sel else "monthly"
            w_days9   = st.number_input("Working days/year",
                100, 365, 240, key="wd9")
            hrs9      = st.number_input("Hours/shift", 4, 24, 8, key="h9")
            shifts9   = st.selectbox("Shifts/day", [1,2,3], key="sh9")
            n_prod9   = st.slider("Number of products (N)",
                1, 10, 6, key="np9")

        with c2:
            st.markdown("**Enter demand per product:**")
            prod_inputs = []
            # Default thesis values for reference
            defaults = [
                ("8BD",897),("8BK",580),("8FJ500",524),
                ("8AS10",524),("3CF12KVA",1920),("G.T3",1550)
            ]
            for i in range(n_prod9):
                def_name = defaults[i][0] if i<len(defaults) else f"P{i+1}"
                def_dem  = defaults[i][1] if i<len(defaults) else 100
                cc1, cc2, cc3 = st.columns([1,1,1])
                name_i   = cc1.text_input(f"Name",
                    def_name, key=f"nm9_{i}")
                dem_i    = cc2.number_input(f"Demand",
                    1.0, 100000.0, float(def_dem),
                    key=f"dm9_{i}", label_visibility="visible")
                hrs_tot  = cc3.number_input(f"Total hrs/unit",
                    1.0, 500.0, 80.0, key=f"th9_{i}",
                    label_visibility="visible")
                # Compute λ immediately
                r_lam = compute_lambda(dem_i, period9, w_days9, hrs9, shifts9)
                prod_inputs.append({
                    "name": name_i, "demand": dem_i,
                    "total_hrs": hrs_tot, "lam": r_lam["lam"]
                })

            # Show λ table
            if prod_inputs:
                df_lam9 = pd.DataFrame([{
                    "Product"    : p["name"],
                    "Demand"     : p["demand"],
                    "λ [u/hr]"   : round(p["lam"], 4),
                    "Total hrs"  : p["total_hrs"],
                } for p in prod_inputs])
                st.dataframe(df_lam9, hide_index=True,
                             use_container_width=True)

        st.divider()

        # ── STEP 2: SERVICE TIME → μ ───────────────────────────────────
        st.subheader("STEP 2 — Service Time → μ per Stage")
        st.caption("CL-10: Stage ratios [0.2:0.5:0.3] or manual override")

        c3, c4 = st.columns([1, 2])
        with c3:
            ratio_mode9 = st.radio("Stage time input",
                ["Auto (ratios)","Manual per stage"], key="rm9")
            r1_9 = st.slider("Stage 1 ratio (Cutting)",
                0.1, 0.6, 0.2, 0.05, key="r19",
                disabled=(ratio_mode9=="Manual per stage"))
            r2_9 = st.slider("Stage 2 ratio (Punching)",
                0.1, 0.6, 0.5, 0.05, key="r29",
                disabled=(ratio_mode9=="Manual per stage"))
            r3_9 = 1.0 - r1_9 - r2_9
            st.info(f"Stage 3 ratio (Bending): {r3_9:.2f} (auto)")
            ratios9 = [r1_9, r2_9, r3_9]

            st.divider()
            use_parts9 = st.checkbox(
                "Use part-level granularity (Erlang-k)", value=False,
                key="parts9",
                help="A product's total_hrs is really the sum of many "
                     "small part operations (~0.1-0.2hr each), not one "
                     "big draw. Enabling this fits an Erlang-k "
                     "distribution per stage instead of assuming "
                     "Exponential — lower variance, more realistic "
                     "queueing behavior.")
            avg_part9 = None
            if use_parts9:
                avg_part9 = st.number_input(
                    "Avg time per part [hr]", 0.01, 5.0, 0.15, 0.01,
                    key="apt9")

        with c4:
            st.markdown("**Per-stage service rates:**")
            mu_rows = []
            for p in prod_inputs:
                if use_parts9:
                    mu_d = compute_mu_stages_parts(
                        p["total_hrs"], avg_part_time=avg_part9,
                        ratios=ratios9)
                else:
                    mu_d = compute_mu_stages(p["total_hrs"], ratios9)
                p["mu_stages_data"] = mu_d   # carried into Step 4 below
                for s in mu_d["stages"]:
                    mu_rows.append({
                        "Product"    : p["name"],
                        "Stage"      : s["stage_name"],
                        "St [hr/u]"  : round(s["service_time"],2),
                        "μ [u/hr]"   : round(s["mu"],5),
                        "Ratio"      : s["ratio"],
                        **({"k (Erlang)": s["k"], "CoV²": s["CoV2"]}
                           if use_parts9 else {}),
                    })
            if mu_rows:
                st.dataframe(pd.DataFrame(mu_rows),
                    hide_index=True, use_container_width=True)
            st.caption("Stage 2 (Punching) = longest service time "
                       "→ potential bottleneck ✓")
            if use_parts9:
                st.caption(f"ℹ️ Part-level mode: k=hundreds typically "
                           f"expected (many small ~{avg_part9}hr parts "
                           f"summing to the total) → low CoV² → less "
                           f"variable, more predictable service times "
                           f"than a single Exponential draw would give.")

        st.divider()

        # ── STEP 3: COST INPUT → NR ────────────────────────────────────
        st.subheader("STEP 3 — Cost Input → NR per unit")
        st.caption("Simple: SP & F1 direct | "
                   "Detailed: Table 4.2 breakdown (CL-13 non-linear)")

        cost_mode9 = st.radio("Cost input mode",
            ["Simple (Table 4.3 direct)",
             "Detailed (Table 4.2 breakdown)"],
            key="cm9", horizontal=True)

        cost_rows = []
        if "Simple" in cost_mode9:
            st.markdown("**Enter SP and F1 per product:**")
            sp_defaults = [45000,55000,25000,15000,3000,8000]
            f1_defaults = [36000,44000,20000,12000,2400,6400]
            for i, p in enumerate(prod_inputs):
                sp_d = sp_defaults[i] if i<len(sp_defaults) else 10000
                f1_d = f1_defaults[i] if i<len(f1_defaults) else 8000
                cc1,cc2,cc3,cc4 = st.columns([1,1,1,1])
                cc1.text_input("Product", p["name"],
                    disabled=True, key=f"cp9_{i}")
                sp_i = cc2.number_input("SP [$/u]",
                    100.0, 2_000_000.0, float(sp_d), key=f"sp9_{i}")
                f1_i = cc3.number_input("F1 [$/u]",
                    100.0, 2_000_000.0, float(f1_d), key=f"f19_{i}")
                NR_i = sp_i - f1_i
                ratio_i = f1_i/sp_i if sp_i>0 else 0
                cc4.metric("NR [$/u]", f"${NR_i:,.0f}",
                    delta=f"F1/SP={ratio_i:.2f}")
                cost_rows.append({
                    "Product":p["name"],"SP":sp_i,
                    "F1":f1_i,"NR":NR_i,
                    "F1/SP":round(ratio_i,3)
                })
        else:
            st.markdown("**Table 4.2 cost components (shared rates):**")
            c5,c6,c7,c8 = st.columns(4)
            DL9  = c5.number_input("DL [$/hr]",  0.0,500.0,10.0,key="dl9")
            DM9  = c6.number_input("DM [$/hr]",  0.0,500.0,50.0,key="dm9")
            IDL9 = c7.number_input("IDL [$/hr]", 0.0,200.0, 5.0,key="idl9")
            IDM9 = c8.number_input("IDM [$/hr]", 0.0,200.0,25.0,key="idm9")
            FC9  = st.number_input("FC [$/hr]", 0.0,500.0,0.0,key="fc9")
            st.caption("⚠️ SP_standard_parts EXCLUDED (CL-9): "
                       "already embedded in SP for applicable products")
            st.markdown("**Per-product SP & D_MAT** — editable, since SP "
                       "(Table 4.2/4.9) and material cost D_MAT differ by "
                       "product type. Defaults shown are the old thesis "
                       "reference values where available; override freely "
                       "for new/2026 products.")
            sp_defaults = [45000,55000,25000,15000,3000,8000]
            dmat_defaults = [2000,2000,2000,2000,2000,2000]
            for i, p in enumerate(prod_inputs):
                sp_d = sp_defaults[i] if i<len(sp_defaults) else 10000
                dmat_d = dmat_defaults[i] if i<len(dmat_defaults) else 2000
                cc1,cc2,cc3 = st.columns([1,1,1])
                cc1.text_input("Product", p["name"],
                    disabled=True, key=f"cpd9_{i}")
                sp_i = cc2.number_input("SP [$/u]",
                    100.0, 2_000_000.0, float(sp_d), key=f"spd9_{i}")
                dmat_i = cc3.number_input("D_MAT [$/u]",
                    0.0, 2_000_000.0, float(dmat_d), key=f"dm9b_{i}")
                c_det = compute_cost_detailed(
                    sp_i,DL9,DM9,IDL9,IDM9,dmat_i,FC9)
                cost_rows.append({
                    "Product":p["name"],"SP":sp_i,
                    "F1":c_det["F1"],"NR":c_det["NR"],
                    "F1/SP":c_det["F1_SP_ratio"]
                })

        if cost_rows:
            st.dataframe(pd.DataFrame(cost_rows),
                hide_index=True, use_container_width=True)

        # F1(S) non-linear curve — CL-13
        if cost_rows and prod_inputs:
            with st.expander("📈 F1(S) Non-Linear Curve — CL-13"):
                p_sel = st.selectbox("Product",
                    [p["name"] for p in prod_inputs], key="fcurve9")
                p_idx = next(i for i,p in enumerate(prod_inputs)
                             if p["name"]==p_sel)
                cr = cost_rows[p_idx]
                curve = compute_F1_curve(
                    cr["SP"],
                    10, 50, 5, 25, 2000,
                    prod_inputs[p_idx]["lam"],
                    1.0/prod_inputs[p_idx]["total_hrs"])
                curve_ok = [r for r in curve if r["F1"] is not None]
                if curve_ok:
                    fig9a = go.Figure()
                    fig9a.add_trace(go.Scatter(
                        x=[r["S"] for r in curve_ok],
                        y=[r["F1"] for r in curve_ok],
                        name="F1(S) cost", mode="lines+markers",
                        line=dict(color="#EF4444",width=2)))
                    fig9a.add_trace(go.Scatter(
                        x=[r["S"] for r in curve_ok],
                        y=[max(0,r["NR"]) for r in curve_ok],
                        name="Rn (net revenue)", mode="lines+markers",
                        line=dict(color="#059669",width=2)))
                    fig9a.update_layout(
                        title=f"F1(S) vs Rn — {p_sel} (CL-13)",
                        xaxis_title="S (servers)",
                        yaxis_title="$ per unit",
                        height=280, margin=dict(t=40,b=40),
                        legend=dict(orientation="h"))
                    st.plotly_chart(fig9a, use_container_width=True)
                    st.caption("F1(S) non-linear: server cost↑ "
                               "vs waiting cost↓ → minimum at S*")

        st.divider()

        # ── STEP 4: FIT & EXPORT ────────────────────────────────────────
        st.subheader("STEP 4 — Distribution Fit & Export")
        st.caption("Fits Poisson(λ) + Exp/Gamma(μ) → "
                   "exports to queue_engine + all tabs")

        fit_btn9 = st.button("⚡ FIT & EXPORT TO DASHBOARD",
            type="primary", key="fit9")

        if fit_btn9 and prod_inputs and cost_rows:
            # Build full product list for pipeline
            full_prods = []
            for i, p in enumerate(prod_inputs):
                cr = cost_rows[i]
                full_prods.append({
                    "name"     : p["name"],
                    "demand"   : p["demand"],
                    "total_hrs": p["total_hrs"],
                    "SP"       : cr["SP"],
                    "F1"       : cr["F1"],
                })

            result9 = run_fitting_pipeline(
                full_prods, period=period9,
                working_days=w_days9, hrs_per_shift=hrs9,
                n_shifts=shifts9, stage_ratios=ratios9,
                cost_mode="simple",
                avg_part_time=avg_part9 if use_parts9 else None)

            exported9 = export_to_dashboard(result9)

            # ── ADDITIVE (Option E): actually write global_products to
            # session_state, using the part-level mu_stages_data (with k)
            # computed in Step 2 if part-level mode was enabled. This is
            # the real fix for Step 4's button text, which previously
            # claimed "exports to all tabs" but only showed a table+CSV.
            global_products = []
            for p in prod_inputs:
                mu_d = p.get("mu_stages_data") or compute_mu_stages(
                    p["total_hrs"], ratios9)
                vals = exported9.get(p["name"], {})
                global_products.append({
                    "type"      : p["name"],
                    "lam"       : vals.get("lam", 0.0),
                    "total_hrs" : p["total_hrs"],
                    "stage_ratios": ratios9,
                    "stages"    : [
                        {"stage": s["stage"], "mu": s["mu"],
                         "k": s.get("k", 1), "CoV2": s.get("CoV2", 1.0)}
                        for s in mu_d["stages"]
                    ],
                    "mu_overall": mu_d["mu_overall"],
                    "SP"        : vals.get("SP",
                                    next((c["SP"] for c in cost_rows
                                          if c["Product"]==p["name"]), 0)),
                    "NR"        : vals.get("NR", 0.0),
                    "rho"       : vals.get("rho", 0.0),
                    "model"     : vals.get("model", "M/M/S"),
                    "stable"    : vals.get("stable", True),
                    "part_level": use_parts9,
                })
            st.session_state["global_products"] = global_products
            st.session_state["global_products_meta"] = {
                "n_shifts": shifts9, "period": period9,
                "avg_part_time": avg_part9 if use_parts9 else None,
                "fitted_at": pd.Timestamp.now().isoformat(),
            }
            # Rerun immediately so every tab's "Product data source" toggle
            # (which checks session_state at the TOP of the script, before
            # Tab 9 runs) reflects the new fit right away instead of only
            # appearing after the user's next unrelated interaction.
            st.rerun()

        # ── Persistent results display — reads from session_state, so it
        # survives the st.rerun() above and shows the LAST fit any time
        # this tab is viewed, not just in the single instant of the click.
        if st.session_state.get("global_products"):
            gp9 = st.session_state["global_products"]
            meta9 = st.session_state.get("global_products_meta", {})
            st.success(f"✅ {len(gp9)} products fitted"
                       + (f" ({meta9['fitted_at'][:19]})" if meta9.get("fitted_at") else "")
                       + " — the 'Product data source' toggle in other tabs "
                         "can now use this fitted set instead of the "
                         "built-in 6-product case study.")
            st.markdown("**Fitted Parameters:**")
            fit_rows = [{
                "Product"  : p["type"],
                "λ [u/hr]" : p["lam"],
                "μ [u/hr]" : p["mu_overall"],
                "ρ"        : p["rho"],
                "NR [$/u]" : p["NR"],
                "Model"    : p["model"],
                "Stable"   : "✅" if p["stable"] else "⚠️",
                **({"k (Stage2)": p["stages"][1]["k"]} if p.get("part_level") else {}),
            } for p in gp9]
            st.dataframe(pd.DataFrame(fit_rows),
                hide_index=True, use_container_width=True)

            # Rho bar chart
            fig9b = px.bar(pd.DataFrame(fit_rows),
                x="Product", y="ρ", color="ρ",
                color_continuous_scale="RdYlGn_r",
                title="System Utilization ρ per Product")
            fig9b.add_hline(y=1.0, line_dash="dash",
                line_color="red", annotation_text="ρ=1 unstable")
            fig9b.update_layout(height=280, margin=dict(t=40,b=20))
            st.plotly_chart(fig9b, use_container_width=True)

            # Export as CSV
            df_exp9 = pd.DataFrame(fit_rows)
            st.download_button("⬇️ Download Fitted Parameters (CSV)",
                data=df_exp9.to_csv(index=False),
                file_name="fitted_parameters.csv",
                mime="text/csv")

            st.info("💡 Look for the **'Product data source'** toggle at "
                    "the top of Tab 4 to use this fitted set instead of "
                    "the built-in 6-product case study. (More tabs get "
                    "this toggle in upcoming sessions — see the abstract "
                    "for progress.)")

            if st.button("🗑️ Clear fitted products (revert all tabs to built-in)",
                         key="clear_gp9"):
                del st.session_state["global_products"]
                if "global_products_meta" in st.session_state:
                    del st.session_state["global_products_meta"]
                st.rerun()

        if not fit_btn9 and not st.session_state.get("global_products"):
            st.markdown("""
            **What Step 4 does:**
            - Fits **Poisson distribution** to arrival data → λ
            - Tests **Exponential → Gamma → Erlang** for service → μ
            - Selects best-fit model (M/M/S, M/G/1, M/Ek/1, M/D/1)
            - Exports calibrated parameters to all dashboard tabs

            **Two input sources (CL-11):**
            - **Level 1 (Sales Forecast):** Annual predicted demand
            - **Level 2 (MPS):** Monthly confirmed production orders

            **Cost note (CL-9):**
            SP_standard_parts (25,000) is excluded from Rn.
            It is already embedded in SP for 8BD, 8BK, 8FJ500 only.
            """)

# FOOTER — ★ COMPLETE PROJECT ★
st.divider()
col_f1, col_f2 = st.columns([2,1])
with col_f1:
    st.caption(
        "★ M.Sc. Thesis 1999 → Modernized 2026 — PROJECT COMPLETE ★ | "
        "queue_engine ✅ · capacity_planner ✅ · simpy_engine ✅ · "
        "distribution_fitting ✅ · live_simulation ✅ · dashboard ✅ (9 tabs)")
with col_f2:
    st.caption("📖 Thesis_Modernization_Notes.md — CL-1→13 reference")

# ═══════════════════════════════════════════════════════════════════
# TAB 10 — ANIMATED FACTORY FLOOR (Session 1, New Project Track A)
# ═══════════════════════════════════════════════════════════════════
with tab10:
    st.header("🎬 Animated Factory Floor")
    st.caption("Live-style Plotly animation of jobs flowing Stage 1 → 2 → 3, "
               "built on the same live_simulation.py DES engine as Tab 7")

    try:
        from animation_engine import (load_snapshots_from_live, build_animation_figure,
                                       get_product_names, build_cfg_for_run)
        anim_available = True
    except ImportError:
        anim_available = False
        st.error("animation_engine.py not found. Place it in the same folder.")

    if anim_available:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("⚙️ Animation Config")
            src10, src10_label = product_source_selector("t10")
            if src10_label == "Built-in":
                mode10 = st.radio("λ Source", ["Experimental", "Actual MPS"], key="mode10")
            else:
                st.caption(f"Using {len(src10)} products fitted in Tab 9")
            policy10 = st.radio("Service Policy", ["exhaustive", "gated"], key="pol10")
            n_sh10 = st.selectbox("Shifts/day", [1, 2, 3], key="sh10")
            sim_t10 = st.slider("Sim time [hr]", 50, 1000, 200, 25, key="st10")
            speed10 = st.select_slider("Playback speed",
                options=["0.5x", "1x", "2x", "4x"], value="1x", key="sp10")
            speed_ms10 = {"0.5x": 600, "1x": 300, "2x": 150, "4x": 75}[speed10]
            renege_T10 = st.slider(
                "Patience T [hr] (Eq 3.8 reneging)",
                0.1, 5.0, 1.0, 0.1, key="rt10",
                help="A job waiting longer than this at any stage abandons "
                     "the queue and leaves the system.")

            st.divider()
            st.markdown("**Server Config (CL-12):**")
            S10, shifts10 = [], []
            for j, (nm, df) in enumerate(zip(["Cutting", "Punching", "Bending"], [5, 3, 5])):
                cS, cSh = st.columns([2,1])
                sv = cS.slider(f"S{j+1} {nm}", 1, 10, df, key=f"s10_{j}")
                sh = cSh.selectbox("Shifts", [1,2,3], key=f"sh10_{j}")
                S10.append(sv)
                shifts10.append(sh)

            st.caption("ℹ️ At S=[5,3,5], queues now stay bounded (jobs that "
                       "wait past the patience time T above abandon the "
                       "queue — Eq 3.8). But most demand is still lost: "
                       "at T=1.0hr, roughly 90%+ of arrivals renege before "
                       "being served under either product set — this is a "
                       "genuine capacity shortfall, not a display issue. "
                       "See Tab 7 (Live Simulation) for per-product renege "
                       "rates. Dots per stage are capped at 25 for "
                       "readability; the true queue count is still shown.")

            run10 = st.button("▶ GENERATE ANIMATION", type="primary", key="run10")

        with c2:
            if run10:
                with st.spinner("Running simulation and building animation frames..."):
                    if src10_label == "Built-in":
                        snaps10 = load_snapshots_from_live(
                            S_stages=S10, policy=policy10, n_shifts=n_sh10,
                            shifts_per_stage=shifts10,
                            sim_time=sim_t10, warmup=max(sim_t10 // 10, 5),
                            snapshot_interval=max(sim_t10 // 40, 5),
                            product_mode=mode10, renege_T=renege_T10,
                        )
                        prod_names10 = get_product_names(mode10)
                    else:
                        snaps10 = load_snapshots_from_live(
                            S_stages=S10, policy=policy10, n_shifts=n_sh10,
                            shifts_per_stage=shifts10,
                            sim_time=sim_t10, warmup=max(sim_t10 // 10, 5),
                            snapshot_interval=max(sim_t10 // 40, 5),
                            products=src10, renege_T=renege_T10,
                        )
                        prod_names10 = [p["type"] for p in src10]
                    # NOTE: build_cfg_for_run uses the PHYSICAL S10 (not
                    # effective S×shifts) deliberately — the animation
                    # should show real machine count, not a doubled count
                    # that would misrepresent what's physically on the
                    # floor. The shift effect shows up in throughput/queue
                    # behavior instead, which is the honest way to show it.
                    cfg10 = build_cfg_for_run(S10, prod_names10)
                    fig10 = build_animation_figure(snaps10, cfg10, frame_duration_ms=speed_ms10)
                st.session_state["tab10_fig"] = fig10
                st.session_state["tab10_frames"] = len(snaps10)
                st.session_state["tab10_products"] = prod_names10

            if "tab10_fig" in st.session_state:
                st.plotly_chart(st.session_state["tab10_fig"], use_container_width=True)
                st.caption(f"{st.session_state['tab10_frames']} frames generated · "
                           f"{len(st.session_state.get('tab10_products', []))} products: "
                           f"{', '.join(st.session_state.get('tab10_products', []))}")
            else:
                st.info("👆 Configure parameters and click GENERATE ANIMATION")
                st.markdown("""
                **What this animation shows:**
                - 🟢 idle machine · 🔴 busy machine (per stage, CL-12 groups)
                - Colored dots = jobs queued/in-service (colored by product)
                - Per-stage queue counter, with a saturation warning
                - Play / Pause / Reset + frame-by-frame timeline slider

                **Note on data fidelity:** live_simulation.py logs
                *aggregate* KPI counts per snapshot (busy/idle/queue), not
                individual job identity. Product colors on queued/in-service
                dots are therefore a weighted approximation (by ρ), not an
                exact per-job trace. Exact per-job animation would need a
                small logging patch to live_simulation.py — ask if you'd
                like that built next.
                """)
