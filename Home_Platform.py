"""
Queueing Analytics Platform — Master Hub
Home.py · Entry Point
Author : Mohamed · M3

Multi-Domain Queueing Platform built on M.Sc. thesis engine:
  Manufacturing (Crown) · Bank · Healthcare · Call Center
  → Airport · Restaurant (future)

Architecture:
  One queueing engine (queue_engine.py) powers ALL domains
  Each domain = separate page with domain-specific KPIs
"""
import streamlit as st
import pathlib

st.set_page_config(
    page_title="Queueing Analytics Platform · M3",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("## 🔢 Queueing Platform")
    st.markdown("**M3 · M.Sc. Engine Extended**")
    st.divider()
    st.markdown("### 🗂 Navigate to Domain:")
    st.markdown("📊 Use sidebar pages below ↓")
    st.divider()
    st.markdown("### 📐 Engine: M/M/S")
    st.markdown("```\nρ  = λ/(S×μ)\nLq = f(P₀,ρ,S)\nWq = Lq/λ\nWs = Wq+1/μ\n```")
    st.divider()
    st.caption("M3 · Mechanical Engineer / Data Analyst")
    st.caption("M.Sc. Queueing Theory → Service Applications")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f0f4f8;}
.hero{background:linear-gradient(135deg,#0d1b2a,#1a237e,#00695c);
      padding:52px 48px;border-radius:16px;margin-bottom:32px;
      box-shadow:0 8px 32px rgba(0,0,0,0.18);}
.hero h1{color:#ffffff !important;font-size:2.6rem;font-weight:900;margin:0 0 10px 0;}
.hero p{color:#b3d4ff !important;font-size:1.1rem;margin:0;}
.domain-card{background:#ffffff;border-radius:12px;padding:24px 20px;
  box-shadow:0 3px 16px rgba(0,0,0,0.09);transition:transform 0.2s;
  height:100%;border-top:5px solid #ccc;}
.domain-card:hover{transform:translateY(-3px);}
.domain-card h3{font-size:1.2rem;font-weight:800;margin:8px 0 6px 0;}
.domain-card p{font-size:0.88rem;color:#546e7a;margin:0 0 10px 0;line-height:1.6;}
.domain-card .badge{display:inline-block;padding:3px 10px;border-radius:12px;
  font-size:0.75rem;font-weight:700;color:#fff;margin-bottom:8px;}
.kpi-strip{background:#ffffff;border-radius:10px;padding:16px 20px;
  text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.07);}
.kpi-num{font-size:2rem;font-weight:900;color:#1a237e;}
.kpi-lbl{font-size:0.78rem;color:#546e7a;margin-top:4px;}
.engine-box{background:linear-gradient(135deg,#1a237e,#283593);
  border-radius:12px;padding:20px 24px;color:#fff;}
.future-card{background:#f8f9fa;border-radius:10px;padding:18px;
  border:2px dashed #ccc;opacity:0.75;}
</style>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔢 Queueing Analytics Platform</h1>
  <p>Multi-Domain Capacity Planning · Built on M.Sc. Thesis Engine ·
     M/M/S · Manufacturing → Services · M3</p>
</div>""", unsafe_allow_html=True)

# ── PLATFORM STATS ───────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5],[
    ("4","Active Domains"),
    ("1","Core Engine"),
    ("250K+","Records Simulated"),
    ("9","Tabs per Domain"),
    ("5+","More Coming"),
]):
    col.markdown(f"""<div class="kpi-strip">
      <div class="kpi-num">{num}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── DOMAIN CARDS ─────────────────────────────────────────────
st.markdown("### 🗂 Select a Domain to Analyse")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#c62828;">
      <span class="badge" style="background:#c62828;">M.Sc. Crown ★✅ Live </span>
      <h3 style="color:#c62828;">🏭 Manufacturing</h3>
      <p>Multi-stage job shop · N-stage M/M/S engine ·
         Bottleneck analysis · SimPy simulation ·
         Capacity planning · Live animation</p>
      <b>λ:</b> Jobs/hr · <b>S:</b> Machines per stage<br>
      <b>KPIs:</b> Bottleneck · Throughput · WIP
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Manufacturing.py", label="▶ Open Manufacturing", use_container_width=True)

with col2:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#1565c0;">
      <span class="badge" style="background:#1565c0;">✅ Live</span>
      <h3 style="color:#1565c0;">🏦 Bank Queue</h3>
      <p>Customer teller queues · Arrival patterns ·
         Service time fitting · Optimal teller count ·
         Cost minimisation · Sensitivity heatmap</p>
      <b>λ:</b> 30 cust/hr · <b>S:</b> 3 tellers · <b>ρ:</b> 83.3%<br>
      <b>KPIs:</b> Wait time · Teller cost · Savings
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Bank.py", label="▶ Open Bank Queue", use_container_width=True)

with col3:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#00695c;">
      <span class="badge" style="background:#00695c;">✅ Live</span>
      <h3 style="color:#00695c;">🏥 Healthcare</h3>
      <p>Hospital ED/outpatient queues · Patient acuity ·
         Doctor utilisation · Clinical wait targets ·
         Breach rate analysis · Staffing optimisation</p>
      <b>λ:</b> 12 patients/hr · <b>S:</b> 4 doctors · <b>ρ:</b> 75%<br>
      <b>KPIs:</b> Wait · Breach % · Doctor cost
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Healthcare.py", label="▶ Open Healthcare", use_container_width=True)

with col4:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#e65100;">
      <span class="badge" style="background:#e65100;">✅ Live</span>
      <h3 style="color:#e65100;">📞 Call Center</h3>
      <p>Inbound call queues · Abandonment modelling ·
         SLA achievement · Call type breakdown ·
         Erlang C staffing · AHT optimisation</p>
      <b>λ:</b> 50 calls/hr · <b>S:</b> 3 agents · <b>ρ:</b> 83.3%<br>
      <b>KPIs:</b> SLA % · Abandon · AHT cost
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/CallCenter.py", label="▶ Open Call Center", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

#==============================================================
# ── FUTURE TEMPLATES ─────────────────────────────────────────
#=============================================================
st.markdown("### 🗂 Select a Domain to Analyse")
col5, col6, col7, col8 = st.columns(4)               
with col5:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#c62828;">
      <span class="badge" style="background:#c62828;">✅ Live </span>
      <h3 style="color:#c62828;">✈️ Airport Security</h3>
      
      <p>Passenger screening queues · Passenger screening ·
         Checkpoint capacity · Flight-driven arrivals ·
         Cost minimisation · Sensitivity heatmap</p>
      <b>λ:</b> 200/hr · <b>S:</b> 4 checkpoints<br>
      
      <b>KPIs:</b> Wait time · Checkpoint cost · Savings
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Airport.py", label="▶ Open Airport Security", use_container_width=True)
      
with col6:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#1565c0;">
      <span class="badge" style="background:#1565c0;">✅ Live</span>
      <h3 style="color:#1565c0;">🍽️ Restaurant</h3>
      <p>Table turn rates queues · Table turn rates ·
         Service time fitting · Reservation mix ·
         Cost minimisation · Sensitivity heatmap</p>
      <b>λ:</b> 30 tables/hr · <b>S:</b> 20 tables · <b>ρ:</b> 83.3%<br>
      
      <b>KPIs:</b> Wait time · Peak dinner/lunch rush
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Restaurant.py", label="▶ Open Restaurant", use_container_width=True)
    
      
with col7:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#00695c;">
      <span class="badge" style="background:#00695c;">✅ Live</span>
      <h3 style="color:#00695c;">🛒 Supermarket</h3>
      <p>Checkout lanes queues · Checkout lanes ·
         Basket size effect ·Self-service vs staffed ·
         Breach rate analysis · Staffing optimisation</p>
      <b>λ:</b> 80 cust/hr · <b>S:</b> 6 lanes · <b>ρ:</b> 75%<br>
      
      <b>KPIs:</b> Wait · Basket size effect % · lanes cost
    </div>""", unsafe_allow_html=True)
    st.page_link("pages/Supermarket.py", label="▶ Open Supermarket", use_container_width=True)
   

# ── FUTURE TEMPLATES ─────────────────────────────────────────
#st.markdown("### 🔜 Coming Soon")         
with col8:
    st.markdown("""
    <div class="domain-card" style="border-top-color:#e65100;">
      <span class="badge" style="background:#e65100;">✅ Live</span>
      <h3 style="color:#e65100;">🚢 Port / Logistics</h3>
      <p>Container berths queues · Container berths ·
         SLA achievement · Crane allocation ·
         Erlang C staffing · AHT optimisation</p>
      <b>λ:</b> 8 ships/day · <b>S:</b> 3 berths · <b>ρ:</b> 83.3%<br>
      <b>KPIs:</b> SLA % · Ship turnaround time · AHT cost
    </div>""", unsafe_allow_html=True)
    
    st.page_link("pages/Port_Logistics.py", label="▶ Open Port_Logistics", use_container_width=True) #🔜 Coming Soon
       
st.markdown("<br>", unsafe_allow_html=True)
    
# ── ENGINE ARCHITECTURE ──────────────────────────────────────
st.markdown("### ⚙️ Engine Architecture")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <div class="engine-box">
      <h3 style="color:#fff;margin:0 0 12px 0;">🔢 One Engine · All Domains</h3>
      <p style="color:#b3d4ff;font-size:0.88rem;margin-bottom:12px;">
      The same <b>queue_engine.py</b> from the M.Sc. thesis powers every domain.
      Only labels, KPIs, and data change — the math never changes.</p>
      <pre style="background:#0d1b2a;color:#80cbc4;padding:12px;
      border-radius:8px;font-size:0.78rem;margin:0;">
from queue_engine import MMS

# Works for ANY domain:
MMS(λ=30, μ=12, S=3)   # Bank
MMS(λ=12, μ=4,  S=4)   # Healthcare
MMS(λ=50, μ=20, S=3)   # Call Center
MMS(λ=200,μ=60, S=4)   # Airport (soon)

# Returns: {ρ, Lq, Wq, Ws, Ls, P0}
      </pre>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("**M/M/S Performance Measures:**")
    formula_df = __import__("pandas").DataFrame({
        "Symbol":["λ","μ","S","ρ","P₀","Lq","Wq","Ls","Ws"],
        "Formula":["Input","Input","Input","λ/(S·μ)",
                   "Erlang formula","f(P₀,ρ,S)","Lq/λ","Lq+λ/μ","Wq+1/μ"],
        "Meaning":["Arrival rate","Service rate","Servers",
                   "Utilization","Idle prob","Queue length","Wait time",
                   "System length","System time"],
    })
    st.dataframe(formula_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>
Mohamed · M3 · M.Sc. Mechanical Engineering · Queueing Theory →
Multi-Domain Service Analytics Platform<br>
Manufacturing · Banking · Healthcare · Call Center · More coming soon
</p>""", unsafe_allow_html=True)
