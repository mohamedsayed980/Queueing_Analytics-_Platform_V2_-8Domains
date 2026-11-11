# Queueing Analytics Platform — Ver2_8_domains
**Mohamed · M3 · M.Sc. Queueing Engine Extended to Multi-Domain Service Analytics**

A Streamlit platform for exploring queueing systems across eight service and operations domains. The project extends an M.Sc. queueing theory engine into interactive dashboards for capacity planning, staffing, bottleneck analysis, and service performance.

> **Project status:** Ver2_8_domains is the eight-domain edition. The Manufacturing dashboard is the M.Sc. Crown Project; the other dashboards apply queueing analysis to service and logistics settings. The included sample scenarios are illustrative and should not be treated as live operational forecasts.

---

## 🌐 Live Demo
👉 **[Launch Platform](https://your-app-url.streamlit.app)**

---

## 📌 Overview
A multi-domain queueing analytics platform built on an original M.Sc. thesis engine.
One mathematical engine (M/M/S) powers 8 completely different service domains:

```
## Domains

| Domain | Example analysis |
|---|---|
| Manufacturing | Multi-stage job shop, bottlenecks, throughput, work in process, capacity planning, and simulation |
| Bank | Teller capacity, waiting time, staffing cost, and sensitivity |
| Healthcare | Patient queues, waiting targets, breach rates, and doctor utilization |
| Call Center | Service levels, abandonment, call mix, and agent staffing |
| Airport Security | Passenger screening, checkpoint capacity, and flight-driven arrivals |
| Restaurant | Table capacity, reservation mix, and peak-period demand |
| Supermarket | Checkout lane capacity, basket effects, and staffing |
| Port / Logistics | Berth capacity, ship turnaround, crane allocation, and anchorage queues |

---

## 🏗️ Architecture

```
ONE ENGINE → MULTIPLE DOMAINS

queue_engine.py (M/M/S core)
    ↓
MMS(λ, μ, S) → {ρ, Lq, Wq, Ws, Ls, P0}
    ↓
Domain Templates (one per service):
  Bank · Healthcare · Call Center · Airport .Restaurant . Supermarket .  Port / Logistics
```



## Domain templates

Rates and server counts below follow the defaults shown on the V2.8 home page. Record totals are omitted because the available V2.8 files do not establish consistent counts for all domains; the V1 figures should only be reused if they still match the datasets in the deployed repository.

| Domain | λ (arrivals) | μ (service rate per server) | S (servers) | ρ (utilization) | Records | Key KPIs |
|---|---:|---:|---:|---:|---|---|
| 🏭 Manufacturing | Jobs/hr (scenario input) | Machines/hr (scenario input) | N stages; varies by stage | Varies by scenario | Not verified | OEE, bottleneck, throughput, WIP |
| 🏦 Bank | 30 customers/hr | 12 customers/hr | 3 tellers | 0.833 | Not verified | Wait time, teller cost, savings |
| 🏥 Healthcare | 12 patients/hr | 4 patients/hr | 4 doctors | 0.750 | Not verified | Wait time, breach rate, doctor cost |
| 📞 Call Center | 50 calls/hr | 20 calls/hr | 3 agents | 0.833 | Not verified | SLA, abandonment, AHT cost |
| ✈️ Airport Security | 200 passengers/hr | Not stated on home page | 4 checkpoints | Not verified | Not verified | Wait time, checkpoint cost, savings |
| 🍽️ Restaurant | 30 tables/hr | Not stated on home page | 20 tables | Not verified | Not verified | Wait time, peak lunch/dinner demand |
| 🛒 Supermarket | 80 customers/hr | Not stated on home page | 6 checkout lanes | Not verified | Not verified | Wait time, basket-size effect, lane cost |
| 🚢 Port / Logistics | 8 ships/day | 3.2 ships/berth/day (home-page scenario) | 3 berths | 0.833 | Not verified | SLA, ship turnaround, berth/crane capacity |

---

## 🔢 Mathematical Foundation — M/M/S Model

```
λ  = arrival rate
μ  = service rate per server
S  = number of parallel servers
---
ρ  = λ / (S × μ)           Server utilization
Lq = f(P₀, ρ, S)           Avg queue length
Wq = Lq / λ                Avg waiting time
Ws = Wq + 1/μ              Avg time in system
Ls = λ × Ws                Avg customers in system

```
**All KPIs update live with sidebar sliders — fully interactive.**
---
## How it works

The platform uses a shared multi-server queueing model (M/M/S) across domains. Each domain presents its own inputs, terminology, data, and key performance indicators while reusing the common queueing calculations.

Finite-capacity domains may also account for blocked or diverted arrivals. Simulation and domain-specific calculations supplement the analytical model where appropriate.

## 🗂️ Project Structure
📁 Queueing_Analytics_Platform_V2_8Domains/
├── data/
│   ├── airport_queue_data_part1 … airport_queue_data_part4
│   ├── bank_queue_data
│   ├── callcenter_queue_data
│   ├── healthcare_queue_data
│   ├── port_logistics
│   ├── restaurant_queue_data
│   └── supermarket_queue_data
├── pages/
│   ├── airport.py
│   ├── Bank.py
│   ├── CallCenter.py
│   ├── Healthcare.py
│   ├── Manufacturing.py
│   ├── Port_Logistics.py
│   ├── restaurant.py
│   └── supermarket.py
├── 3M_logo.png
├── airport_dashboard.py
├── animation_engine.py
├── bank_dashboard.py
├── callcenter_dashboard.py
├── data_loader.py
├── distribution_fitting.py
├── healthcare_dashboard.py
├── Home_Platform.py             # Streamlit home page / app entry point
├── live_simulation.py
├── M3_logo.png
├── packages                     
├── port_logistic_dashboard.py
├── queue_engine.py              # Shared queueing model
├── README.md
├── requirements.txt                 
├── runtime.txt                     
├── simpy_engine.py
└── supermarket_dashboard.py

---
## 🚀 How to Run

```bash
git clone https://github.com/mohamedsayed980/Queueing_Analytics_Platform.git
cd Queueing_Analytics_Platform
pip install -r requirements.txt
streamlit run Home.py
```
---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`
`Plotly` · `SciPy` · `SimPy` · `Statsmodels`

---

## 👑 Crown Project — Manufacturing Origin

The platform is built on a rebuilt M.Sc. thesis in job-shop capacity planning:

| Engine File | Lines | Tests |
|-------------|-------|-------|
| queue_engine.py | 1,638 | 19/21 ✅ |
| simpy_engine.py | 488 | 6/6 ✅ |
| distribution_fitting.py | 921 | 17/17 ✅ |
| animation_engine.py | 515 | Complete ✅ |
| live_simulation.py | 852 | 4/4 ✅ |

```

If the repository uses a differently named entry point, substitute that file name. Keep the domain pages, shared engine modules, and any required data files in the paths expected by the application.

## Deploy to Streamlit Community Cloud

1. Push the complete project to a GitHub repository.
2. Confirm the repository contains the app entry point, all referenced page and engine modules, required data files, and `requirements.txt`.
3. In Streamlit Community Cloud, create an app from the repository and select `Home_Platform.py` as the main file.
4. Add any required secrets through the deployment platform's secrets settings. Do not commit credentials or private data.

```
## Data and interpretation

Use synthetic/demo data only as examples unless a dataset is explicitly identified as real and authorized for use. Model outputs depend on the arrival-rate, service-rate, staffing, and capacity assumptions supplied to the app. Validate those assumptions before using results for operational decisions.

## Author

**Mohamed · M3**  
**Mohamed · M3 · M.Sc. Mechanical Engineering · Queueing Theory → Service Analytics**




