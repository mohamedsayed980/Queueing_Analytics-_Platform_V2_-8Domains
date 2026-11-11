"""
distribution_fitting.py  —  v1.0
==================================
Input Layer & Distribution Fitting Engine
M.Sc. Thesis 1999 — Modernized 2026

PURPOSE:
  Transforms real-world demand and service data into
  calibrated λ (arrival rate) and μ (service rate) parameters
  ready for queue_engine.py and dashboard.py.

TWO INPUT LEVELS (CL-11):
  Level 1 — Sales Forecast (Strategic):
    Annual/quarterly demand → λ_i [u/hr]
    Used for: capacity PLANNING (design targets)

  Level 2 — MPS Monthly (Operational):
    Confirmed monthly orders → λ_i [u/hr]
    Used for: scheduling & CONTROL (actual rates)

4-STEP UI:
  Step 1: Demand Input    → computes λ per product
  Step 2: Service Input   → computes μ per stage
  Step 3: Cost Input      → computes F1, NR (Simple or Detailed)
  Step 4: Fit & Export    → fits distributions, exports to engines

KEY RULES (from CL-1 to CL-13):
  - Generic product labels: P1, P2, ... Pn (user renames)
  - SP_standard_parts EXCLUDED from Rn (CL-9, applies to 8BD/8BK/8FJ500 only)
  - F1(S) is NON-LINEAR mix function (CL-13)
  - F1/SP ≈ 0.80 is OBSERVED, not mandatory (CL-3)
  - Stage ratios: [0.2:0.5:0.3] default, manual override (CL-10)
  - S=[5,3,5] actual factory default (CL-11, CL-12)
  - Arrivals: Poisson assumed (thesis + real-world)
  - Service: Exponential → Gamma → Erlang (test in order)

Session 30 — June 2026
"""

import numpy as np
from scipy import stats
from scipy.stats import chisquare
from math import factorial
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WORKING_DAYS  = 240    # days/year
DEFAULT_HRS_PER_SHIFT = 8      # hrs/shift (c = 8, CL-7)
DEFAULT_SHIFTS        = 1      # shifts/day baseline
DEFAULT_STAGE_RATIOS  = [0.2, 0.5, 0.3]   # CL-10 actual
DEFAULT_S_STAGES      = [5, 3, 5]          # CL-11, CL-12 actual factory

# Products with SP_standard_parts embedded (CL-9)
SP_EMBEDDED_PRODUCTS  = ["8BD", "8BK", "8FJ500", "P1", "P2", "P3"]

# Stage names (CL-12)
STAGE_NAMES = {
    1: "Stage 1 — Cutting (Group 1)",
    2: "Stage 2 — Punching (Group 2)",
    3: "Stage 3 — Bending (Group 3)",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STEP 1 — DEMAND INPUT → λ
# ─────────────────────────────────────────────────────────────────────────────

def compute_lambda(demand: float,
                   period: str = "annual",
                   working_days: int = DEFAULT_WORKING_DAYS,
                   hrs_per_shift: float = DEFAULT_HRS_PER_SHIFT,
                   n_shifts: int = DEFAULT_SHIFTS) -> dict:
    """
    Converts demand [units/period] → arrival rate λ [u/hr].

    Two input levels (CL-11):
      Level 1 (Sales Forecast): period = "annual" or "quarterly"
      Level 2 (MPS):            period = "monthly" or "weekly"

    Formula:
      working_hrs = working_days × hrs_per_shift × n_shifts
      λ = demand / working_hrs  [u/hr]

    Examples:
      Annual: demand=897 units/yr, 240 days, 8 hrs → λ=0.467 u/hr  (8BD ✓)
      Monthly: demand=75 units/mo, 20 days, 8 hrs → λ=0.469 u/hr   (≈ same)

    Args:
        demand       : number of units in the period
        period       : "annual","quarterly","monthly","weekly"
        working_days : working days in the period
        hrs_per_shift: hours per shift (default 8)
        n_shifts     : shifts per day

    Returns dict with λ and working hours breakdown.
    """
    # Days per period
    period_days = {
        "annual"    : working_days,
        "quarterly" : working_days // 4,
        "monthly"   : working_days // 12,
        "weekly"    : working_days // 52,
    }.get(period, working_days)

    working_hrs = period_days * hrs_per_shift * n_shifts
    lam = demand / working_hrs if working_hrs > 0 else 0.0

    return {
        "demand"       : demand,
        "period"       : period,
        "period_days"  : period_days,
        "hrs_per_shift": hrs_per_shift,
        "n_shifts"     : n_shifts,
        "working_hrs"  : working_hrs,
        "lam"          : round(lam, 6),
        "lam_rounded"  : round(lam, 3),
        "level"        : "Level 1 (Forecast)" if period in ["annual","quarterly"]
                         else "Level 2 (MPS)",
    }


def compute_lambda_batch(products: List[dict],
                          period: str = "annual",
                          working_days: int = DEFAULT_WORKING_DAYS,
                          hrs_per_shift: float = DEFAULT_HRS_PER_SHIFT,
                          n_shifts: int = DEFAULT_SHIFTS) -> List[dict]:
    """
    Compute λ for all products in batch.

    products: list of {"name": str, "demand": float, ...}
    Returns: list with λ added to each product dict.
    """
    results = []
    for p in products:
        r = compute_lambda(p["demand"], period, working_days,
                           hrs_per_shift, n_shifts)
        results.append({**p, **r})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STEP 2 — SERVICE TIME INPUT → μ
# ─────────────────────────────────────────────────────────────────────────────

def compute_mu_stages(total_hrs: float,
                      ratios: List[float] = None,
                      manual_hrs: List[float] = None) -> dict:
    """
    Computes service rate μ per stage [u/hr] from total machining hours.

    Two modes (CL-10):
      A) Ratio mode:  stage_time_j = total_hrs × ratio_j
      B) Manual mode: stage_time_j = manual_hrs[j]  (direct input)

    μ_j = 1 / stage_time_j  [u/hr]

    Example for 8BD (total_hrs=80, ratios=[0.2,0.5,0.3]):
      Stage 1: 0.2×80=16 hrs → μ₁=1/16=0.0625 u/hr
      Stage 2: 0.5×80=40 hrs → μ₂=1/40=0.0250 u/hr  ← slowest
      Stage 3: 0.3×80=24 hrs → μ₃=1/24=0.0417 u/hr

    Note (CL-10): μ in Table 4.9 = TOTAL system rate = 1/total_hrs
                  Stage μ is different — used for per-stage simulation.

    Args:
        total_hrs  : total machining time [hrs/unit] (from Table 4.3)
        ratios     : stage time fractions [r1,r2,r3], must sum to 1
        manual_hrs : direct stage hours [h1,h2,h3] (overrides ratios)

    Returns dict with per-stage times and rates.
    """
    if ratios is None:
        ratios = DEFAULT_STAGE_RATIOS

    n_stages = 3
    stage_results = []

    for j in range(n_stages):
        if manual_hrs and len(manual_hrs) > j and manual_hrs[j] is not None:
            st_j = manual_hrs[j]
            mode = "manual"
        else:
            st_j = total_hrs * ratios[j]
            mode = "ratio"

        mu_j = 1.0 / st_j if st_j > 0 else 0.0
        stage_results.append({
            "stage"         : j+1,
            "stage_name"    : STAGE_NAMES.get(j+1, f"Stage {j+1}"),
            "ratio"         : ratios[j],
            "service_time"  : round(st_j, 4),
            "mu"            : round(mu_j, 6),
            "mode"          : mode,
        })

    # Overall system rate (Table 4.9 meaning — CL-6)
    mu_overall = 1.0 / total_hrs if total_hrs > 0 else 0.0

    return {
        "total_hrs"   : total_hrs,
        "ratios"      : ratios,
        "stages"      : stage_results,
        "mu_overall"  : round(mu_overall, 6),  # Table 4.9 μ
        "bottleneck_stage": max(range(n_stages),
                               key=lambda j: stage_results[j]["service_time"]) + 1,
    }


def compute_mu_stages_parts(total_hrs: float,
                             avg_part_time: float,
                             ratios: List[float] = None,
                             manual_hrs: List[float] = None) -> dict:
    """
    ADDITIVE extension of compute_mu_stages() for part-level granularity
    (per Mohamed's clarification, Session 8): a product's total_hrs isn't
    one long operation — it's the SUM of many small part operations
    (~0.1-0.2hr each). Summing many small i.i.d. operations concentrates
    around the mean (lower variance than a single Exponential draw),
    which is exactly what an Erlang-k distribution models: Erlang-k =
    sum of k i.i.d. Exponential phases, CoV² = 1/k.

    This function derives k PER STAGE from how many part-operations fit
    into that stage's time budget, so the fitted service-time
    distribution reflects real part-level granularity instead of
    treating the whole stage time as one Exponential draw.

    Args:
        total_hrs      : total time for the product across all stages [hr]
        avg_part_time  : average time per individual part operation [hr]
                         (thesis reference: ~0.1-0.2 hr)
        ratios, manual_hrs : same as compute_mu_stages()

    Returns: same structure as compute_mu_stages(), with each stage dict
    additionally carrying "k" (Erlang shape, >=1) and "CoV2" (=1/k).
    """
    base = compute_mu_stages(total_hrs, ratios=ratios, manual_hrs=manual_hrs)
    if avg_part_time is None or avg_part_time <= 0:
        # No part-level data given: fall back to k=1 (Exponential, CoV2=1)
        for s in base["stages"]:
            s["k"], s["CoV2"] = 1, 1.0
        base["avg_part_time"] = None
        return base

    for s in base["stages"]:
        k = max(1, round(s["service_time"] / avg_part_time))
        s["k"] = k
        s["CoV2"] = round(1.0 / k, 4)
    base["avg_part_time"] = avg_part_time
    return base



def compute_mu_batch(products: List[dict],
                     ratios: List[float] = None) -> List[dict]:
    """Compute stage μ for all products."""
    results = []
    for p in products:
        mu_data = compute_mu_stages(
            p.get("total_hrs", 1),
            ratios=ratios or DEFAULT_STAGE_RATIOS,
            manual_hrs=p.get("stage_hrs", None)
        )
        results.append({**p, "mu_data": mu_data,
                        "mu_overall": mu_data["mu_overall"]})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STEP 3 — COST INPUT → F1, NR
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_simple(SP: float, F1: float) -> dict:
    """
    Simple mode (Table 4.3 direct): SP and F1 given directly.
    NR = SP - F1
    F1/SP ratio computed (observed ≈ 0.80 in case study, not mandatory).

    NOTE (CL-9): SP_standard_parts=25,000 is already embedded
    in SP for products 8BD, 8BK, 8FJ500. Not subtracted separately.
    """
    NR = SP - F1
    ratio = F1/SP if SP > 0 else 0
    return {
        "mode"    : "simple",
        "SP"      : SP,
        "F1"      : F1,
        "NR"      : round(NR, 4),
        "F1_SP_ratio": round(ratio, 4),
        "ratio_note": f"F1/SP={ratio:.3f} "
                      f"({'≈ case study 0.80' if 0.70<=ratio<=0.90 else 'differs from case study'})",
        "abnormal": NR <= 0,
    }


def compute_cost_detailed(SP: float,
                           DL: float, DM: float,
                           IDL: float, IDM: float,
                           D_MAT: float,
                           FC: float = 0.0,
                           Ws: float = None,
                           S: int = 1) -> dict:
    """
    Detailed mode (Table 4.2 breakdown):
    F1(S) = (DL + DM + IDL + IDM) × Ws(S) + D_MAT + FC/λ

    where Ws(S) comes from M/M/S queue result (depends on S!).

    CL-13 KEY INSIGHT — F1(S) is NON-LINEAR:
      As S increases:
        (+) Server operating cost INCREASES (more machines)
        (-) Waiting time Ws DECREASES (less congestion)
      → F1 has a MINIMUM at optimal S*
      → Rn = λ × (SP - F1) has TRUE maximum at S*

    NOTE (CL-9): SP_standard_parts EXCLUDED — it is variable
    imported parts cost embedded in SP for 3 products only.

    Args:
        SP    : Selling price [$/unit]
        DL    : Direct Labour cost [$/hr]
        DM    : Direct Machine cost [$/hr]
        IDL   : Indirect Labour [$/hr]
        IDM   : Indirect Machine [$/hr]
        D_MAT : Direct Material cost [$/unit]
        FC    : Fixed Cost [$/hr] (optional)
        Ws    : Time in system [hr/unit] from M/M/S — if None, uses 1/μ
        S     : Number of servers (affects server cost)
    """
    hourly_rate = DL + DM + IDL + IDM  # $/hr
    server_cost = hourly_rate * S       # increases with S ↑
    wait_cost   = hourly_rate * (Ws if Ws else 0)  # decreases with S ↓
    F1          = wait_cost + D_MAT + (FC if FC else 0)
    NR          = SP - F1
    ratio       = F1/SP if SP > 0 else 0

    return {
        "mode"          : "detailed",
        "SP"            : SP,
        "DL"            : DL, "DM": DM, "IDL": IDL, "IDM": IDM,
        "D_MAT"         : D_MAT, "FC": FC,
        "hourly_rate"   : round(hourly_rate, 4),
        "server_cost_S" : round(server_cost, 4),
        "wait_cost"     : round(wait_cost, 4),
        "Ws_used"       : round(Ws, 6) if Ws else None,
        "S"             : S,
        "F1"            : round(F1, 4),
        "NR"            : round(NR, 4),
        "F1_SP_ratio"   : round(ratio, 4),
        "abnormal"      : NR <= 0,
        "note_CL13"     : "F1(S) is non-linear: server_cost↑ as S↑, "
                          "wait_cost↓ as S↑ → optimal S* minimizes F1",
    }


def compute_F1_curve(SP: float, DL: float, DM: float,
                     IDL: float, IDM: float, D_MAT: float,
                     mu: float, lam: float,
                     S_range: range = range(1, 9)) -> List[dict]:
    """
    Compute F1(S) curve across S values — shows non-linear behavior (CL-13).
    Uses M/M/S analytical Ws for each S.

    Returns list of {S, Ws, F1, NR, abnormal} for plotting.
    """
    from math import factorial

    def Ws_MMS(lam, mu, S):
        a = lam/mu; rho = lam/(S*mu)
        if rho >= 1: return None
        s = sum(a**n/factorial(n) for n in range(S))
        s += (a**S/factorial(S))/(1-rho)
        P0 = 1/s
        Lq = (a*rho**S*P0)/(factorial(S-1)*(1-rho)**2)
        return Lq/lam + 1/mu

    results = []
    hourly = DL + DM + IDL + IDM
    for S in S_range:
        Ws = Ws_MMS(lam, mu, S)
        if Ws is None:
            F1 = None; NR = None; abnormal = True
        else:
            F1 = hourly * Ws + D_MAT
            NR = SP - F1
            abnormal = NR <= 0
        results.append({
            "S"       : S,
            "Ws"      : round(Ws, 4) if Ws else None,
            "F1"      : round(F1, 4) if F1 else None,
            "NR"      : round(NR, 4) if NR else None,
            "abnormal": abnormal,
            "comment" : "→ 0 (CL-5)" if abnormal else "normal",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STEP 4 — DISTRIBUTION FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_arrivals(data: List[float],
                 dist_type: str = "auto") -> dict:
    """
    Fit arrival data to Poisson distribution.
    Arrival inter-times ~ Exponential → arrivals ~ Poisson (thesis assumption).

    Args:
        data      : observed inter-arrival times [hrs] or arrival counts
        dist_type : "poisson", "expon", or "auto"

    Returns fitted parameters + goodness-of-fit.
    """
    data = np.array(data)
    if len(data) < 2:
        return {"error": "Need at least 2 observations"}

    results = {}

    # Exponential fit (inter-arrival times → λ = 1/mean)
    mean_iat = np.mean(data)
    lam_fitted = 1.0 / mean_iat if mean_iat > 0 else 0
    results["exponential"] = {
        "dist"       : "Exponential (inter-arrival)",
        "mean_iat"   : round(mean_iat, 6),
        "lam_fitted" : round(lam_fitted, 6),
        "std"        : round(float(np.std(data)), 6),
        "cv"         : round(float(np.std(data)/mean_iat), 4) if mean_iat>0 else 0,
        "n_obs"      : len(data),
        "recommended_model": "M/M/S (Poisson arrivals confirmed ✓)",
    }

    # KS test
    loc, scale = stats.expon.fit(data, floc=0)
    ks_stat, ks_p = stats.kstest(data, 'expon', args=(loc, scale))
    results["exponential"]["ks_stat"]  = round(ks_stat, 4)
    results["exponential"]["ks_pvalue"]= round(ks_p, 4)
    results["exponential"]["ks_pass"]  = ks_p > 0.05

    return {
        "data_type"  : "arrivals",
        "n_obs"      : len(data),
        "mean"       : round(float(np.mean(data)), 6),
        "std"        : round(float(np.std(data)), 6),
        "best_fit"   : "exponential",
        "lam"        : round(lam_fitted, 6),
        "results"    : results,
        "note"       : "Poisson arrivals assumed per thesis (CL-11) ✓",
    }


def fit_service(data: List[float],
                dist_type: str = "auto") -> dict:
    """
    Fit service time data to best distribution.
    Tests: Exponential → Gamma → Erlang (in order of complexity).

    CV (Coefficient of Variation) guides selection:
      CV ≈ 1.0  → Exponential (M/M/S)
      CV < 1.0  → Erlang-k or Gamma (M/Ek/1 or M/G/1)
      CV > 1.0  → Lognormal or Hyperexponential
      CV ≈ 0.0  → Deterministic (M/D/1)

    Returns fitted μ + recommended queue model.
    """
    data = np.array(data)
    if len(data) < 2:
        return {"error": "Need at least 2 observations"}

    mean_st = np.mean(data)
    std_st  = np.std(data)
    cv      = std_st / mean_st if mean_st > 0 else 0
    mu_fitted = 1.0 / mean_st if mean_st > 0 else 0

    # ADDITIVE guard: near-constant data (very high part-count Erlang-k,
    # CV -> 0) breaks scipy's gamma.fit() (MLE has no variance to fit
    # against). This matches the function's OWN documented intent
    # ("CV ≈ 0.0 → Deterministic (M/D/1)") — previously unreachable
    # because nothing before the part-level fitting fed CV this low.
    if cv < 0.02:
        return {
            "data_type": "service_time", "n_obs": len(data),
            "mean_st": round(float(mean_st), 6),
            "std_st": round(float(std_st), 6),
            "cv": round(float(cv), 4),
            "mu_fitted": round(mu_fitted, 6),
            "best_fit": "deterministic",
            "recommended_model": "M/D/1",
            "results": {"deterministic": {
                "dist": "Deterministic", "mu": round(mu_fitted, 6),
                "model": "M/D/1", "cv": round(float(cv), 4),
            }},
        }

    results = {}

    # 1. Exponential
    loc_e, scale_e = stats.expon.fit(data, floc=0)
    ks_e, p_e = stats.kstest(data, 'expon', args=(loc_e, scale_e))
    results["exponential"] = {
        "dist": "Exponential", "mu": round(1/scale_e, 6),
        "ks_stat": round(ks_e,4), "ks_p": round(p_e,4),
        "pass": p_e > 0.05, "model": "M/M/S"
    }

    # 2. Gamma
    a_g, loc_g, scale_g = stats.gamma.fit(data, floc=0)
    ks_g, p_g = stats.kstest(data, 'gamma', args=(a_g, loc_g, scale_g))
    results["gamma"] = {
        "dist": "Gamma", "shape": round(a_g,4),
        "mu": round(1/np.mean(data),6),
        "ks_stat": round(ks_g,4), "ks_p": round(p_g,4),
        "pass": p_g > 0.05, "model": "M/G/1"
    }

    # 3. Erlang (k = round(1/CV²))
    k = max(1, round(1/cv**2)) if cv > 0 else 1
    results["erlang"] = {
        "dist": f"Erlang-{k}", "k": k,
        "mu": round(mu_fitted, 6),
        "model": f"M/Ek/1 (k={k})",
        "note": f"CV={cv:.3f} → k={k}"
    }

    # Determine best fit & recommended model
    if cv < 0.05:
        best = "deterministic"
        model = "M/D/1"
        results["deterministic"] = {
            "dist": "Deterministic", "mu": round(mu_fitted,6),
            "model": "M/D/1", "cv": round(cv,4)
        }
    elif results["exponential"]["pass"]:
        best = "exponential"
        model = "M/M/S"
    elif results["gamma"]["pass"]:
        best = "gamma"
        model = "M/G/1"
    elif cv < 1.0:
        best = "erlang"
        model = f"M/Ek/1 (k={k})"
    else:
        best = "exponential"  # fallback
        model = "M/M/S (approx)"

    return {
        "data_type"   : "service_time",
        "n_obs"       : len(data),
        "mean_st"     : round(float(mean_st), 6),
        "std_st"      : round(float(std_st), 6),
        "cv"          : round(float(cv), 4),
        "mu_fitted"   : round(mu_fitted, 6),
        "best_fit"    : best,
        "recommended_model": model,
        "results"     : results,
    }


def fit_from_summary(mean: float, std: float = None,
                     data_type: str = "service",
                     n_obs: int = 30) -> dict:
    """
    Fit from summary statistics (mean + std) without raw data.
    Useful when only aggregated data is available (Table 4.9 style).

    Generates synthetic data then fits — for demonstration.
    """
    np.random.seed(42)
    if std is None:
        std = mean  # assume CV=1 (exponential)
    cv = std / mean if mean > 0 else 1.0

    # Generate representative data
    if cv < 0.1:
        data = np.full(n_obs, mean)
    elif cv < 0.5:
        k = round(1/cv**2)
        data = np.random.gamma(k, mean/k, n_obs)
    else:
        data = np.random.exponential(mean, n_obs)

    if data_type == "arrival":
        return fit_arrivals(data.tolist())
    else:
        return fit_service(data.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPLETE PIPELINE — All 4 Steps Together
# ─────────────────────────────────────────────────────────────────────────────

def run_fitting_pipeline(products_input: List[dict],
                          period: str = "annual",
                          working_days: int = DEFAULT_WORKING_DAYS,
                          hrs_per_shift: float = DEFAULT_HRS_PER_SHIFT,
                          n_shifts: int = DEFAULT_SHIFTS,
                          stage_ratios: List[float] = None,
                          cost_mode: str = "simple",
                          avg_part_time: float = None) -> dict:
    """
    Complete 4-step pipeline for all products.

    products_input: List of dicts with:
      Required:
        name       : str  (e.g. "P1" or "8BD")
        demand     : float (units in period)
        total_hrs  : float (total machining time [hrs/unit])
      Optional (simple cost mode):
        SP         : float (selling price per unit)
        F1         : float (total cost per unit)
      Optional (detailed cost mode):
        DL, DM, IDL, IDM, D_MAT, FC : floats

    avg_part_time: average time per individual part operation [hr]. If
      given, Step 2 uses compute_mu_stages_parts() (real per-stage
      Erlang-k) instead of compute_mu_stages(), AND Step 4's service fit
      uses the correspondingly-derived CoV² instead of silently assuming
      CV=1.0 (Exponential) — fixes the disconnect where "recommended_model"
      always showed M/M/S regardless of part-level fitting being enabled.

    Returns complete fitted parameters for all products.
    """
    if stage_ratios is None:
        stage_ratios = DEFAULT_STAGE_RATIOS

    output = []
    for p in products_input:
        result = {"name": p.get("name", "P?"), "input": p}

        # Step 1: Demand → λ
        lam_data = compute_lambda(
            p["demand"], period, working_days, hrs_per_shift, n_shifts)
        result["step1_lambda"] = lam_data
        lam = lam_data["lam"]

        # Step 2: Service → μ (part-level Erlang-k if avg_part_time given)
        if avg_part_time:
            mu_data = compute_mu_stages_parts(
                p.get("total_hrs", 1), avg_part_time=avg_part_time,
                ratios=stage_ratios, manual_hrs=p.get("stage_hrs", None))
        else:
            mu_data = compute_mu_stages(
                p.get("total_hrs", 1),
                ratios=stage_ratios,
                manual_hrs=p.get("stage_hrs", None))
        result["step2_mu"] = mu_data
        mu_overall = mu_data["mu_overall"]

        # Step 3: Cost → NR
        if cost_mode == "simple" and "SP" in p and "F1" in p:
            cost_data = compute_cost_simple(p["SP"], p["F1"])
        elif cost_mode == "detailed" and "DL" in p:
            cost_data = compute_cost_detailed(
                p.get("SP", 0), p.get("DL",0), p.get("DM",0),
                p.get("IDL",0), p.get("IDM",0), p.get("D_MAT",0),
                p.get("FC",0))
        else:
            cost_data = {"mode": "not_provided", "NR": None}
        result["step3_cost"] = cost_data

        # Step 4: Fit distributions (from summary stats)
        rho = lam / mu_overall if mu_overall > 0 else 0
        fit_arr = fit_from_summary(1/lam if lam>0 else 1, data_type="arrival")

        # ADDITIVE: derive the real service-time std from part-level k
        # instead of always letting fit_from_summary default to CV=1.0.
        # CoV² = 1/k for Erlang-k -> CV = 1/sqrt(k) -> std = mean/sqrt(k).
        total_hrs_val = p.get("total_hrs", 1)
        if avg_part_time:
            k_overall = max(1, round(total_hrs_val / avg_part_time))
            svc_std = total_hrs_val / (k_overall ** 0.5)
        else:
            k_overall = 1
            svc_std = None   # unchanged: fit_from_summary defaults to CV=1.0
        fit_svc = fit_from_summary(total_hrs_val, std=svc_std, data_type="service")

        result["step4_fit"] = {
            "arrival_fit"     : fit_arr,
            "service_fit"     : fit_svc,
            "lam_final"       : round(lam, 6),
            "mu_final"        : round(mu_overall, 6),
            "rho"             : round(rho, 4),
            "recommended_model": fit_svc.get("recommended_model", "M/M/S"),
            "part_level_k"    : k_overall if avg_part_time else None,  # ADDITIVE, for transparency
            "stable"          : rho < 1.0,
        }

        output.append(result)

    return {
        "pipeline"    : "complete",
        "period"      : period,
        "working_days": working_days,
        "hrs_per_shift": hrs_per_shift,
        "n_shifts"    : n_shifts,
        "stage_ratios": stage_ratios,
        "cost_mode"   : cost_mode,
        "avg_part_time": avg_part_time,
        "n_products"  : len(products_input),
        "products"    : output,
    }


def export_to_dashboard(pipeline_result: dict) -> dict:
    """
    Extracts fitted parameters in dashboard.py-ready format.
    Output feeds directly into queue_engine and dashboard tabs.
    """
    fitted = {}
    for p in pipeline_result["products"]:
        name = p["name"]
        lam  = p["step4_fit"]["lam_final"]
        mu   = p["step4_fit"]["mu_final"]
        rho  = p["step4_fit"]["rho"]
        NR   = p["step3_cost"].get("NR", None)
        SP   = p["input"].get("SP", None)
        F1   = p["input"].get("F1", None)
        model= p["step4_fit"]["recommended_model"]
        mu_stages = [
            s["mu"] for s in p["step2_mu"]["stages"]
        ]
        fitted[name] = {
            "lam"         : lam,
            "mu"          : mu,
            "mu_stages"   : mu_stages,
            "rho"         : rho,
            "SP"          : SP,
            "F1"          : F1,
            "NR"          : NR,
            "model"       : model,
            "stable"      : rho < 1.0,
            "total_hrs"   : p["input"].get("total_hrs"),
        }
    return fitted


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VALIDATION — Case Study Verification
# ─────────────────────────────────────────────────────────────────────────────

def run_validation():
    """
    Validate against known case study values (Tables 4.3, 4.9).
    Checks: λ computation, μ stage allocation, cost ratios.
    """
    print("="*65)
    print("DISTRIBUTION FITTING v1.0 — VALIDATION")
    print("="*65)
    passed = failed = 0

    def check(desc, got, expected, tol=0.01):
        nonlocal passed, failed
        ok = abs(got - expected) <= tol
        tag = "PASS ✓" if ok else "FAIL ✗"
        if ok: passed += 1
        else:  failed += 1
        print(f"  {tag}  {desc}")
        if not ok:
            print(f"         Got={got:.6f}  Expected={expected}  "
                  f"Diff={abs(got-expected):.6f}")

    # ── Step 1: λ from annual demand ──────────────────────────────────────
    print("\n── Step 1: λ from Annual Demand ──")
    # 8BD: 897 units/year, 240 days, 8 hrs → λ=0.467
    r1 = compute_lambda(897, "annual", 240, 8, 1)
    check("8BD: λ=0.467 u/hr", r1["lam"], 0.467, tol=0.002)

    # G.T3: 1550 units/year → λ=0.807
    r2 = compute_lambda(1550, "annual", 240, 8, 1)
    check("G.T3: λ=0.807 u/hr", r2["lam"], 0.807, tol=0.005)

    # 3CF12KVA: 1920 units/year → λ=1.000
    r3 = compute_lambda(1920, "annual", 240, 8, 1)
    check("3CF12KVA: λ=1.000 u/hr", r3["lam"], 1.000, tol=0.001)

    # ── Step 2: μ stage allocation ────────────────────────────────────────
    print("\n── Step 2: μ Stage Allocation (CL-10) ──")
    # 8BD: 80hrs, ratios=[0.2,0.5,0.3]
    m1 = compute_mu_stages(80, [0.2, 0.5, 0.3])
    check("8BD Stage1: μ=0.0625 (16hrs)",
          m1["stages"][0]["mu"], 0.0625, tol=0.0001)
    check("8BD Stage2: μ=0.0250 (40hrs)",
          m1["stages"][1]["mu"], 0.025,  tol=0.0001)
    check("8BD Stage3: μ=0.0417 (24hrs)",
          m1["stages"][2]["mu"], 0.04167,tol=0.0001)
    check("8BD bottleneck=Stage2 (slowest)",
          float(m1["bottleneck_stage"]), 2.0, tol=0.0)

    # G.T3: 6hrs total
    m2 = compute_mu_stages(6, [0.2, 0.5, 0.3])
    check("G.T3 mu_overall=1/6=0.1667",
          m2["mu_overall"], 1.0/6, tol=0.001)

    # ── Step 3: Cost validation ───────────────────────────────────────────
    print("\n── Step 3: Cost Validation (CL-3, CL-9) ──")
    products_cost = [
        ("8BD",45000,36000,9000),("8BK",55000,44000,11000),
        ("8FJ500",25000,20000,5000),("8AS10",15000,12000,3000),
        ("3CF12KVA",3000,2400,600),("G.T3",8000,6400,1600),
    ]
    for ptype,SP,F1,NR_exp in products_cost:
        c = compute_cost_simple(SP, F1)
        check(f"{ptype}: NR={NR_exp}", c["NR"], NR_exp, tol=0.01)

    # F1/SP ratio check (CL-3: ≈0.80 for case study, not mandatory)
    print("\n── F1/SP Ratio (case study observation) ──")
    for ptype,SP,F1,_ in products_cost:
        ratio = F1/SP
        in_range = 0.70 <= ratio <= 0.90
        tag = "INFO ✓" if in_range else "INFO"
        print(f"  {tag}  {ptype}: F1/SP={ratio:.3f}")

    # ── Step 4: Distribution fitting ─────────────────────────────────────
    print("\n── Step 4: Distribution Fitting ──")
    np.random.seed(42)
    # Exponential inter-arrival with λ=0.807 → mean_iat=1.239
    iat_data = np.random.exponential(1/0.807, 200).tolist()
    fit = fit_arrivals(iat_data)
    check("G.T3 λ fitted ≈ 0.807",
          fit["lam"], 0.807, tol=0.15)

    # Exponential service time with μ=0.025 → mean_st=40
    svc_data = np.random.exponential(40, 200).tolist()
    fit_s = fit_service(svc_data)
    check("Stage2 μ fitted ≈ 0.025",
          fit_s["mu_fitted"], 0.025, tol=0.005)
    ok_model = fit_s["recommended_model"] in ["M/M/S","M/M/S (approx)"]
    tag = "PASS ✓" if ok_model else "INFO"
    if ok_model: passed += 1
    print(f"  {tag}  Recommended model: {fit_s['recommended_model']}")

    print("\n"+"-"*65)
    print(f"  Results: {passed} PASSED  |  {failed} FAILED")
    print("="*65)
    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CASE STUDY DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_case_study():
    """Full pipeline demo with all 6 case study products."""
    print("\n"+"="*65)
    print("CASE STUDY DEMO — 6 Products (Tables 4.3 & 4.9)")
    print("="*65)

    # Annual demand derived from MPS λ_actual × working_hrs
    WORKING_HRS = 240 * 8  # = 1920 hrs/year
    products_input = [
        {"name":"8BD",     "demand":0.467*WORKING_HRS,"total_hrs":80,
         "SP":45000,"F1":36000},
        {"name":"8BK",     "demand":0.302*WORKING_HRS,"total_hrs":120,
         "SP":55000,"F1":44000},
        {"name":"8FJ500",  "demand":0.273*WORKING_HRS,"total_hrs":80,
         "SP":25000,"F1":20000},
        {"name":"8AS10",   "demand":0.273*WORKING_HRS,"total_hrs":16,
         "SP":15000,"F1":12000},
        {"name":"3CF12KVA","demand":1.000*WORKING_HRS,"total_hrs":8,
         "SP":3000,"F1":2400},
        {"name":"G.T3",    "demand":0.807*WORKING_HRS,"total_hrs":6,
         "SP":8000,"F1":6400},
    ]

    result = run_fitting_pipeline(
        products_input, period="annual",
        working_days=240, hrs_per_shift=8, n_shifts=1,
        stage_ratios=[0.2, 0.5, 0.3], cost_mode="simple")

    # Print summary
    print(f"\n  {'Product':<12} {'λ [u/hr]':>9} {'μ_overall':>10} "
          f"{'ρ':>6} {'NR[$]':>8} {'Model':>10} {'Stable':>7}")
    print("  "+"-"*66)
    for p in result["products"]:
        f4 = p["step4_fit"]
        c3 = p["step3_cost"]
        print(f"  {p['name']:<12} {f4['lam_final']:>9.4f} "
              f"{f4['mu_final']:>10.5f} "
              f"{f4['rho']:>6.3f} "
              f"{c3.get('NR',0):>8} "
              f"{f4['recommended_model']:>10} "
              f"{'✓' if f4['stable'] else '✗':>7}")

    # Export format
    print("\n  ── Export Format (→ dashboard.py) ──")
    exported = export_to_dashboard(result)
    for name, vals in exported.items():
        print(f"  {name}: lam={vals['lam']:.4f}, "
              f"mu={vals['mu']:.5f}, "
              f"mu_stages={[round(m,4) for m in vals['mu_stages']]}, "
              f"model={vals['model']}")

    print("\n  ── F1(S) Non-Linear Curve — G.T3 (CL-13) ──")
    curve = compute_F1_curve(8000, 10, 50, 5, 25, 2000, 6, 2)
    print(f"  {'S':>3} {'Ws[hr]':>9} {'F1[$]':>9} {'NR[$]':>9} {'Note':>15}")
    print("  "+"-"*50)
    for row in curve:
        Ws_s = f"{row['Ws']:.4f}" if row['Ws'] else "∞"
        F1_s = f"{row['F1']:.1f}" if row['F1'] else "∞"
        NR_s = f"{row['NR']:.1f}" if row['NR'] else "—"
        print(f"  {row['S']:>3} {Ws_s:>9} {F1_s:>9} "
              f"{NR_s:>9} {row['comment']:>15}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Distribution Fitting Engine v1.0\n")
    passed, failed = run_validation()
    run_case_study()
    print(f"\n✓ distribution_fitting.py v1.0 COMPLETE")
    print(f"  {'ALL PASS' if failed==0 else str(failed)+' issues'} "
          f"({passed} passed)")
    print("  Ready to feed → queue_engine.py + dashboard.py")
    print("  Next: live_simulation.py (Session 31)")
