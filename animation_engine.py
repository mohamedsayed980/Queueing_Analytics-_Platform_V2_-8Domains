"""
animation_engine.py — v1.0
============================================================
SESSION 1 DELIVERABLE — "New Project" (Animation Engine)
Building on: Queueing-Oriented Capacity Planning System (1999 -> 2026)

PURPOSE
    Plotly-based animated visualization of the 3-stage job shop:
    Stage 1 (Cutting) -> Stage 2 (Punching) -> Stage 3 (Bending)
    Jobs = colored dots flowing between stages.
    Machines = rectangles that flip Idle (green) / Busy (red).

STATUS / IMPORTANT NOTE
    The real `live_simulation.py`, `queue_engine.py`, `capacity_planner.py`
    etc. described in the New Project Abstract were NOT present in this
    session's file system, only the abstract .md itself. So this module
    is built to run STANDALONE today:

      - If `live_simulation.py` exists on the path and exposes a
        `get_snapshots(...)` function returning the schema documented
        below, `load_snapshots()` will use it automatically.
      - Otherwise it falls back to `DemoSimulator`, a lightweight
        time-stepped generator that produces the same schema, so the
        animation engine and Tab 10 wiring are fully testable now.

    Upload the real core/*.py files in a future session and this file
    does not need to change — only `load_snapshots()` will pick them up.

SNAPSHOT SCHEMA (what any data source must return)
    List[dict], one dict per animation frame:
    {
        "t": float,                       # simulation time
        "jobs": [
            {"job_id": str, "product": str, "stage": int (1-3),
             "status": "queue"|"service"|"done", "x": float, "y": float}
        ],
        "machines": [
            {"stage": int, "server_id": int, "status": "idle"|"busy"}
        ],
        "queue_len": {1: int, 2: int, 3: int}
    }

CONFIG (per CL-11 / CL-12, locked clarifications)
    Stage 1 = Group 1 = Cutting  (shears),   S = 5
    Stage 2 = Group 2 = Punching,            S = 3 (+2 recently added)
    Stage 3 = Group 3 = Bending,             S = 5

Author: build session for Mohamed (Eng. Mohammed Mohammed Sayed Mohammed)
============================================================
"""

from __future__ import annotations

import random
import importlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go


# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------

@dataclass
class SimConfig:
    stage_names: Dict[int, str] = field(default_factory=lambda: {
        1: "Stage 1 — Cutting",
        2: "Stage 2 — Punching",
        3: "Stage 3 — Bending",
    })
    servers_per_stage: Dict[int, int] = field(default_factory=lambda: {1: 5, 2: 3, 3: 5})  # CL-11
    products: Dict[str, str] = field(default_factory=lambda: {
        "8BD": "#1f77b4",
        "6AX": "#ff7f0e",
        "STD": "#2ca02c",
    })
    mu_split: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.3])  # CL-6 note
    arrival_rate: float = 0.6          # lambda (jobs / time unit)
    total_service_time: float = 8.0    # avg total mu across 3 stages (hours), demo default
    sim_time: float = 60.0
    dt: float = 0.5                    # snapshot interval
    seed: int = 42


# ------------------------------------------------------------------
# 2. DEMO SIMULATOR (fallback data source — self-contained)
# ------------------------------------------------------------------

class _Job:
    __slots__ = ("job_id", "product", "stage", "status", "remaining")

    def __init__(self, job_id, product, stage_service_times):
        self.job_id = job_id
        self.product = product
        self.stage = 1
        self.status = "queue"
        self.remaining = stage_service_times[0]


class DemoSimulator:
    """
    Lightweight time-stepped (not event-based) simulator. Good enough to
    drive a believable animation; NOT a substitute for the real
    simpy_engine.py, which should replace this once uploaded.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.jobs: List[_Job] = []
        self._next_id = 1
        self.busy: Dict[int, List[Optional[str]]] = {
            s: [None] * n for s, n in cfg.servers_per_stage.items()
        }
        self.queues: Dict[int, List[str]] = {s: [] for s in cfg.servers_per_stage}
        self.done: List[str] = []

    def _stage_times(self):
        base = self.cfg.total_service_time
        return [base * f for f in self.cfg.mu_split]

    def _spawn(self):
        if random.random() < self.cfg.arrival_rate * self.cfg.dt:
            pid = random.choice(list(self.cfg.products))
            j = _Job(f"J{self._next_id}", pid, self._stage_times())
            self._next_id += 1
            self.jobs.append(j)
            self.queues[1].append(j.job_id)

    def _by_id(self, jid):
        return next(j for j in self.jobs if j.job_id == jid)

    def _advance(self):
        for stage, servers in self.busy.items():
            for i, occ in enumerate(servers):
                if occ is not None:
                    j = self._by_id(occ)
                    j.remaining -= self.cfg.dt
                    if j.remaining <= 0:
                        servers[i] = None
                        if stage < 3:
                            j.stage += 1
                            j.status = "queue"
                            j.remaining = self._stage_times()[stage]  # next stage time
                            self.queues[stage + 1].append(j.job_id)
                        else:
                            j.status = "done"
                            self.done.append(j.job_id)
            # pull from queue into free servers
            for i, occ in enumerate(servers):
                if occ is None and self.queues[stage]:
                    jid = self.queues[stage].pop(0)
                    servers[i] = jid
                    self._by_id(jid).status = "service"

    def snapshot(self, t) -> dict:
        jobs_out = []
        x_lane = _compute_x_lane(self.cfg.servers_per_stage.keys())
        for stage, servers in self.busy.items():
            for slot, occ in enumerate(servers):
                if occ:
                    j = self._by_id(occ)
                    jobs_out.append({
                        "job_id": j.job_id, "product": j.product, "stage": stage,
                        "status": "service", "x": x_lane[stage] + 0.6,
                        "y": slot + 0.5,
                    })
        for stage, q in self.queues.items():
            for k, jid in enumerate(q):
                j = self._by_id(jid)
                jobs_out.append({
                    "job_id": j.job_id, "product": j.product, "stage": stage,
                    "status": "queue", "x": x_lane[stage] - 0.8,
                    "y": k * 0.4,
                })
        machines_out = [
            {"stage": s, "server_id": i, "status": "busy" if occ else "idle"}
            for s, servers in self.busy.items() for i, occ in enumerate(servers)
        ]
        return {
            "t": t, "jobs": jobs_out, "machines": machines_out,
            "queue_len": {s: len(q) for s, q in self.queues.items()},
        }

    def run(self) -> List[dict]:
        snaps, t = [], 0.0
        while t <= self.cfg.sim_time:
            self._spawn()
            self._advance()
            snaps.append(self.snapshot(t))
            t += self.cfg.dt
        return snaps


# ------------------------------------------------------------------
# 3. DATA SOURCE ADAPTER — prefers the real live_simulation.py
# ------------------------------------------------------------------

def _convert_live_snapshot(raw: dict, products: List[dict]) -> dict:
    """
    Adapter: converts a LiveSimulation snapshot into animation frame format.

    EXACT MODE (live_simulation.py v1.1+, has "jobs_detail"):
      Uses the real per-job (job_id, product, stage, status) records
      tagged directly on each SimPy Request object — no approximation.

    FALLBACK MODE (older snapshots without "jobs_detail"):
      Approximates per-job product identity via round-robin weighted by
      each product's rho (share of system load) — kept only for backward
      compatibility with snapshots taken before the live_simulation.py
      per-job patch.
    """
    x_lane = _compute_x_lane(s["stage"] for s in raw["stage_status"])
    MAX_DOTS = 25  # cap rendered dots/stage; true count kept in queue_len

    machines_out, queue_len = [], {}
    for s in raw["stage_status"]:
        stage = s["stage"]
        for i in range(s["S"]):
            machines_out.append({"stage": stage, "server_id": i,
                                  "status": "busy" if i < s["busy"] else "idle"})
        queue_len[stage] = s["queue"]

    # ADDITIVE: per-stage average Wq (waiting time), unweighted mean
    # across products' cumulative-so-far stage_Wq. Clearly an average,
    # not a per-job instantaneous value — queue_len above already gives
    # the exact instantaneous count; this adds the time dimension
    # alongside it (previously only the count was surfaced in Tab 10).
    wq_by_stage = {}
    pkis = raw.get("product_kpis", [])
    for stage in x_lane:
        vals = [pk["stage_Wq"][stage-1] for pk in pkis
                if len(pk.get("stage_Wq", [])) >= stage]
        wq_by_stage[stage] = round(sum(vals)/len(vals), 3) if vals else 0.0

    jobs_out = []
    if raw.get("jobs_detail"):
        # EXACT: real job identity, grouped per stage/status for layout
        by_key = defaultdict(list)
        for j in raw["jobs_detail"]:
            by_key[(j["stage"], j["status"])].append(j)
        for stage in x_lane:
            q_jobs = by_key.get((stage, "queue"), [])
            for k, j in enumerate(q_jobs[:MAX_DOTS]):
                jobs_out.append({
                    "job_id": j["job_id"], "product": j["product"], "stage": stage,
                    "status": "queue", "x": x_lane[stage] - 0.8, "y": k * 0.4,
                })
            s_jobs = by_key.get((stage, "service"), [])
            for slot, j in enumerate(s_jobs):
                jobs_out.append({
                    "job_id": j["job_id"], "product": j["product"], "stage": stage,
                    "status": "service", "x": x_lane[stage] + 0.6, "y": slot + 0.5,
                })
    else:
        # FALLBACK: ρ-weighted approximation (legacy snapshots only)
        weights = [max(p.get("rho", 0.1), 0.05) for p in products]
        names = [p["type"] for p in products]
        for s in raw["stage_status"]:
            stage = s["stage"]
            for k in range(min(s["queue"], MAX_DOTS)):
                pname = random.choices(names, weights=weights, k=1)[0]
                jobs_out.append({
                    "job_id": f"S{stage}Q{k}", "product": pname, "stage": stage,
                    "status": "queue", "x": x_lane[stage] - 0.8, "y": k * 0.4,
                })
            for slot in range(s["busy"]):
                pname = random.choices(names, weights=weights, k=1)[0]
                jobs_out.append({
                    "job_id": f"S{stage}B{slot}", "product": pname, "stage": stage,
                    "status": "service", "x": x_lane[stage] + 0.6, "y": slot + 0.5,
                })

    return {"t": raw["sim_time"], "jobs": jobs_out, "machines": machines_out,
            "queue_len": queue_len, "wq_by_stage": wq_by_stage}


def get_product_names(product_mode: str = "Experimental") -> List[str]:
    """
    Returns the actual product type names used by live_simulation.py for
    the given mode — the source of truth for building a matching cfg.
    (Forward-compatible: once Global Product Source (Option E) exists,
    a Tab-9-fitted product name list can be passed straight into
    build_cfg_for_run() below instead of this function's output.)
    """
    live = importlib.import_module("live_simulation")
    products = (live.PRODUCTS_EXP if product_mode == "Experimental"
                else live.PRODUCTS_ACTUAL)
    return [p["type"] for p in products]


_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def build_cfg_for_run(S_stages: List[int], product_names: List[str],
                       stage_names: List[str] = None) -> SimConfig:
    """
    Builds a SimConfig whose `servers_per_stage`, `products`, and
    `stage_names` all ACTUALLY match what was simulated, instead of
    relying on SimConfig()'s demo placeholder defaults (S=[5,3,5],
    products={8BD,6AX,STD}, 3 hardcoded stage names) — which silently
    drop any job whose product name isn't one of those 3, mis-render
    machine slots if S_stages differs from the default, and previously
    (before Session 15's N-stage generalization) would KeyError on any
    stage number not in the hardcoded 3-entry stage_names dict.

    stage_names: optional, one name per stage (e.g. from Table 4.4 —
    ["Plate Shears","Punching","Bending","Welding"]). Defaults to
    "Stage 1".."Stage N" if not given.
    """
    products = {name: _PALETTE[i % len(_PALETTE)]
                for i, name in enumerate(product_names)}
    servers = {i + 1: s for i, s in enumerate(S_stages)}
    names = (stage_names if stage_names and len(stage_names) == len(S_stages)
             else [f"Stage {i+1}" for i in range(len(S_stages))])
    names_dict = {i + 1: n for i, n in enumerate(names)}
    return SimConfig(servers_per_stage=servers, products=products,
                      stage_names=names_dict)


def load_snapshots_from_live(S_stages=None, policy="exhaustive", n_shifts=1,
                              sim_time=1500, warmup=150, snapshot_interval=50,
                              seed=42, product_mode="Experimental",
                              renege_T=1.0, products=None,
                              shifts_per_stage=None,
                              stage_ratios=None, stage_names=None) -> List[dict]:
    """
    Real integration path: runs your actual live_simulation.LiveSimulation
    and converts its aggregate KPI snapshots into animation frames.

    products: if given (e.g. from Tab 9's Global Product Source), used
    directly instead of looking up PRODUCTS_EXP/PRODUCTS_ACTUAL via
    product_mode. Each product dict may carry "k_stages" for Erlang-k
    service sampling (see live_simulation.py's _job_process).

    shifts_per_stage: optional [sh1,sh2,sh3] — per-stage shift count,
    passed straight through to LiveSimulation (see its docstring for why
    this is mathematically identical to manually multiplying S_stages).

    stage_ratios, stage_names: N-stage generalization (Session 15) —
    passed straight through to LiveSimulation. len(S_stages) determines
    N; stage_ratios must match that length and sum to 1.0.
    """
    live = importlib.import_module("live_simulation")
    if products is None:
        products = (live.PRODUCTS_EXP if product_mode == "Experimental"
                    else live.PRODUCTS_ACTUAL)
    sim = live.LiveSimulation(
        products, S_stages=S_stages or live.S_DEFAULT, policy=policy,
        n_shifts=n_shifts, shifts_per_stage=shifts_per_stage,
        stage_ratios=stage_ratios, stage_names=stage_names,
        sim_time=sim_time, warmup=warmup,
        snapshot_interval=snapshot_interval, seed=seed, renege_T=renege_T,
    )
    result = sim.run()
    return [_convert_live_snapshot(s, products) for s in result["snapshots"]]


def load_snapshots(cfg: SimConfig = None, use_real: bool = True) -> List[dict]:
    """
    Tries the real engine first (live_simulation.py), falls back to the
    self-contained demo simulator if it isn't importable.
    """
    cfg = cfg or SimConfig()
    if use_real:
        try:
            return load_snapshots_from_live(
                S_stages=list(cfg.servers_per_stage.values()),
                sim_time=cfg.sim_time * 25,       # scale demo dt(0.5)->hrs
                snapshot_interval=max(cfg.sim_time * 25 / 40, 5),
                seed=cfg.seed,
            )
        except ModuleNotFoundError:
            pass
    return DemoSimulator(cfg).run()


# ------------------------------------------------------------------
# 4. ANIMATION BUILDER
# ------------------------------------------------------------------

STATUS_COLOR = {"idle": "#2ecc71", "busy": "#e74c3c"}  # green / red

def _compute_x_lane(stage_keys) -> Dict[int, float]:
    """
    Dynamically spaces stage lanes left-to-right based on however many
    stages are actually configured. Was hardcoded to exactly 3 fixed
    positions ({1:0, 2:3, 3:6}) in FOUR separate places throughout this
    file — all four now derive it from the real stage count instead
    (Session 15, N-stage generalization, Item 3).
    """
    return {stage: idx * 3.0 for idx, stage in enumerate(sorted(stage_keys))}


def _frame_traces(snap: dict, cfg: SimConfig) -> List[go.Scatter]:
    traces = []
    x_lane = _compute_x_lane(cfg.servers_per_stage.keys())
    # machines (rectangles drawn as large square markers per server slot)
    for stage, n in cfg.servers_per_stage.items():
        mstates = [m for m in snap["machines"] if m["stage"] == stage]
        traces.append(go.Scatter(
            x=[x_lane[stage] + 0.6] * len(mstates),
            y=[m["server_id"] + 0.5 for m in mstates],
            mode="markers",
            marker=dict(symbol="square", size=26,
                        color=[STATUS_COLOR[m["status"]] for m in mstates],
                        line=dict(width=1, color="black")),
            name=f"Servers {cfg.stage_names[stage]}",
            hoverinfo="skip",
            showlegend=False,
        ))
    # jobs, grouped by product for legend/color
    for pname, color in cfg.products.items():
        pj = [j for j in snap["jobs"] if j["product"] == pname]
        traces.append(go.Scatter(
            x=[j["x"] for j in pj], y=[j["y"] for j in pj],
            mode="markers",
            marker=dict(size=10, color=color,
                        line=dict(width=1, color="white")),
            name=pname,
            text=[f"{j['job_id']} ({j['status']})" for j in pj],
            hoverinfo="text",
        ))
    return traces


def build_animation_figure(snapshots: List[dict], cfg: SimConfig = None,
                            frame_duration_ms: int = 300) -> go.Figure:
    cfg = cfg or SimConfig()
    if not snapshots:
        raise ValueError("No snapshots to animate — check the data source.")

    x_lane = _compute_x_lane(cfg.servers_per_stage.keys())
    n_stages = len(x_lane)
    base_traces = _frame_traces(snapshots[0], cfg)
    def _q_label(st, s):
        n = s["queue_len"][st]
        wq = s.get("wq_by_stage", {}).get(st, 0.0)  # graceful default for old snapshots
        busy_frac = sum(1 for m in s["machines"] if m["stage"] == st and m["status"] == "busy")
        total = cfg.servers_per_stage[st]
        sat = " ⚠️ SATURATED" if busy_frac == total and n > total * 2 else ""
        return f"Q{st}: {n}  |  Wq≈{wq:.2f}hr{sat}"

    frames = [
        go.Frame(data=_frame_traces(s, cfg), name=f"t={s['t']:.1f}",
                  layout=go.Layout(
                      annotations=[
                          dict(x=x_lane[st] + 0.6, y=-1.2,
                               text=_q_label(st, s),
                               showarrow=False, font=dict(size=12))
                          for st in cfg.servers_per_stage
                      ]))
        for s in snapshots
    ]

    fig = go.Figure(data=base_traces, frames=frames)

    max_slots = max(cfg.servers_per_stage.values())
    title = (f"Job Shop — Live Animated Flow "
              f"({' -> '.join('Stage ' + str(s) for s in sorted(x_lane))})"
             if n_stages != 3 else "Job Shop — Live Animated Flow (Stage 1 -> 2 -> 3)")
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-2, max(x_lane.values(), default=6.0) + 2],
                   tickvals=list(x_lane.values()),
                   ticktext=[cfg.stage_names[s] for s in x_lane], title=""),
        yaxis=dict(range=[-2, max_slots + 1], title="Server slot / queue position"),
        height=520,
        legend_title="Product type",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.12,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate"}]),
                dict(label="Reset", method="animate",
                     args=[[frames[0].name], {"frame": {"duration": 0, "redraw": True},
                                               "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[f.name],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        label=f.name) for f in frames],
            x=0.0, y=-0.05, len=1.0,
        )],
        annotations=[
            dict(x=x_lane[st] + 0.6, y=-1.2, text=_q_label(st, snapshots[0]),
                 showarrow=False, font=dict(size=12))
            for st in cfg.servers_per_stage
        ],
    )
    return fig


# ------------------------------------------------------------------
# 5. STANDALONE DEMO
# ------------------------------------------------------------------

if __name__ == "__main__":
    cfg = SimConfig()
    snaps = load_snapshots(cfg)
    fig = build_animation_figure(snaps, cfg, frame_duration_ms=250)
    out_path = "/mnt/user-data/outputs/animation_demo.html"
    fig.write_html(out_path, auto_play=False)
    print(f"Generated {len(snaps)} frames -> {out_path}")
