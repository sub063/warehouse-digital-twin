import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import time 
from collections import Counter

from sim_step2 import run_warehouse_sim
from sim_mc import monte_carlo
from layout import build_layout_graph

# --- Page setup ---
st.set_page_config(
    page_title="Warehouse Simulator Control Center",
    layout="wide",
)

# --- Custom CSS (optional) ---
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0D1B2A, #1B263B); }
      .stSidebar { background-color: #1B263B; }
      .stButton>button { background-color: #E63946 !important; color: white !important; border-radius: 8px; }
      .stSlider>div>div>div>button { background-color: #E63946 !important; border: none; }
      h1,h2,h3 { color: #EAEAEA !important; }
      .stMarkdown { color: #A8DADC !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load & cache orders.csv ---
@st.cache_data
def load_orders():
    return pd.read_csv("orders.csv", dtype={"sku_slot": str})
orders_df = load_orders()

# --- Preset templates ---
st.sidebar.title("⚙️ Settings & Presets")
presets = {
    "Default": {
        "n_aisles": 5, "bays": 20, "cross_freq": 1,
        "batch": 4, "pickers": 3,
        "pick_dist": "Fixed", "pick_mean": 30.0, "pick_cv": 50,
        "shift_hrs": 8, "walk_speed": 1.5,
        "arrival_rate": 60.0,
        "congestion": 50,
        "order_size": 50,
        "mc_runs": 200
    },
    "High-throughput": {
        "n_aisles": 10, "bays": 40, "cross_freq": 2,
        "batch": 10, "pickers": 8,
        "pick_dist": "Fixed", "pick_mean": 25.0, "pick_cv": 20,
        "shift_hrs": 10, "walk_speed": 1.8,
        "arrival_rate": 300.0,
        "congestion": 20,
        "order_size": 200,
        "mc_runs": 500
    },
    "Slow-moving SKUs": {
        "n_aisles": 5, "bays": 20, "cross_freq": 1,
        "batch": 2, "pickers": 2,
        "pick_dist": "Log-normal", "pick_mean": 40.0, "pick_cv": 80,
        "shift_hrs": 8, "walk_speed": 1.2,
        "arrival_rate": 20.0,
        "congestion": 80,
        "order_size": 100,
        "mc_runs": 100
    }
}
sel = st.sidebar.selectbox("Choose a preset", list(presets.keys()))
sp = presets[sel]

# --- Sidebar Controls ---
st.sidebar.title("🔧 Layout & Simulation Settings")

with st.sidebar.expander("Layout Dimensions", expanded=True):
    n_aisles       = st.slider("Number of aisles", 1, 20, sp["n_aisles"])
    bays_per_aisle = st.slider("Bays per aisle", 1, 100, sp["bays"])
    cross_freq     = st.slider("Cross-aisle frequency", 0, 10, sp["cross_freq"])

with st.sidebar.expander("Order & Picker Settings", expanded=True):
    batch_size   = st.slider("Batch size", 1, 20, sp["batch"])
    picker_count = st.slider("Picker count", 1, 10, sp["pickers"])
    single_pick  = st.checkbox("Single-pick mode: each pick = one order")

    order_size = (
        st.slider("Number of picks", 1, min(1000, len(orders_df)), sp["order_size"])
        if single_pick else
        st.slider("Number of orders", 1, len(orders_df["order_id"].unique()), sp["order_size"])
    )
    subset = (
        orders_df.head(order_size)
        if single_pick else
        orders_df[orders_df["order_id"].isin(orders_df["order_id"].unique()[:order_size])]
    )
    st.markdown(f"**Unique bins:** {subset['sku_slot'].nunique()}")

with st.sidebar.expander("Pick Time Distribution", expanded=False):
    pick_dist = st.selectbox(
        "Pick-time dist",
        ["Fixed", "Log-normal"],
        index=["Fixed","Log-normal"].index(sp["pick_dist"])
    )
    pick_mean = st.number_input("Avg pick time (s)", 0.1, 300.0, float(sp["pick_mean"]))
    if pick_dist == "Log-normal":
        cv = st.slider("Variability (%)", 0, 200, sp["pick_cv"])
        pick_std = pick_mean * cv / 100.0
        mu    = np.log((pick_mean**2)/np.sqrt(pick_std**2+pick_mean**2))
        sigma = np.sqrt(np.log(1 + (pick_std**2)/(pick_mean**2)))
        dfp = pd.DataFrame({"t": np.random.lognormal(mu,sigma,500)})
        st.sidebar.altair_chart(
            alt.Chart(dfp).mark_bar(opacity=0.7)
              .encode(alt.X("t:Q", bin=alt.Bin(maxbins=30)), alt.Y("count()"))
              .properties(width=300, height=150, title="Pick-time preview")
        )
    else:
        pick_std = 0.0

with st.sidebar.expander("Shift & Travel", expanded=False):
    shift_hours   = st.slider("Shift length (hrs)", 1, 24, sp["shift_hrs"])
    walking_speed = st.number_input("Walking speed (m/s)", 0.5, 3.0, float(sp["walk_speed"]))

with st.sidebar.expander("Arrivals & Congestion", expanded=False):
    arrival_rate_hr      = st.number_input("Orders/hr", 0.0, 10000.0, float(sp["arrival_rate"]))
    congestion_pct       = st.slider("Slowdown (% per extra picker)", 0, 200, sp["congestion"])
    congestion_slowdown  = congestion_pct / 100.0

with st.sidebar.expander("Breaks & Downtime", expanded=False):
    break_interval_min   = st.number_input("Break interval (min)", 0.0, 180.0, 60.0)
    break_duration_min   = st.number_input("Break duration (min)", 0.0, 60.0, 10.0)
    downtime_chance_pct  = st.slider("Equipment downtime chance (%)", 0, 100, 10)
    downtime_chance      = downtime_chance_pct / 100.0
    downtime_duration_min= st.number_input("Downtime duration (min)", 0.0, 60.0, 5.0)

with st.sidebar.expander("Monte Carlo Settings", expanded=False):
    mc_runs = st.number_input("Monte Carlo runs", 1, 2000, sp.get("mc_runs", 200), step=50)

# Input validation
if shift_hours <= 0:
    st.sidebar.error("Shift length must be positive")
if picker_count < 1:
    st.sidebar.error("At least one picker required")

run_btn = st.sidebar.button("🚀 Run scenario")

# --- Session state init ---
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []
if "selected_bins" not in st.session_state:
    st.session_state.selected_bins = set()

# Helper
mean = lambda xs: sum(xs)/len(xs) if xs else 0

# --- Run simulation & profiling ---
if run_btn:
    sim_orders = (
        orders_df.head(order_size).copy() if single_pick else subset.copy()
    )
    if single_pick:
        sim_orders["order_id"] = sim_orders.index.astype(str)

    params = {
        "batch_size":           batch_size,
        "num_pickers":          picker_count,
        "shift_duration":       shift_hours * 3600,
        "num_aisles":           n_aisles,
        "bays_per_aisle":       bays_per_aisle,
        "cross_aisle_frequency":cross_freq,
        "depot_location":       (0, 0),
        "pick_time_dist":       pick_dist.lower(),
        "pick_time":            pick_mean if pick_dist=="Fixed" else None,
        "pick_time_mean":       pick_mean,
        "pick_time_std":        pick_std,
        "walking_speed":        walking_speed,
        "order_rate":           arrival_rate_hr/3600.0,
        "congestion_slowdown":  congestion_slowdown,
        "break_interval":       break_interval_min * 60,
        "break_duration":       break_duration_min * 60,
        "downtime_chance":      downtime_chance,
        "downtime_duration":    downtime_duration_min * 60
    }

    # 1) Main sim
    summary, details = run_warehouse_sim(params, orders_df=sim_orders)

    # 2) Monte Carlo with progress & ETA
    progress = st.progress(0)
    t0 = time.time()
    def mc_callback(i, total):
        progress.progress(i/total)
        elapsed = time.time() - t0
        eta = elapsed * (total/i - 1) if i>0 else 0
        st.sidebar.write(f"MC ETA: {eta:.1f}s")
    distances = monte_carlo(n=mc_runs, params=params, progress_callback=mc_callback)
    t1 = time.time()

    # 3) Store scenario
    name = f"Run {len(st.session_state.scenarios)+1}"
    st.session_state.scenarios.append({
        "name":          name,
        "params":        params,
        "summary":       summary,
        "details":       details,
        "distances":     distances,
        "simulation_time_s": summary.get("simulation_time_s", 0.0),
        "mc_time_s":     t1 - t0
    })

# --- Top KPIs & deltas ---
st.markdown("## 📊 Key Metrics & Profiling")
if st.session_state.scenarios:
    base = st.session_state.scenarios[0]
    last = st.session_state.scenarios[-1]
    bS, lS = base["summary"], last["summary"]
    bD, lD = mean(base["distances"]), mean(last["distances"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg cycle (s)", f"{lS['avg_cycle_time']:.1f}",
              delta=f"{(lS['avg_cycle_time']-bS['avg_cycle_time'])/bS['avg_cycle_time']*100:+.1f}%")
    c2.metric("Throughput",       f"{lS['throughput']}",
              delta=f"{(lS['throughput']-bS['throughput'])/bS['throughput']*100:+.1f}%")
    c3.metric("Sim time (s)",     f"{last.get('simulation_time_s',0.0):.2f}")
    c4.metric("MC time (s)",      f"{last['mc_time_s']:.2f}")
else:
    st.info("No scenarios yet. Click 🚀 Run scenario to begin.")

# --- Detail tabs ---
tabs = st.tabs(["🗺️ Layout","📋 Scenarios","🔄 Cycle-Time","⏱️ Throughput","🎲 Monte Carlo"])

# Layout tab
with tabs[0]:
    st.markdown("## Layout & Storage Placement")
    for a in range(n_aisles):
        cols = st.columns(bays_per_aisle, gap="small")
        for b, col in enumerate(cols):
            lbl = f"{a},{b}"
            sel = lbl in st.session_state.selected_bins
            if col.button("🟧" if sel else "⬜", key=lbl):
                if sel:
                    st.session_state.selected_bins.remove(lbl)
                else:
                    st.session_state.selected_bins.add(lbl)
    st.write(f"Selected bins: {len(st.session_state.selected_bins)}")

# Scenarios tab
with tabs[1]:
    st.markdown("## Scenario List & Metrics")
    if st.session_state.scenarios:
        df = pd.DataFrame([{
            "Name": s["name"],
            "Batch": s["params"]["batch_size"],
            "Pickers": s["params"]["num_pickers"],
            "AvgCycle": s["summary"]["avg_cycle_time"],
            "Throughput": s["summary"]["throughput"],
            "MeanDist": mean(s["distances"])
        } for s in st.session_state.scenarios]).set_index("Name")
        st.dataframe(df)
    else:
        st.info("No scenarios to display.")

# Cycle-Time tab
with tabs[2]:
    st.markdown("## Cycle-Time Distribution")
    if st.session_state.scenarios:
        recs = []
        for s in st.session_state.scenarios:
            for ct in s["details"]["cycle_times"]:
                recs.append({"Scenario": s["name"], "CycleTime": ct})
        fig = px.box(pd.DataFrame(recs),
                     x="Scenario", y="CycleTime",
                     points="all", title="Cycle-Time Spread")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a scenario to view cycle times.")

# Throughput tab
with tabs[3]:
    st.markdown("## Throughput Timeline")
    if st.session_state.scenarios:
        lines = []
        for s in st.session_state.scenarios:
            for t,o in s["details"]["throughput_timeline"]:
                lines.append({"Scenario": s["name"], "Time": t, "Orders": o})
        fig = px.line(pd.DataFrame(lines),
                      x="Time", y="Orders",
                      color="Scenario", markers=True,
                      title="Cumulative Throughput")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a scenario to view throughput.")

# Monte Carlo tab
with tabs[4]:
    st.markdown("## Monte Carlo Distance Distribution")
    if st.session_state.scenarios:
        mc = []
        for s in st.session_state.scenarios:
            for d in s["distances"]:
                mc.append({"Scenario": s["name"], "Distance": d})
        fig = px.histogram(pd.DataFrame(mc),
                           x="Distance", color="Scenario",
                           barmode="overlay",
                           title="Distance Variability")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a scenario to view Monte Carlo results.")
