import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sim_step2 import run_warehouse_sim

# --- 1) Configuration: set simulation parameters to match your historical day ---
params = {
    "batch_size":        5,
    "num_pickers":       3,
    "shift_duration":    8 * 3600,          # 8-hour shift
    "num_aisles":        5,
    "bays_per_aisle":    20,
    "cross_aisle_frequency": 1,
    "depot_location":    (0, 0),
    # pick time (empirical): use fixed or lognormal
    "pick_time_dist":    "Fixed",
    "pick_time":         30.0,
    "pick_time_mean":    None,
    "pick_time_std":     None,
    # travel speed
    "walking_speed":     1.5,
    # no Poisson arrivals for back-test (all orders at t=0)
    "order_rate":        0.0,
    # no breaks or downtime
    "break_interval":    0.0,
    "break_duration":    0.0,
    "downtime_chance":   0.0,
    "downtime_duration": 0.0
}

# --- 2) Load your historical orders & actual WMS logs ---
# Replace these filenames with your real files
orders = pd.read_csv("historical_orders.csv", dtype={"sku_slot": str})
logs   = pd.read_csv(
    "actual_wms_logs.csv",
    parse_dates=["start_time", "end_time"]
)

# Compute actual cycle times and throughput timeline
logs["cycle_time"] = (logs["end_time"] - logs["start_time"]).dt.total_seconds()
# Throughput: cumulative orders completed by minute
logs["minute"] = logs["end_time"].dt.floor("min")
actual_tl = (
    logs.groupby("minute").size()
        .cumsum()
        .reset_index(name="actual_orders")
)

# --- 3) Run simulation on the same orders ---
summary_pred, details_pred = run_warehouse_sim(params, orders_df=orders)
sim_tl_raw = pd.DataFrame(
    details_pred["throughput_timeline"],
    columns=["seconds", "predicted_orders"]
)
# Align simulation times with real timestamps
shift_start = actual_tl["minute"].min()
sim_tl_raw["timestamp"] = sim_tl_raw["seconds"].apply(lambda s: shift_start + timedelta(seconds=s))
sim_tl = sim_tl_raw[["timestamp", "predicted_orders"]]

# --- 4) Error metrics ---
# Cycle-time distribution error
ct_pred = np.array(details_pred["order_cycle_times"])
ct_act  = logs["cycle_time"].values
mae_ct = np.mean(np.abs(np.mean(ct_pred) - np.mean(ct_act)))
# Throughput timeline error at matching times
# Interpolate actual orders to sim timestamps
actual_interp = np.interp(
    (sim_tl_raw["seconds"]),
    (actual_tl["minute"] - shift_start).dt.total_seconds(),
    actual_tl["actual_orders"]
)
mae_tl = np.mean(np.abs(sim_tl_raw["predicted_orders"] - actual_interp))

print(f"MAE cycle-time (s): {mae_ct:.2f}")
print(f"MAE throughput (orders): {mae_tl:.2f}")

# --- 5) Plots: Predicted vs. Actual ---
# Throughput overlay
fig1 = px.line(
    actual_tl, x="minute", y="actual_orders", title="Throughput: Actual vs Predicted"
)
fig1.add_scatter(
    x=sim_tl["timestamp"], y=sim_tl["predicted_orders"], mode="lines", name="Predicted"
)
fig1.show()

# Cycle-time distributions
df_ct = pd.DataFrame({
    "cycle_time": np.concatenate([ct_act, ct_pred]),
    "type": ["actual"]*len(ct_act) + ["predicted"]*len(ct_pred)
})
fig2 = px.box(df_ct, x="type", y="cycle_time", title="Cycle-Time: Actual vs Predicted")
fig2.show()
