# sensitivity_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
from sim_step2 import run_warehouse_sim

# 1) Load your full orders dataset
orders_df = pd.read_csv("orders.csv", dtype={"sku_slot": str})

# 2) Base simulation parameters (with Poisson arrivals & shorter shift)
default_params = {
    "batch_size":           4,
    "num_pickers":          3,
    "shift_duration":       2 * 3600,      # 2-hour shift
    "num_aisles":           5,
    "bays_per_aisle":       20,
    "cross_aisle_frequency":1,
    "depot_location":       (0, 0),
    # pick-time fixed at 30s
    "pick_time_dist":       "Fixed",
    "pick_time":            30.0,
    "pick_time_mean":       None,
    "pick_time_std":        None,
    "walking_speed":        1.5,
    # **enable Poisson arrivals** at 100 orders/hr
    "order_rate":           100.0 / 3600.0,
    # no breaks/downtime for this sweep
    "break_interval":       0.0,
    "break_duration":       0.0,
    "downtime_chance":      0.0,
    "downtime_duration":    0.0
}

# 3) Parameters to sweep
grid = {
    "batch_size":   list(range(1, 21, 2)),       # 1,3,5,...19
    "num_pickers":  [1, 2, 3, 4, 5, 6, 8, 10],
    "order_size":   [50, 100, 200, 500, 1000],   # now up to 1k orders
}

results = []

for param, values in grid.items():
    for v in values:
        params = default_params.copy()

        # if sweeping order_size, take only first v order_ids
        if param == "order_size":
            subset = orders_df[orders_df["order_id"].isin(
                orders_df["order_id"].unique()[:v]
            )]
        else:
            subset = orders_df.copy()

        # assign the sweep value
        if param != "order_size":
            params[param] = v

        # run the sim
        summary, _ = run_warehouse_sim(params, orders_df=subset)

        results.append({
            "parameter":      param,
            "value":          v,
            "avg_cycle_time": summary["avg_cycle_time"],
            "throughput":     summary["throughput"],
        })

df = pd.DataFrame(results)
df.to_csv("sensitivity_results.csv", index=False)
print("Done. Results in sensitivity_results.csv")

# 4) Plot results
for kpi in ["avg_cycle_time", "throughput"]:
    fig = px.line(df, x="value", y=kpi, color="parameter",
                  markers=True,
                  title=f"Sensitivity of {kpi.replace('_',' ').title()}",
                  labels={"value": "Parameter Value", kpi: kpi.replace('_',' ').title()})
    fig.show()
