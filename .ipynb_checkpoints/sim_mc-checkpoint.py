import pandas as pd
import numpy as np
from layout import build_layout
import networkx as nx
import simpy

# ─── 1. Load orders & precompute distances ──────────────
orders = pd.read_csv("orders.csv")
layout = build_layout()
dist = dict(nx.all_pairs_dijkstra_path_length(layout, weight="weight"))
PICK_TIME = 6  # seconds per pick

# ─── 2. Single‐run simulation function ──────────────────
def run_sim(order_df):
    KPIS = {"distance": 0, "lines": 0}

    def picker(env, rows):
        travel = 0
        for _, row in rows.iterrows():
            frm, to = eval(row["current_slot"]), eval(row["sku_slot"])
            travel += dist[frm][to]
            yield env.timeout(dist[frm][to])  # walk
            yield env.timeout(PICK_TIME)      # pick
        KPIS["distance"] += travel
        KPIS["lines"]    += len(rows)

    def shift(env):
        for oid in order_df["order_id"].unique():
            batch = order_df[order_df.order_id == oid]
            env.process(picker(env, batch))
        yield env.timeout(8 * 3600)  # 8-hour shift

    env = simpy.Environment()
    env.process(shift(env))
    env.run()

    return KPIS["distance"] / order_df["order_id"].nunique()

# ─── 3. Monte-Carlo wrapper ─────────────────────────────
def monte_carlo(n=10):
    results = []
    for i in range(n):
        spike  = np.random.lognormal(mean=0, sigma=0.3)
        sample = orders.sample(frac=spike, replace=True)
        avg    = run_sim(sample)
        results.append(avg)
    return results

# ─── 4. Run & save results ─────────────────────────────
if __name__ == "__main__":
    sims = 20
    distances = monte_carlo(sims)
    df = pd.DataFrame({"avg_distance": distances})
    print(df)
    df.to_csv("mc_results.csv", index=False)
    print(f"\nSaved Monte-Carlo results to mc_results.csv ({sims} runs)")
