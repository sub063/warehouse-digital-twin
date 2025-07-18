import pandas as pd
import numpy as np
from sim_step2 import run_warehouse_sim

# Pre-load your base orders
_orders = pd.read_csv("orders.csv", dtype={"sku_slot": str})

def monte_carlo(n=100, params=None, progress_callback=None):
    """
    Runs n Monte Carlo draws, invoking progress_callback(i, n) after each draw.
    Returns list of average distances per order.
    """
    results = []
    for i in range(1, n+1):
        # Resample orders (you can customize this logic)
        spike  = np.random.lognormal(mean=0, sigma=0.3)
        sample = _orders.sample(frac=spike, replace=True)

        # Run simulation on the resampled orders
        summary, _ = run_warehouse_sim({**params, **{"orders_df": sample}})

        # Compute avg distance per order
        thru = summary["throughput"]
        dist = summary["total_distance"]
        avg_dist = dist / thru if thru > 0 else 0
        results.append(avg_dist)

        # Update progress bar in the dashboard
        if progress_callback:
            progress_callback(i, n)

    return results
