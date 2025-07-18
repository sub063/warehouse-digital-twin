import pandas as pd
import networkx as nx
import simpy
import numpy as np
import random
import time
from networkx.algorithms.approximation import greedy_tsp
from layout import build_layout_graph

def run_warehouse_sim(params, orders_df=None):
    """
    Runs a warehouse picking simulation with:
      - Poisson arrivals
      - Nearest-neighbor TSP routing
      - Real travel-time (distance / walking_speed)
      - Fixed or log-normal pick-time
      - Breaks & downtime
    Returns (summary, details).
    """
    # 1) Load orders
    if orders_df is None:
        orders_df = pd.read_csv("orders.csv", dtype={"sku_slot": str})

    # 2) KPI containers
    kpis = {
        "orders_completed":   0,
        "total_distance":     0.0,      # <— ensure this is tracked
        "total_lines_picked": 0,
        "picker_wait_times":  [],
        "order_cycle_times":  [],
        "picker_busy_time":   0.0,
        "throughput_timeline":[] 
    }
    picker_paths = []

    # 3) Build layout & distances
    G, _ = build_layout_graph(
        n_aisles=params["num_aisles"],
        bays_per_aisle=params["bays_per_aisle"],
        cross_aisle_frequency=params["cross_aisle_frequency"]
    )
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    # 4) Prep pick-time distribution
    if params["pick_time_dist"] == "lognormal":
        m, s = params["pick_time_mean"], params["pick_time_std"]
        mu    = np.log((m**2) / np.sqrt(s**2 + m**2))
        sigma = np.sqrt(np.log(1 + (s**2)/(m**2)))
    else:
        mu = sigma = None

    # 5) Picker process (with breaks & downtime)
    def picker(env, batch_rows, arrival_time):
        start_work = env.now
        last_break = env.now
        depot = params["depot_location"]
        current = depot
        to_visit = {eval(loc) for loc in batch_rows["sku_slot"]}

        # build TSP subgraph
        nodes = [depot] + list(to_visit)
        K = nx.Graph()
        K.add_nodes_from(nodes)
        for u in nodes:
            for v in nodes:
                if u != v:
                    K.add_edge(u, v, weight=dist[u][v])
        route = greedy_tsp(K, source=depot, weight="weight")
        if route[-1] != depot:
            route.append(depot)

        for nxt in route[1:]:
            # scheduled break
            if params["break_interval"] > 0 and (env.now - last_break) >= params["break_interval"]:
                yield env.timeout(params["break_duration"])
                last_break = env.now

            # travel
            d = dist[current][nxt]
            t_travel = d / params["walking_speed"]
            yield env.timeout(t_travel)
            kpis["total_distance"] += d   # <— accumulate total_distance
            picker_paths.append((current, nxt))
            current = nxt

            # picking
            if nxt in to_visit:
                lines = (batch_rows["sku_slot"] == str(nxt)).sum()
                if params["pick_time_dist"] == "lognormal":
                    draws = np.random.lognormal(mu, sigma, size=lines)
                    t_pick = float(draws.sum())
                else:
                    t_pick = params["pick_time"] * lines
                yield env.timeout(t_pick)
                kpis["total_lines_picked"] += lines

        # return to depot
        if current != depot:
            d = dist[current][depot]
            yield env.timeout(d / params["walking_speed"])
            kpis["total_distance"] += d

        completion = env.now
        count_orders = batch_rows["order_id"].nunique()
        kpis["orders_completed"] += count_orders
        cycle = completion - arrival_time
        kpis["order_cycle_times"].extend([cycle] * count_orders)
        kpis["picker_busy_time"] += (completion - start_work)

        # throughput timeline
        tl = kpis["throughput_timeline"]
        if not tl or tl[-1][0] < completion:
            tl.append((completion, kpis["orders_completed"]))
        else:
            tl[-1] = (completion, kpis["orders_completed"])

    # 6) Shift generator
    def shift_gen(env, pool, all_orders):
        uids = all_orders["order_id"].unique().tolist()
        i = 0
        while env.now < params["shift_duration"] and i < len(uids):
            lam = params["order_rate"] / params["batch_size"] if params["batch_size"]>0 else 0
            inter = random.expovariate(lam) if lam>0 else 0
            yield env.timeout(inter)

            batch_ids = uids[i : i + params["batch_size"]]
            batch     = all_orders[all_orders["order_id"].isin(batch_ids)]
            arrival   = env.now

            with pool.request() as req:
                yield req
                kpis["picker_wait_times"].append(env.now - arrival)
                yield env.process(picker(env, batch, arrival))

            if random.random() < params["downtime_chance"]:
                yield env.timeout(params["downtime_duration"])

            i += params["batch_size"]

    # 7) Run sim & profile
    env  = simpy.Environment()
    pool = simpy.Resource(env, capacity=params["num_pickers"])
    env.process(shift_gen(env, pool, orders_df))
    t0 = time.time()
    env.run(until=params["shift_duration"])
    t1 = time.time()

    # 8) Summarize
    summary = {
        "batch_size":        params["batch_size"],
        "num_pickers":       params["num_pickers"],
        "throughput":        kpis["orders_completed"],
        "total_distance":    kpis["total_distance"],      # <— expose it here
        "avg_cycle_time":    float(np.mean(kpis["order_cycle_times"])) if kpis["order_cycle_times"] else 0.0,
        "avg_wait_time":     float(np.mean(kpis["picker_wait_times"])) if kpis["picker_wait_times"] else 0.0,
        "utilization":       (kpis["picker_busy_time"] / (params["num_pickers"] * env.now) * 100) if env.now>0 else 0.0,
        "simulation_time_s": t1 - t0
    }
    details = {
        "cycle_times":          kpis["order_cycle_times"],
        "throughput_timeline":  kpis["throughput_timeline"],
        "picker_paths":         picker_paths
    }
    return summary, details
