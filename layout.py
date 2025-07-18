# layout.py

import networkx as nx

def build_layout_graph(n_aisles=5, bays_per_aisle=20, cross_aisle_frequency=1):
    """
    Builds the warehouse graph with parameterized dimensions and weights,
    and returns:
      - G   : a NetworkX Graph of (aisle, bay) nodes
      - pos : dict mapping each node -> (x,y) coordinate for plotting
    """
    G = nx.Graph()

    # 1) Create every bay node
    for a in range(n_aisles):
        for b in range(bays_per_aisle):
            G.add_node((a, b))

    # 2) Connect bays along each aisle
    for a in range(n_aisles):
        for b in range(bays_per_aisle - 1):
            G.add_edge((a, b), (a, b + 1), weight=1)

    # 3) Add cross‐aisles at roughly equal intervals
    if cross_aisle_frequency > 0:
        step = max(1, bays_per_aisle // (cross_aisle_frequency + 1))
        for a in range(n_aisles - 1):
            for b in range(step, bays_per_aisle, step):
                G.add_edge((a, b), (a + 1, b), weight=1)

    # 4) Build the position map (for plotting)
    #    We flip the y‐axis so aisle 0 is at the top in most plotting libraries
    pos = {
        (a, b): (b, n_aisles - 1 - a)
        for (a, b) in G.nodes()
    }

    return G, pos


def build_layout(n_aisles=5, bays_per_aisle=20, cross_aisle_frequency=1):
    """
    Backward‐compatible helper for code that only wants the Graph.
    """
    G, _ = build_layout_graph(n_aisles, bays_per_aisle, cross_aisle_frequency)
    return G
