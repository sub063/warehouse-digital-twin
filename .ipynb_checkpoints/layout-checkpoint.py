import networkx as nx

def build_layout(n_aisles=5, bays_per_aisle=20):
    G = nx.Graph()
    for a in range(n_aisles):
        for b in range(bays_per_aisle):
            node = (a, b)            # (aisle, bay)
            G.add_node(node)
            if b > 0:                # connect along aisle
                G.add_edge((a, b-1), node, weight=1)
    # cross-aisle every 5 bays
    for b in range(0, bays_per_aisle, 5):
        for a in range(n_aisles-1):
            G.add_edge((a, b), (a+1, b), weight=2)
    return G
G = build_layout()
pos = { (a, b): (b, -a) for (a, b) in G.nodes() }
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
nx.draw(
    G,
    pos=pos,
    node_size=30,
    with_labels=False,
    edge_color="gray"
)
plt.axis("off")
plt.show()
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    L = build_layout()
    nx.draw(L, with_labels=False, node_size=30)
    plt.show()


