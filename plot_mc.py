import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your Monte‐Carlo results
df = pd.read_csv("mc_results.csv")

# 2. Make a violin plot
plt.violinplot(df["avg_distance"])
plt.ylabel("Avg distance per order (m)")
plt.title("Monte Carlo: Avg Distance Distribution")
plt.show()
