import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a professional plot style
sns.set_theme(style="whitegrid")

# Load the summary data
summary_df = pd.read_csv("simulation_summary.csv")
batch_scenarios = summary_df["batch_size"].unique()

## --- 1. Comparison Plot (Bar Chart) ---
plt.figure(figsize=(12, 7))
sns.barplot(data=summary_df, x="batch_size", y="avg_cycle_time", palette="viridis")
plt.title("Average Order Cycle Time vs. Batch Size", fontsize=16)
plt.xlabel("Batch Size (Orders per Tour)", fontsize=12)
plt.ylabel("Average Cycle Time (seconds)", fontsize=12)
plt.savefig("comparison_plot.png")
plt.show()


## --- 2. Distribution Plot (Histogram/KDE) ---
fig, axes = plt.subplots(1, len(batch_scenarios), figsize=(18, 6), sharey=True)
fig.suptitle("Distribution of Order Cycle Times", fontsize=16)

for i, batch in enumerate(batch_scenarios):
    cycle_times_df = pd.read_csv(f"run_data_batch_{batch}_cycletimes.csv")
    sns.histplot(cycle_times_df["cycle_time"], kde=True, ax=axes[i])
    axes[i].set_title(f"Batch Size = {batch}")
    axes[i].set_xlabel("Cycle Time (s)")
plt.savefig("distribution_plot.png")
plt.show()


## --- 3. Time Series Plot ---
plt.figure(figsize=(12, 7))
for batch in batch_scenarios:
    timeline_df = pd.read_csv(f"run_data_batch_{batch}_timeline.csv")
    plt.plot(timeline_df["time"], timeline_df["orders"], label=f"Batch Size = {batch}")

plt.title("Throughput Over Time", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Cumulative Orders Completed", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("timeseries_plot.png")
plt.show()