# generate_synthetic_orders.py

import csv
import random

def gen_orders_csv(
    filename="orders.csv",
    num_orders=500,
    max_hops=4,
    n_aisles=5,
    bays_per_aisle=20
):
    """
    Creates a synthetic orders.csv with `num_orders` orders.
    Each order starts at (0,0) and randomly hops up to max_hops times
    to random (aisle,bay) slots within your layout dimensions.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id","current_slot","sku_slot"])

        for oid in range(1, num_orders+1):
            # start at depot
            x, y = 0, 0
            for hop in range(max_hops):
                # pick a random next slot
                nx_ = random.randint(0, n_aisles-1)
                ny_ = random.randint(0, bays_per_aisle-1)
                writer.writerow([oid, f"({x},{y})", f"({nx_},{ny_})"])
                x, y = nx_, ny_

if __name__ == "__main__":
    # tweak these to match your layout or desired volume
    gen_orders_csv(
        filename="orders.csv",
        num_orders=500,
        max_hops=5,
        n_aisles=5,
        bays_per_aisle=20
    )
    print("Generated synthetic orders.csv with 500 orders.")
