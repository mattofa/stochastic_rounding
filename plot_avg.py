import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal


# pi = 3.141592653589793115997963468544185161590576171875000000000000 

# Read first file
df1 = pd.read_csv("results/sr_error.csv", header=None).T  # Transpose to get single column
df1.columns = ["y_values"]
df1["y_values"] = df1["y_values"].apply(Decimal)
df1["x_values"] = df1.index * 1000  # Scale X by 1000

# Read second file
df2 = pd.read_csv("results/alt_sr_error.csv", header=None).T  # Transpose
df2.columns = ["y_values"]
df2["y_values"] = df2["y_values"].apply(Decimal)
df2["x_values"] = df2.index * 1000  # Scale X by 1000

# Convert to float for plotting
x1, y1 = df1["x_values"].astype(float), df1["y_values"].astype(float)
x2, y2 = df2["x_values"].astype(float), df2["y_values"].astype(float)

# Plot both datasets
plt.plot(x1, y1, marker="o", linestyle="-", markersize=1, label="SR rounding")
plt.plot(x2, y2, marker="s", linestyle="-", markersize=1, label="Alternate SR rounding")
# plt.axhline(y=pi, color='r', linestyle='--', label="Stored as Double")

# Labels and title
plt.xlabel("Iterations of Stochastic Rounding")
plt.ylabel("Values of pi")
plt.title("Comparison of Two Datasets")
plt.legend()  # Show legend
plt.savefig("figures/pi_avgs.png")
plt.show()
