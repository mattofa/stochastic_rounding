import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal

df = pd.read_csv("results/harmonic_errors.csv", header=None).T
print(df.head())

# split the data frame into columns
df.columns = ["b32", "b32_sr", "b32_conc"]
df['x_values'] = (df['b32'].index)


# Convert to float for plotting
x1, y1 = df['x_values'].astype(float), df["b32"].astype(float)
x2, y2 = df['x_values'].astype(float), df["b32_sr"].astype(float)
x3, y3 = df['x_values'].astype(float), df["b32_conc"].astype(float)

plt.plot(x1, y1, marker="o", linestyle='-', markersize=1, label="Binary32")
plt.plot(x2, y2, marker="o", linestyle='-', markersize=1, label="Binary32 with SR")
plt.plot(x3, y3, marker="o", linestyle='-', markersize=1, label="Binary32 with Compensated Sum")

# set axes to logscale
plt.xscale("log")
plt.yscale("log")


plt.xlabel("Log of iteration x10 -6")
plt.ylabel("Log of Absolute Error")
plt.title("Absolute Errors of Harmonic Series with regards to Iteration")
plt.legend()
plt.savefig("figures/harmonic_errors.png")
plt.show()