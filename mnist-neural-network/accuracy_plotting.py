import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file without headers
df = pd.read_csv("results/accuracy_errors.csv", header=None)
print(df.head())

# Set the first column as headers and remove it from the data
columns = df.iloc[:, 0]  # First column becomes headers
df = df.iloc[:, 1:].T       # Remove first column, transpose for plotting
df.columns = columns

# Convert values to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Plot the data
plt.plot(df.index, df["Binary32"], marker="o", linestyle="-", markersize=1, label="Binary32")
plt.plot(df.index, df["Binary64"], marker="o", linestyle="-", markersize=1, label="Binary64")
plt.plot(df.index, df["Binary32 with SR"], marker="o", linestyle="-", markersize=1, label="Binary32 with SR")
plt.plot(df.index, df["Binary32 with Compensated Sum"], marker="o", linestyle="-", markersize=1, label="Binary32 with Compensated Sum")

# plt.yscale("log")
plt.xscale("log")
# Labels and title
plt.xlabel("log of Training Batches")
plt.ylabel("Accuracy")
plt.title("Accuracy per Training Batches for different FP accuracies")
plt.legend()

# Show the plot
plt.savefig("figures/mnist_accuracies.png")
plt.show()
