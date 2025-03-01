import matplotlib.pyplot as plt
import numpy as np

# Function to read results from a file
def read_results(filename):
    results = {}
    with open(filename, "r") as f:
        for line in f:
            q, success_count = line.strip().split(": ")
            results[float(q)] = int(success_count)
    return results

# Read the results
bot_1_results = read_results("bot_1_results.txt")
bot_2_results = read_results("bot_2_results.txt")

# Extract q values (x-axis) and success rates (y-axis)
q_values = sorted(bot_1_results.keys())  # Assuming both files have the same q values
num_ships = bot_1_results[0]  # Total ships used in simulation

# Compute success rates
bot_1_success_rates = [bot_1_results[q] / num_ships for q in q_values]
bot_2_success_rates = [bot_2_results[q] / num_ships for q in q_values]

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(q_values, bot_1_success_rates, marker="o", linestyle="-", label="Bot 1")
plt.plot(q_values, bot_2_success_rates, marker="s", linestyle="--", label="Bot 2")

plt.xlabel("q (Probability of Fire Spread)")
plt.ylabel("Success Rate")
plt.title("Bot Performance vs. Fire Spread Probability")
plt.ylim(-0.1, 1.1)

# Set x-axis ticks at intervals of 0.05
plt.xticks(np.arange(0, 1.05, 0.05))

plt.legend()
plt.grid(True)
plt.show()
