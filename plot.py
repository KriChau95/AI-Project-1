# plot.py is used to parse the data stored by test.py in the .txt files, interpret it, and plot it

# import plotting library and numpy
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

# Read the results for each bot and the winnable txt file
bot_1_results = read_results("bot_1_results.txt")
bot_2_results = read_results("bot_2_results.txt")
bot_3_results = read_results("bot_3_results.txt")
bot_4_results = read_results("bot_4_results.txt")
winnability_results = read_results("winnable_f.txt")

# Extract q values (x-axis) and success rates (y-axis)
q_values = sorted(bot_1_results.keys())  # Assuming all files have the same q values
num_ships = bot_1_results[0]  # Total ships used in simulation

# Compute success rates for each bot
bot_1_success_rates = [bot_1_results[q] / num_ships for q in q_values]
bot_2_success_rates = [bot_2_results[q] / num_ships for q in q_values]
bot_3_success_rates = [bot_3_results[q] / num_ships for q in q_values]
bot_4_success_rates = [bot_4_results[q] / num_ships for q in q_values]

# Compute winnability for each q value
winnability_rates = [winnability_results[q] / num_ships for q in q_values]

# Compute adjusted success rates based on only factoring in winnable maps
adj_bot_1_success_rates = [bot_1_results[q] / winnability_results[q] for q in q_values]
adj_bot_2_success_rates = [bot_2_results[q] / winnability_results[q] for q in q_values]
adj_bot_3_success_rates = [bot_3_results[q] / winnability_results[q] for q in q_values]
adj_bot_4_success_rates = [bot_4_results[q] / winnability_results[q] for q in q_values]

# Create side-by-side subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

# First subplot: Bot Performance
axes[0].plot(q_values, bot_1_success_rates, marker="o", linestyle="-", label="Bot 1")
axes[0].plot(q_values, bot_2_success_rates, marker="s", linestyle="--", label="Bot 2")
axes[0].plot(q_values, bot_3_success_rates, marker="^", linestyle=":", label="Bot 3")
axes[0].plot(q_values, bot_4_success_rates, marker="x", linestyle="-.", label="Bot 4")
axes[0].set_xlabel("q (Probability of Fire Spread)")
axes[0].set_ylabel("Success Rate")
axes[0].set_title("Bot Performance vs. Fire Spread Probability")
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(0.3, 1.1)

# Second subplot: Winnability
axes[1].plot(q_values, winnability_rates, marker="d", linestyle="-", color="black", label="Winnability")
axes[1].set_xlabel("q (Probability of Fire Spread)")
axes[1].set_title("Overall Winnability vs. Fire Spread Probability")
axes[1].legend()
axes[1].grid(True)

# Third subplot: Bot Performance Adjusted for Winnability
axes[2].plot(q_values, adj_bot_1_success_rates, marker="o", linestyle="-", label="Bot 1")
axes[2].plot(q_values, adj_bot_2_success_rates, marker="s", linestyle="--", label="Bot 2")
axes[2].plot(q_values, adj_bot_3_success_rates, marker="^", linestyle=":", label="Bot 3")
axes[2].plot(q_values, adj_bot_4_success_rates, marker="x", linestyle="-.", label="Bot 4")
axes[2].set_xlabel("q (Probability of Fire Spread)")
axes[2].set_ylabel("Success Rate")
axes[2].set_title("Bot Performance Adjusted for Winnability")
axes[2].legend()
axes[2].grid(True)
axes[2].set_ylim(0.3, 1.1)

# Set x-axis ticks at intervals of 0.05 for both subplots
for ax in axes:
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0.3, 1.05, 0.1))
    ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-tick labels are displayed

# Show the plot
plt.tight_layout()
plt.show()
