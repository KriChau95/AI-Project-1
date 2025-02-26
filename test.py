from collections import defaultdict
from ship import *
import matplotlib.pyplot as plt
import os

def save_success_dict(success_dict, filename):
    """Save the success dictionary to a file."""
    with open(filename, "w") as f:
        for q, count in success_dict.items():
            f.write(f"{q} {count}\n")

def load_success_dict(filename):
    """Load the success dictionary from a file, or return an empty one if not found."""
    success_dict = defaultdict(int)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                q, count = line.split()
                success_dict[float(q)] = int(count)
    return success_dict

num_tests = 40
bots = [bot1, bot2, bot3]
bot_names = ["Bot 1", "Bot 2", "Bot 3"]
colors = ["blue", "red", "green"]
markers = ["o", "s", "^"]

# Dictionary to store success results for each bot
bot_success_dicts = {}

for i, bot in enumerate(bots):
    filename = f"success_dict_{bot_names[i].lower().replace(' ', '_')}.txt"
    success_dict = load_success_dict(filename)
    
    for q_int in range(0, 100, 5):
        q = q_int / 100
        if q in success_dict:
            continue  # Skip if already computed
        
        print(f"Running {bot_names[i]} for q = {q}")
        
        for j in range(num_tests):
            random.seed(j)
            ship_info = init_ship(40)
            res, fire, fire_path, t = bot(ship_info.copy(), q)

            if res == 'success':
                success_dict[q] += 1

    # Save results for the current bot
    save_success_dict(success_dict, filename)
    bot_success_dicts[bot_names[i]] = success_dict

# Plot results
plt.figure(figsize=(8, 5))

for i, bot_name in enumerate(bot_names):
    success_dict = bot_success_dicts[bot_name]
    q_values = sorted(success_dict.keys())
    success_probs = [success_dict[q] / num_tests for q in q_values]
    
    plt.plot(q_values, success_probs, marker=markers[i], linestyle='-', color=colors[i], label=bot_name)

plt.xlabel('q')
plt.ylabel('Success Probability')
plt.title('Success Probability vs q for Different Bots')
plt.legend()
plt.grid(True)
plt.show()
