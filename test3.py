import os
import random
from ship import *
import matplotlib.pyplot as plt
from collections import defaultdict

# Check if the results files exist
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"

random.seed(42)

def precompute_fire_progs(ships, num_ships):
    fire_progs = defaultdict(dict)  # {q: {ship_index: fire_prog}}
    for j in range(5, 101, 5):
        q = j / 100
        print("Precomputing fire_progs for q =", q)
        for i in range(num_ships):
            fire_progs[q][i] = create_fire_prog(ships[i], q)
    return fire_progs

def simulate_bots(ships, fire_progs, num_ships):
    bot_1_results = defaultdict(int)
    bot_2_results = defaultdict(int)

    # Run simulation for each q value
    for q in range(5, 101, 5):
        q_value = q / 100
        print("Testing q =", q_value)

        # Evaluate all ships with current q
        for i in range(num_ships):
            fire_prog = fire_progs[q_value][i]

            res_1 = bot1(ships[i], fire_prog)
            if res_1 == 'success':
                bot_1_results[q_value] += 1
                print(f"bot 1 subtest n={i} success")
            else:
                print(f"bot 1 subtest n={i} failure")
            
            res_2 = bot2_2(ships[i], fire_prog)
            if res_2 == 'success':
                bot_2_results[q_value] += 1
                print(f"bot 2 subtest n={i} success")
            else:
                print(f"bot 2 subtest n={i} failure")

    return bot_1_results, bot_2_results

def save_results(file_path, results):
    with open(file_path, "w") as f:
        for q, success_count in results.items():
            f.write(f"{q}: {success_count}\n")
    print(f"Results saved to {file_path}.")

# Check if the results already exist
if not os.path.exists(bot_1_results_file) or not os.path.exists(bot_2_results_file):
    num_ships = 30
    ships = [init_ship(40) for _ in range(num_ships)]

    # Precompute fire_progs for each ship and each q value
    fire_progs = precompute_fire_progs(ships, num_ships)

    # Simulate both bots
    bot_1_results, bot_2_results = simulate_bots(ships, fire_progs, num_ships)

    # Save results to files
    save_results(bot_1_results_file, bot_1_results)
    save_results(bot_2_results_file, bot_2_results)

else:
    print("Results already exist. Skipping simulation.")
