import os
from ship import *
import matplotlib.pyplot as plt
from collections import defaultdict

# Check if the results file exists
results_file = "bot1_results.txt"

if not os.path.exists(results_file):
    num_ships = 30
    ships = []
    fire_progs = dict()

    random.seed(42)

    for i in range(num_ships):
        info = init_ship(40)
        ships.append(info)

    bot_1_results = defaultdict(int)

    for j in range(5, 101, 5):
        q = j / 100
        print("testing q =", q)

        for i in range(len(ships)):
            fire_prog = create_fire_prog(copy.deepcopy(ships[i]), q)
            res = bot1(copy.deepcopy(ships[i]), fire_prog)

            if res == 'success':
                bot_1_results[q] += 1
                print("subtest n =", i, "success")
            else:
                print("subtest n =", i, "failure")

    # Save results to a text file
    with open(results_file, "w") as f:
        for q, success_count in bot_1_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot1_results.txt.")
else:
    print("bot1_results.txt already exists. Skipping simulation.")
