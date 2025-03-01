import os
from ship import *
import matplotlib.pyplot as plt
from collections import defaultdict

# Check if the results files exist
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"

random.seed(42)

if not os.path.exists(bot_1_results_file) or not(os.path.exists(bot_2_results_file)):
    num_ships = 50
    ships = []

    for i in range(num_ships):
        info = init_ship(40)
        ships.append(info)

    bot_1_results = defaultdict(int)
    bot_2_results = defaultdict(int)

    bot_1_results[0] = num_ships
    bot_2_results[0] = num_ships

    for j in range(5, 101, 5):

        q = j / 100
        print("testing q =", q)

        for i in range(len(ships)):

            visualize = False
                
            fire_prog = create_fire_prog(copy.deepcopy(ships[i]), q)
            
            res_1 = bot1(ships[i], fire_prog, visualize)
            
            if res_1 == 'success':
                bot_1_results[q] += 1
                print("bot 1 subtest n =", i, "success")
            else:
                print("bot 1 subtest n =", i, "failure")
            
            res_2 = bot2_2(ships[i], fire_prog, visualize)



            if res_2 == 'success':
                bot_2_results[q] += 1
                print("bot 2 subtest n =", i, "success")
            else:
                print("bot 2 subtest n =", i, "failure")
            
            del fire_prog

    # Save results to a text file
    with open(bot_1_results_file, "w") as f:
        for q, success_count in bot_1_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_1_results.txt.")

    with open(bot_2_results_file, "w") as f:
        for q, success_count in bot_2_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_2_results.txt.")

else:
    print("Results already exist. Skipping simulation.")
