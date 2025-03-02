import os
from ship import *
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import copy

# Check if the results files exist
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"
bot_3_results_file = "bot_3_results.txt"
bot_4_results_file = "bot_4_results.txt"

winnable_frequency_file = "winnable_f.txt"
notes = "notes.txt"

random.seed(42) 

if not os.path.exists(bot_1_results_file):
    num_ships = 50
    ships = []

    for i in range(num_ships):
        info = init_ship(40)
        ships.append(info)

    bot_1_results = defaultdict(int)
    bot_2_results = defaultdict(int)
    bot_3_results = defaultdict(int)
    bot_4_results = defaultdict(int)

    bot_1_results[0] = num_ships
    bot_2_results[0] = num_ships
    bot_3_results[0] = num_ships
    bot_4_results[0] = num_ships

    winnability = dict()
    winnability[0] = num_ships

    room_4_growth = []

    for j in range(5, 101, 5):

        q = j / 100
        print("testing q =", q)

        num_winnable = 0

        for i in range(len(ships)):

            visualize = False #i == 42 and 0.4 <= q <= 0.7

            fire_prog = create_fire_prog(copy.deepcopy(ships[i]), q)

            if winnable(ships[i], fire_prog):
                num_winnable += 1
            else:
                del fire_prog
                continue
            ###in winnable
            ## data structure - [ship that winnable but failed for bot2 and bot3]
            res_1 = bot1(ships[i], fire_prog, False)
            
            if res_1 == 'success':
                bot_1_results[q] += 1
                print("bot 1 subtest n =", i, "success")
            else:
                print("bot 1 subtest n =", i, "failure")
            
            res_2 = bot2(ships[i], fire_prog, visualize)

            if res_2 == 'success':
                bot_2_results[q] += 1
                print("bot 2 subtest n =", i, "success")
            else:
                room_4_growth.append(("ship 2", q, i))
                print("bot 2 subtest n =", i, "failure")

            res_3 = bot3(ships[i], fire_prog, visualize)

            if res_3 == 'success':
                bot_3_results[q] += 1
                print("bot 3 subtest n =", i, "success")
            else:
                room_4_growth.append(("ship 3", q, i))
                print("bot 3 subtest n =", i, "failure")

            res_4, path = bot4(ships[i], fire_prog, q, visualize)

            
            
            if res_4 == 'success':
                bot_4_results[q] += 1
                print("bot 4 subtest n =", i, "success")
            else:
                room_4_growth.append(("ship 4", q, i))
                print("bot 4 subtest n =", i, "failure")
                # if q == .25:
                visualize_ship(ships[i]['ship'], path)
                
            del fire_prog
        
        winnability[q] = num_winnable

    with open(notes, "w") as f:
        for x in room_4_growth:
            f.write(f"{q}: {x}\n")

    # Save results to a text file
    with open(bot_1_results_file, "w") as f:
        for q, success_count in bot_1_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_1_results.txt.")

    with open(bot_2_results_file, "w") as f:
        for q, success_count in bot_2_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_2_results.txt.")

    with open(bot_3_results_file, "w") as f:
        for q, success_count in bot_3_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_3_results.txt.")

    with open(bot_4_results_file, "w") as f:
        for q, success_count in bot_4_results.items():
            f.write(f"{q}: {success_count}\n")

    print("Results saved to bot_4_results.txt.")

    with open(winnable_frequency_file, "w") as f:
        for q, winnable_count in winnability.items():
            f.write(f"{q}: {winnable_count}\n")

    print("Results saved to winnable_f.txt.")


else:
    print("Results already exist. Skipping simulation.")
