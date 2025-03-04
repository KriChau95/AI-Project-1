# test.py is a file used for running tests on all the bots

# Importing libraries for randomness, data structures, and data visualization
import os
from ship import *
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import copy

# run rm *.txt to remove all txt files cd if existing results exist

# create the txt files to store results for each bot
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"
bot_3_results_file = "bot_3_results.txt"
bot_4_results_file = "bot_4_results.txt"

# create txt file to store results for winnability of maps
winnable_frequency_file = "winnable_f.txt"

# set the random seed
random.seed(14) 

# specify number ships to create
num_ships = 50

# initialize array to store num_ships ships
ships = []

# create num_ships ships by calling int_ship method and add them all to ships array
for i in range(num_ships):
    info = init_ship(40)
    ships.append(info)

# create dictionaries to store results
# key = q value
# value = number of ships that are successful
bot_1_results = defaultdict(int)
bot_2_results = defaultdict(int)
bot_3_results = defaultdict(int)
bot_4_results = defaultdict(int)

# all bots will always be successful when q = 0 - fire never spreads
bot_1_results[0] = num_ships
bot_2_results[0] = num_ships
bot_3_results[0] = num_ships
bot_4_results[0] = num_ships

# create a dictionary to store winnability - keep track of how many of the ships are winnable based on the fire progression
winnability = dict()

# ships with q = 0 will always be winnable
winnability[0] = num_ships
random.seed(20)
fire_prog = create_fire_prog(copy.deepcopy(ships[34]), 0.45)
# visualize_ship(ships[34]['ship'], None)
#bot2(ships[34], fire_prog, visualize=True)
# bot4(ships[34], fire_prog, q = 0.45, visualize=True)



# for q = 0.05, 0.10, .. 1.00
for j in range(5, 101, 5):

    q = j / 100

    print("testing q =", q)

    # keep track of number of winnable simulations for each q value
    num_winnable = 0

    # for each ship
    for i in range(len(ships)):

        visualize = False # flag to determine if we want to visualize certain ships

        # create a 3D array fire progression based on our current ship and q value and store it
        fire_prog = create_fire_prog(copy.deepcopy(ships[i]), q)

        # determine if ship was winnable based on that fire_prog and increment fire_prog accordingly
        if winnable(ships[i], fire_prog):
            num_winnable += 1
        else:
            del fire_prog # saves storage
            continue # if ship is unwinnable, skip to next iteration without incrementing bots' success hashmaps
        
        # test bot 1 for that specific ship and fire progression
        res_1 = bot1(ships[i], fire_prog, False)
        
        # increment bot 1 success count if it succeeded
        if res_1 == 'success':
            bot_1_results[q] += 1
            print("bot 1 subtest n =", i, "success")
        else:
            print("bot 1 subtest n =", i, "failure")
        
        # test bot 2 for that specific ship and fire progression
        res_2 = bot2(ships[i], fire_prog, visualize)

        # increment bot 2 success count if it succeeded
        if res_2 == 'success':
            bot_2_results[q] += 1
            print("bot 2 subtest n =", i, "success")
        else:
            print("bot 2 subtest n =", i, "failure")

        # test bot 3 for that specific ship and fire progression
        res_3 = bot3(ships[i], fire_prog, visualize)

        # increment bot 3 success count if it succeeded
        if res_3 == 'success':
            bot_3_results[q] += 1
            print("bot 3 subtest n =", i, "success")
        else:
            print("bot 3 subtest n =", i, "failure")


        # test bot 4 for that specific ship and fire progression
        # also pass in q so it can run simulations as part of its methodology
        res_4, path, prob_fire = bot4(ships[i], fire_prog, q, visualize)

        # if res_2 == 'failure' and res_4 == 'success':
        #     visualize_ship(ships[i]['ship'], path)

        # increment bot 4 success count if it succeeded
        if res_4 == 'success':
            bot_4_results[q] += 1
            print("bot 4 subtest n =", i, "success")
        else:
            print("bot 4 subtest n =", i, "failure")  
            #visualize_ship(ships[i]['ship'],path) 
            #visualize_probabilistic_fire(prob_fire,0)
                    
        del fire_prog # to save storage
    
    # update winnability of q to store number of winnable simulations for that specific q
    winnability[q] = num_winnable

# Save bot 1 results to a text file
with open(bot_1_results_file, "w") as f:
    for q, success_count in bot_1_results.items():
        f.write(f"{q}: {success_count}\n")

print("Results saved to bot_1_results.txt.")

# Save bot 2 results to a text file
with open(bot_2_results_file, "w") as f:
    for q, success_count in bot_2_results.items():
        f.write(f"{q}: {success_count}\n")

print("Results saved to bot_2_results.txt.")

# Save bot 3 results to a text file
with open(bot_3_results_file, "w") as f:
    for q, success_count in bot_3_results.items():
        f.write(f"{q}: {success_count}\n")

print("Results saved to bot_3_results.txt.")

# Save bot 4 results to a text file
with open(bot_4_results_file, "w") as f:
    for q, success_count in bot_4_results.items():
        f.write(f"{q}: {success_count}\n")

print("Results saved to bot_4_results.txt.")

# Save winnable frequency results to a text file
with open(winnable_frequency_file, "w") as f:
    for q, winnable_count in winnability.items():
        f.write(f"{q}: {winnable_count}\n")

print("Results saved to winnable_f.txt.")