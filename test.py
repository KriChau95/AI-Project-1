import os
from ship import *
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Check if the results files exist
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"
bot_3_results_file = "bot_3_results.txt"
winnable_frequency_file = "winnable_f.txt"

random.seed(42)

def winnable(info,fire_prog):
    directions = [(0,1), (1,0), (-1,0), (0,-1)]
    d = len(info['ship'])
    visited = set()
    queue = deque()
    level = 0

    queue.append(info['bot'])

    while level<len(fire_prog):
        # print("level ", level, "queue", queue)
        x = len(queue)
        for i in range(x):
            r,c = queue.popleft()
            if fire_prog[level][r][c] == -2:
                return True

            for dr,dc in directions:
                tr,tc = r+dr,c+dc
                if 0<=tr<d and 0<=tc<d and (fire_prog[level][tr][tc] == 1 or fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                    queue.append((tr,tc))
                    visited.add((tr,tc))

        level += 1
    
    return False
   

if not os.path.exists(bot_1_results_file):
    num_ships = 50
    ships = []

    for i in range(num_ships):
        info = init_ship(40)
        ships.append(info)

    bot_1_results = defaultdict(int)
    bot_2_results = defaultdict(int)
    bot_3_results = defaultdict(int)

    bot_1_results[0] = num_ships
    bot_2_results[0] = num_ships
    bot_3_results[0] = num_ships

    winnability = dict()
    winnability[0] = num_ships

    for j in range(5, 101, 5):

        q = j / 100
        print("testing q =", q)

        num_winnable = 0

        for i in range(len(ships)):

            visualize = False
                
            fire_prog = create_fire_prog(copy.deepcopy(ships[i]), q)

            if winnable(ships[i], fire_prog):
                num_winnable += 1
            else:
                del fire_prog
                continue

            res_1 = bot1(ships[i], fire_prog, visualize)
            
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
                print("bot 2 subtest n =", i, "failure")

            res_3 = bot3(ships[i], fire_prog, visualize)

            if res_3 == 'success':
                bot_3_results[q] += 1
                print("bot 3 subtest n =", i, "success")
            else:
                print("bot 3 subtest n =", i, "failure")
            
            del fire_prog
        
        winnability[q] = num_winnable

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

    with open(winnable_frequency_file, "w") as f:
        for q, winnable_count in winnability.items():
            f.write(f"{q}: {winnable_count}\n")

    print("Results saved to winnable_f.txt.")


else:
    print("Results already exist. Skipping simulation.")
