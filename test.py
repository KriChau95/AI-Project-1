from collections import defaultdict
from ship import *
import matplotlib.pyplot as plt
import os
import copy

def save_success_dict(success_dict, filename):
    with open(filename, "w") as f:
        for q, count in success_dict.items():
            f.write(f"{q} {count}\n")

def load_success_dict(filename):
    success_dict = defaultdict(int)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                q, count = line.split()
                success_dict[float(q)] = int(count)
    return success_dict


def winnable(info,fire_prog):
    print("in winnable")
    directions = [(0,1), (1,0), (-1,0), (0,-1)]
    d = len(info['ship'])
    visited = set()
    queue = deque()
    level = 0

    queue.append(ship['bot'])

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
    

num_tests = 20
bots = [bot1, bot2, bot3]
bot_names = ["Bot 1", "Bot 2", "Bot 3"]
colors = ["blue", "red", "green"]
markers = ["o", "s", "^"]
ships = [init_ship(40) for _ in range(num_tests)]
bot_success_dicts = {name: load_success_dict(f"success_dict_{name.lower().replace(' ', '_')}.txt") for name in bot_names}

# for i,ship in enumerate(ships):
#     print("this is ship number ",i)
#     visualize_ship(ship["ship"],None)
    
    
for j in range(5,105,5):

    q=j/100
    print("running for q = ", q)
    for i, ship in enumerate(ships):

        fire_prog = create_fire_prog(ship, q)

        print("fireprog created")


        res = bot1(copy.deepcopy(ship), q, fire_prog)
        possible = winnable(ship,fire_prog)
        print("running ship number ",i, "is it possible", possible)
        print("running bot1 ",i)

        if res == "success":
            bot_success_dicts["Bot 1"][q] += 1
        else:
            print("fail bot1")
        
        # print("running bot2 ",i)

        # res, fire, fire_path, t = bot2(copy.deepcopy(ship), q)
        # if res == "success":
        #     bot_success_dicts["Bot 2"][q] += 1
            
        # else:
        #     print("fail bot2")

        # print("running bot3 ",i)

        # res, fire, fire_path, t = bot3(copy.deepcopy(ship), q)

        # if res == "success":
        #     bot_success_dicts["Bot 3"][q] += 1
        # else:
        #     print("fail bot3")

        

for i, bot_name in enumerate(bot_names):
    save_success_dict(bot_success_dicts[bot_name], f"success_dict_{bot_name.lower().replace(' ', '_')}.txt")

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


