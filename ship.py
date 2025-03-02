import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque
import copy
import datetime

global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal

random.seed(4)

# Initializes the maze with bot, button, fire, and open and closed cells based on an input dimension d - number of rows and columns
def init_ship(dimension):
        
    d = dimension
    
    # 0 = open
    # 1 = closed 
    # 2 = bot
    # -2 = button
    # -1 = fire

    # initialize ship to size dimension
    ship = [[1] * d for _ in range(d)] 

    # open up a random cell on interior
    to_open = random.sample(range(1,d-1), 2) # range is set from 1 to d-1 to ensure we select from interior
    row, col = to_open
    ship[row][col] = 0

    single_neighbor = set() # stores all cells' blocked coordinates that have exactly 1 open neighbor
    closed = set() # stores cells that have no chance for being blocked coordinates with exactly 1 open neighbor

    # initialize single_neighbor set based on first open cell
    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < d and 0 <= c < d:
            single_neighbor.add((r,c))

    # Iteratively opening up cells to create maze structure
    while single_neighbor:

        chosen_coordinate = (random.choice(list(single_neighbor))) # choose cell randomly
        single_neighbor.remove(chosen_coordinate) # once cell is open, it can no longer be a blocked cell
        
        row, col = chosen_coordinate 
        ship[row][col] = 0 # open it up

        for dr,dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < d and 0 <= c < d and ship[r][c] == 1 and (r,c) not in closed:
                if (r,c) in single_neighbor:
                    single_neighbor.remove((r,c))
                    closed.add((r,c))
                else:
                    single_neighbor.add((r,c))
    
    # Identifying and handling deadend cells

    deadends = dict()

    # deadends = open cells with exactly 1 open neighbor
    # deadends dictionary:
    # key: (r,c) s.t. (r,c) is an open cell with exactly 1 open neighbor
    # value: list of (r,c) tuples that represent key's closed neighbors

    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0: # open cell
                open_n_count = 0
                closed_neighbors = []
                for dr,dc in directions:
                    nr,nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        if ship[nr][nc] == 0: # open neighbor
                            open_n_count += 1
                        elif ship[nr][nc] == 1:
                            closed_neighbors.append((nr,nc))
                if open_n_count == 1:
                    deadends[(r,c)] = closed_neighbors

    # for ~ 1/2 of deadend cells, pick 1 of their closed neighbors at random and open it
    for i in range(len(deadends)//2):
        list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys()))) # retrieve a random deadend cell's list of closed neighbors
        r,c = random.choice(list_closed_neighbors) # choose a random closed neighbor
        ship[r][c] = 0 # open it

    # create sets that  store coordinates of all the open cells and all the closed cells
    open_cells = set()
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_cells.add((r,c))

    # randomly place bot in one of the remaining open cells
    bot_r,bot_c = (random.choice(list(open_cells)))
    open_cells.remove((bot_r,bot_c))
    
    # randomly place fire in one of the remaining open cells
    fire_r,fire_c = (random.choice(list(open_cells)))
    open_cells.remove((fire_r,fire_c))

    # randomly place button in one of the remaining open cels
    button_r,button_c = (random.choice(list(open_cells)))
    open_cells.remove((button_r,button_c))

    ship[bot_r][bot_c] = 2
    ship[fire_r][fire_c] = -1
    ship[button_r][button_c] = -2

    fire_q = deque()

    for dr, dc in directions:
        nr, nc = fire_r + dr, fire_c + dc
        if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == 0:
            fire_q.append((nr,nc))

    info = dict()

    info['ship'] = ship
    info['bot'] = (bot_r, bot_c)
    info['fire'] = (fire_r, fire_c)
    info['button'] = (button_r, button_c)
    info['fire_q'] = fire_q

    return info

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
                if 0<=tr<d and 0<=tc<d and (fire_prog[level][tr][tc] == 0 or fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                    queue.append((tr,tc))
                    visited.add((tr,tc))

        level += 1
    
    return False
  

def probabilistic_search(info,fire_prog):
    
    directions = [(0,1), (1,0), (-1,0), (0,-1)]
    d = len(info['ship'])
    button = info['button']
    bot = info['bot']

    def create_path(threshold):

        visited = set()
        queue = deque()
        prev = dict()
        prev[bot] = None
        level = 0

        queue.append(info['bot'])
        
        while level<len(fire_prog):
            # print("inside create path", threshold, datetime.datetime.now().time())
            # print("level ", level, "queue", queue)
            x = len(queue)
            for i in range(x):
                r,c = queue.popleft()
                if (r,c) == button:
                    #print(" is this the dagger?found a path")
                    curr_p = (r,c)
                    path = deque()
                    while curr_p != bot:
                        #print(curr_p,bot)
                        path.appendleft(curr_p)
                        curr_p = prev[curr_p]
                    #print(list(path))
                    return list(path).copy()


                for dr,dc in directions:
                    tr,tc = r+dr,c+dc
                    if 0<=tr<d and 0<=tc<d and ((threshold < fire_prog[min(len(fire_prog)-1,level+15)][tr][tc] <= 0) or fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                        queue.append((tr,tc))
                        visited.add((tr,tc))
                        prev[(tr,tc)] = (r,c)

            level += 1

    threshold = -0.5
    result = []
    while not result and threshold > -1:
        result = create_path(threshold)
        # print(result,threshold)
        threshold -= .1

    return result if result else []

def create_fire_prog(info, q, cap = float('inf')):

    fire = info['ship']
    fire_q = info['fire_q']
    d = len(fire)

    inQueue = set(fire_q)

    temp = np.array(fire)  # Shape (40, 40)

    temp = temp.reshape(1, 40, 40)  # Shape (1, 40, 40)

    fire_prog = temp

    counter = 0

    while fire_q and counter<cap: 

        counter += 1
        
        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # ##print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            ##print("result ", result, "iteration ", t)
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 1 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))

        temp = np.array(fire) 
        temp = temp.reshape(1, 40, 40) 
        fire_prog = np.concatenate((fire_prog, temp), axis=0)

    return fire_prog
 
def astar(start, map, button):
    
    def heuristic(cell1):
        return abs(cell1[0] - button[0]) + abs(cell1[1]-button[1])
    
    
    d = len(map)

    fringe = []

    #((0,0), 10) is pushed to the heap

    heapq.heappush(fringe, (heuristic(start),start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    while fringe:

        curr = heapq.heappop(fringe)

        if curr[1] == button:
            
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        r,c = curr[1]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            child = (nr,nc)
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1

                est_total_cost = cost + heuristic(child)

                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))

    return []               

# find path from bot to goal using A* algorithm
def bot1(info, fire_prog, visualize = False):

    bot_start = info['bot']
    button = info['button']

    path = astar(bot_start, fire_prog[0], button)
    t = 0
    
    res = "success"

    while t<len(path):
        if visualize: visualize_ship(fire_prog[t], list(path)[:t+1])

        tr, tc = path[t]
        if fire_prog[t][tr][tc] == -1:
            res = "failure"
            break
        t+=1

    return res

def bot2(info, fire_prog, visualize = False):

    bot_start = info['bot']
    button = info['button']

    path = astar(bot_start, fire_prog[0], button)
    if visualize: visualize_ship(fire_prog[0], path)

    i = 1
    while True:
        curr_pos = path[1]
        if curr_pos == button:
            return "success"
        path = astar(curr_pos, fire_prog[i], button)
        if not path:
            return "failure"
        if visualize: visualize_ship(fire_prog[i], path)
        i += 1
 
def bot3(info, fire_prog, visualize = False):

    bot_start = info['bot']
    button = info['button']
    ship = info['ship']

    curr_pos = bot_start

    i = 0
    

    while True:

        inflation_map = [row.copy() for row in fire_prog[i]]

        d = len(ship)

        for row in range(d):
            for col in range(d):
                if fire_prog[i][row][col] == -1:
                    for dr,dc in directions:
                        r = row + dr
                        c = col + dc
                        if 0 <= r < d and 0 <= c < d and (fire_prog[i][r][c] != 1 and fire_prog[i][r][c] != -1):
                            inflation_map[r][c] = -1
        
        if visualize: visualize_ship(inflation_map, None, bot3=f"Inflation Map {i}")

        path = astar(curr_pos, inflation_map, button)

        if path:
            if visualize: 
                visualize_ship(inflation_map, path, bot3 = f"Inflation Map w/ Path {i}")
                visualize_ship(fire_prog[i], path, bot3 = f"Actual Fire Prog {i}")
        else:
            path = astar(curr_pos, fire_prog[i], button)
            if visualize: visualize_ship(fire_prog[i], path, bot3 = f"Actual Fire Prog{i}")

        
        if not path:
            return "failure"

        curr_pos = path[1]
        if curr_pos == button:
            return "success"
        
        i += 1


def visualize_probabilistic_fire(prob_fire_prog, threshold, title="Probabilistic Fire Spread"):
    """
    Visualize the fire progression over time based on probability values.
    - Walls (1) → Black
    - Fire (-1) → Red
    - Empty cells (0) → White
    - Probabilities (0 < x < 1) → Gradient shades of red
    """
    timesteps, d, _ = prob_fire_prog.shape

    for t in range(timesteps):
        fig, ax = plt.subplots()
        img = np.zeros((d, d, 3))  # RGB image

        for i in range(d):
            for j in range(d):
                val = prob_fire_prog[t, i, j]
                if val == 1:      # Wall
                    img[i, j] = mcolors.to_rgb('black')
                elif val == -1:   # Fire
                    img[i, j] = mcolors.to_rgb('red')
                elif val == 0:    # Empty
                    img[i, j] = mcolors.to_rgb('white')
                else:  # Fire probability (-1 < val < 0)
                    
                    intensity = abs(val)  # 1 at -1, 0 at 0
                    red = mcolors.to_rgb('red')    # (1, 0, 0)
                    white = mcolors.to_rgb('white') # (1, 1, 1)
                    # Linear interpolation: color = (1 - t) * start + t * end
                    if val < threshold:
                        img[i, j] = tuple((intensity) * red[k] + (1-intensity) * white[k] for k in range(3))
                    else:
                        img[i, j] = mcolors.to_rgb('orange')



        ax.imshow(img, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} - Step {t}")
        plt.show()


def bot4(info, fire_prog, q, visualize=False):

    samples = []
    num_samples = 50
    min_fp_len = float('inf')  # Use inf to track minimum length

    approx_len = len(astar(info['bot'], info['ship'], info['button'])) + 10

    for i in range(num_samples):
        #print("inside generate fire prog for num samples", datetime.datetime.now().time())
        sample_fire_prog = create_fire_prog(copy.deepcopy(info), q, approx_len)
        min_fp_len = min(min_fp_len, len(sample_fire_prog))
        samples.append(sample_fire_prog)  # Keep as a list for now

    # Truncate all fire progressions to min_fp_len and convert to NumPy array
    samples = np.array([fp[:min_fp_len] for fp in samples])

    # Replace negative values with 0
    samples = np.where(samples < -1, 0, samples)
    samples = np.where(samples > 1, 0, samples)

    prob_fire_prog = np.mean(samples, axis=0)  # Shape: (timesteps, rows, cols)

    #visualize_probabilistic_fire(prob_fire_prog,-0.5)
    path = probabilistic_search(info, prob_fire_prog)
    # print(path)
    # return prob_fire_prog
    t = 0
    
    res = "success"
    # print(len(path))

    while t<len(path):    
        
        if visualize: visualize_ship(fire_prog[t], list(path)[:t+1])

        tr, tc = path[t]
        if fire_prog[t][tr][tc] == -1:
            res = "failure"
            break
        t+=1

    return res, path



    



def visualize_ship(ship, path, bot3 = ""):

    color_map = {
        1: 'black',  # Wall
        0: 'white',  # Empty space
        -1: 'red',   # Fire
        2: 'blue',   # Bot
        -2: 'green'  # Button
    }
    
    d = len(ship)
    img = np.zeros((d, d, 3))
    
    for i in range(d):
        for j in range(d):
            img[i, j] = mcolors.to_rgb(color_map[ship[i][j]])  
    

    if path is not None:
        for i in range(len(path)):
            r, c = path[i]
            img [r, c] = mcolors.to_rgb('orange')

    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if bot3 != "":
        plt.title(bot3)
    plt.show()


def main():

    ship_info = init_ship(40)
    q = 0.65
    fire_prog = create_fire_prog(copy.deepcopy(ship_info),q)

    #print(winnable(ship_info, fire_prog))

    res = bot4(ship_info, fire_prog, q, visualize=True)
    print(res)


    # res, fire, fire_path, t =
    # fire_prog = create_fire_prog(ship_info,q)
    # for i in range(len(fire_prog)):
    #     visualize_ship(fire_prog[i], None)
    # visualize_ship(ship_info['ship'],fire_path)
    # visualize_ship(ship_info["ship"],None)
    # calculate_prob_fire_map(ship_info,q)
    #spreadFire(ship_info,q)

  

if __name__ == "__main__":
    main()

