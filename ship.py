# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque
import copy

global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal

# Initializes the maze with bot, button, fire, and open and closed cells based on an input dimension d - number of rows and columns
def init_ship(dimension):
        
    d = dimension
    
    # 0 = open cell
    # 1 = closed cell
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

        # determine which cells are new candidates to be single neighbors and add cells that have already been dealt with to a closed set
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

    # modifying the cells in the 2D array to store the appropriate special objects - bot, fire, burron
    ship[bot_r][bot_c] = 2
    ship[fire_r][fire_c] = -1
    ship[button_r][button_c] = -2

    # fire_q is a double ended que that stores the open cells adjacent to the initial fire
    
    fire_q = deque()

    for dr, dc in directions:
        nr, nc = fire_r + dr, fire_c + dc
        if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == 0:
            fire_q.append((nr,nc))

    # Condense all the information created within this function into a hashmap and return the hashmap

    info = dict()

    info['ship'] = ship
    info['bot'] = (bot_r, bot_c)
    info['fire'] = (fire_r, fire_c)
    info['button'] = (button_r, button_c)
    info['fire_q'] = fire_q

    return info


# Given:
# 
# an info hashmap: stores information about the ship
# q value: a parameter corresponding to fire spreadability
# cap: max number of time units in the future to look forward
#   when this is left at default value of infinity, will simulate fire progression till entire ship on fire
# 
# Return a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
# 
#   fire_prog[0] = initial 2D array representing ship
#   fire_prog[1] = 2D array representing fire spread on ship after 1 time unit
#   ...
#   fire_prog[n] = 2D array representing fire spread on ship after n time units
def create_fire_prog(info, q, cap = float('inf')):

    d = len(info['ship']) # length of ship stored concisely
    
    ship = info['ship'] # initial 2D representation of ship
    fire_q = info['fire_q'] # fire's open neighbors at time t = 0

    # convert ship to numpy array for convenience and easier terminal visualizations
    temp = np.array(ship) 
    temp = temp.reshape(1, 40, 40) 
    fire_prog = temp

    # initialize variables for BFS-like simulation of fire progression
    time_counter = 0 # how many time units we have progressed
    inQueue = set(fire_q) # used to prevent duplicate insertions into fire_q to avoid false probability inflations

    # expand fire either until ship is filled, or we have advanced and time passed = cap
    while fire_q and time_counter < cap: 

        time_counter += 1 # increment time by 1
        
        # BFS-like simulation of fire spread
        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            # determine how many neighbors of this open cell are on fire
            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == -1:
                    k += 1
            
            # compute probability of fire based on given q and determined k
            prob_f = (1 - ((1-q) ** k))
            
            # determine fire spreading or not based on randomly weighted simulation
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]

            # spread the fire
            if result=="fire":
                ship[row][col] = -1 # mark this cell as on fire
                # append this cell's adjacent cells to fire_q (stores all open cells adjacent to on fire ones) and add to inQueue set
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (ship[nr][nc] != 1 and ship[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            # don't spread fire, but acknowledge visited by adding to fire_q and inQueue
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))

        # convert the current 2D array ship to a numpy array and add it to the 3D numpy array fire_prog
        temp = np.array(ship) 
        temp = temp.reshape(1, 40, 40) 
        fire_prog = np.concatenate((fire_prog, temp), axis=0)

    return fire_prog

# function that takes in 2 parameters:
# info - hashmap/dictionary that contains all important ship information - 2D array representing ship, tuples representing bot, button, and initial fire position
# fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
# returns True if it is possible for the bot to reach the button given this specific fire progression, False otherwise
def winnable(info,fire_prog):
    
    d = len(info['ship'])
    
    # initializing important variables for BFS
    visited = set()
    queue = deque()
    level = 0
    queue.append(info['bot'])

    # running BFS to determine if there was any possible path the bot could have taken to succeed based on this specific fire progression
    while level < len(fire_prog):
        x = len(queue)
        for i in range(x):
            r,c = queue.popleft()
            if fire_prog[level][r][c] == -2: # if both reaches button, return True
                return True

            # visit all adjacent cells and check if it is possible to reach them at that time step without catching on fire
            for dr,dc in directions:
                tr,tc = r+dr,c+dc
                if 0<=tr<d and 0<=tc<d and (fire_prog[level][tr][tc] == 0 or fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                    queue.append((tr,tc))
                    visited.add((tr,tc))
        # increment level to represent one time unit has advanced
        level += 1
    
    return False
  
# A* search algorithm implementation that takes in:
# start - tuple of (start row, start col) of where to start search
# map - contains information of map in current state - 2D array
# button - tuple of (button row, button col) of final destination
def astar(start, map, button):
    
    # heuristic used for A* - Manhattan distance between 2 points (x_1, y_1) and (x_2, y_2)
    # returns sum of absolute value in difference of x and absolute value in difference of y
    # takes in tuple cell1 (row, col) and returns Manhattan distance to goal - button
    def heuristic(cell1):
        return abs(cell1[0] - button[0]) + abs(cell1[1]-button[1])
    
    # initializing useful variables for A*
    d = len(map)
    fringe = []

    # more initialization of variables
    heapq.heappush(fringe, (heuristic(start),start))
    # items on the fringe (heap) will look like (23, (2,5))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    # A* loop
    while fringe:
        
        # pop the cell with the lowest estimated cost
        curr = heapq.heappop(fringe)

        # curr = (heuristic((x,y)), (x,y))
        # curr pos = curr[1]
        # heuristic evaluated at curr pos = curr[0]

        # if we have reached the goal, reconstruct the path
        if curr[1] == button:
            
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        # get current cell's row and column
        r,c = curr[1]

        # explore neighboring cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc # calculating neighbor's coordinates
            child = (nr,nc) # child is tuple that represents neighbor's coordinates

            # check if neighbor is in bounds, not a wall, and not a fire
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1

                # compute estimated total cost as sum of actual cost and heurisitc
                est_total_cost = cost + heuristic(child)

                # if this path to child is better (or if it's not visited yet), update
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))

    # if no path was found, return an empty list
    return []               

# Bot 1: find path from bot to goal using A* algorithm
# Parameters:
#   info - hashmap containing data about ship - 2D array representation, coords of bot, button, initial fire
#   fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
#   visualize - boolean flag that is set to True if we want to visualize ship at different points in Bot1's process; it is False by default
# Returns "success" if Bot 1 succeeds on A* computed path based on the specific fire progression; "failure" otherwise
def bot1(info, fire_prog, visualize = False):

    # extract necessary information from info - bot start position and button position tuples
    bot_start = info['bot']
    button = info['button']

    # calculate and store A* path from bot start position to button position
    path = astar(bot_start, info['ship'], button)
    
    # initialize t = 0 to represent time state
    t = 0
    
    # res is a string variable that is used to store whether or not Bot1 succeeds on the path based on the fire progression
    res = "success"

    # move through the path computed by A*
    while t<len(path):

        if visualize: visualize_ship(fire_prog[t], list(path)[:t+1]) # visualize if requested

        tr, tc = path[t] # tr, tc represents row and column at curr time step
        # if cell at current time step is on fire in the simulated fire_prog at the corresponding time_step, fire has spread to our bot and we have failed
        
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
                    if 0<=tr<d and 0<=tc<d and ((threshold < fire_prog[min(len(fire_prog)-1,level+26)][tr][tc] <= 0) or fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                        queue.append((tr,tc))
                        visited.add((tr,tc))
                        prev[(tr,tc)] = (r,c)

            level += 1

    threshold = -0.3
    result = []
    #note: make threshold a function of q for goated results
    while not result and threshold > -1:
        result = create_path(threshold)
        # print(result,threshold)
        threshold -= .5

    return result if result else []

def bot4(info, fire_prog, q, visualize=False):

    samples = []
    num_samples = 50
    min_fp_len = float('inf')  # Use inf to track minimum length

    approx_len = len(astar(info['bot'], info['ship'], info['button'])) + 25

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

