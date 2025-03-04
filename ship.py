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
random.seed(4)

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
# an info hashmap: stores information about the ship
# q value: a parameter corresponding to fire spreadability
# cap: max number of time units in the future to look forward
#   when this is left at default value of infinity, will simulate fire progression till entire ship on fire
# Return a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
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
    button_r,button_c = info['button']
    # initializing important variables for BFS
    visited = set()
    queue = deque()
    level = 0
    queue.append(info['bot'])

    # running BFS to determine if there was any possible path the bot could have taken to succeed based on this specific fire progression
    while level < len(fire_prog):
        if fire_prog[level][button_r][button_c] == -1:
            return False
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
def bot1(info, fire_prog, visualize):

    # extract necessary information from info - bot start position and button position tuples
    bot_start = info['bot']
    button = info['button']

    # calculate and store A* path from bot start position to button position
    path = astar(bot_start, info['ship'], button)
    if visualize: visualize_ship(fire_prog[0], list(path))
    
    # initialize t = 0 to represent time state
    t = 0
    
    # res is a string variable that is used to store whether or not Bot1 succeeds on the path based on the fire progression
    res = "success"

    # move through the path computed by A*
    while t<len(path):

        if visualize: visualize_ship(fire_prog[t], list(path)[:t+1]) # visualize path if requested

        tr, tc = path[t] # tr, tc represents row and column at curr time step

        # if cell at current time step is on fire in the simulated fire_prog at the corresponding time_step, fire has spread to our bot and we have failed
        if fire_prog[t][tr][tc] == -1: 
            res = "failure"
            break

        # if button on fire, return failure
        if fire_prog[t][button[0]][button[1]] == -1:
            return "failure"
        
        # increment time by 1
        t += 1 
    
    # Bot1 made it to end without catching on fire, so we return "success"
    return res  

# Bot 2: find path from bot to goal using A* algorithm, after each move, account for current state of fire and rerun A*
# Parameters:
#   info - hashmap containing data about ship - 2D array representation, coords of bot, button, initial fire
#   fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
#   visualize - boolean flag that is set to True if we want to visualize ship at different points in Bot1's process; it is False by default
# Returns "success" if Bot 2 succeeds with its approach on the specific fire progression; "failure" otherwise
def bot2(info, fire_prog, visualize):

    # extract necessary information from info - bot start position and button position tuples
    bot_start = info['bot']
    button = info['button']

    # calculate and store A* path from bot start position to button position
    path = astar(bot_start, fire_prog[0], button)

    if visualize: visualize_ship(fire_prog[0], path) # visualize path if requested

    # initialize t = 0 to represent time state
    t = 1 

    # loop until either success or failure
    while True:

        curr_pos = path[1] # path = list of tuples [(start_row, start_col), ... (button_row, button_col)], set curr_pos to path[1] - next step

        # return failure if button catches on fire
        if fire_prog[t][button[0]][button[1]] == -1:
            return "failure"
        
        # return failure if bot catches on fire
        if fire_prog[t][curr_pos[0]][curr_pos[1]] == -1:
            return "failure"

        # if we are at button, we succeeded, so return "success"
        if curr_pos == button: 
            return "success"
        
        # if we are not at button, we recompute new path using A* based on current fire spread
        path = astar(curr_pos, fire_prog[t], button) 

        # if no such path exists, we failed
        if not path: 
            return "failure"
        
        # visualize new path if requested
        if visualize: visualize_ship(fire_prog[t], path) 
        
        # increment time by 1
        t += 1 
 
# Bot 3: find path from bot to goal using A* algorithm and an inflation layer which assumes fire spreads to all possible open cells it can in next move
# Take a path that avoids that inflation layer, but if it is impossible to, compute a normal A* path. Repeat after each step
# Parameters:
#   info - hashmap containing data about ship - 2D array representation, coords of bot, button, initial fire
#   fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
#   visualize - boolean flag that is set to True if we want to visualize ship at different points in Bot1's process; it is False by default
# Returns "success" if Bot 3 succeeds with its approach on the specific fire progression; "failure" otherwise
def bot3(info, fire_prog, visualize):

    # extract necessary information from info - ship 2D array, bot start position, and button position tuples
    bot_start = info['bot']
    button = info['button']
    ship = info['ship']

    # store current bot position
    curr_pos = bot_start

    # current time is 0
    t = 0

    # repeat until either success or failure
    while True:

        # compute inflation map by making copy of current state of map
        inflation_map = [row.copy() for row in fire_prog[t]]

        d = len(ship)
        
        # loop through current state of map by accessing fire_prog at current t
        for row in range(d):
            for col in range(d):
                if fire_prog[t][row][col] == -1:
                    # for all adjecent cells to on fire cells, if they are in bounds, open, and not already on fire, set them to on fire in our inflation map
                    for dr,dc in directions: 
                        r = row + dr
                        c = col + dc
                        if 0 <= r < d and 0 <= c < d and (fire_prog[t][r][c] != 1 and fire_prog[t][r][c] != -1):
                            inflation_map[r][c] = -1
        
        if visualize: visualize_ship(inflation_map, None, bot3=f"Inflation Map {t}") # visualize inflation map if visualization is requested

        # compute path based on inflation map
        path = astar(curr_pos, inflation_map, button)

        # if such a path exists, proceed forward, else compute a path using astar without inflation map
        if path:
            if visualize: 
                visualize_ship(inflation_map, path, bot3 = f"Inflation Map w/ Path {t}")
                visualize_ship(fire_prog[t], path, bot3 = f"Actual Fire Prog {t}")
        else:
            path = astar(curr_pos, fire_prog[t], button)
            if visualize: visualize_ship(fire_prog[t], path, bot3 = f"Actual Fire Prog{t}")

        # if still no path exists, return failure
        if not path:
            return "failure"

        # path = list of tuples [(start_row, start_col), ... (button_row, button_col)], set curr_pos to path[1] - next step
        curr_pos = path[1]

        # return failure if bot catches on fire
        if fire_prog[t][curr_pos[0]][curr_pos[1]] == -1:
            return "failure"
        
        # return failure if button catches on fire
        if fire_prog[t][button[0]][button[1]] == -1:
            return "failure"

        # return success if we reach button
        if curr_pos == button:
            return "success"
        
        # increment t by 1 to show we have moved 1 time step forward
        t += 1


# Function to visualize probabilistic fire spread map for many valyes of t- used in bot 4
def visualize_probabilistic_fire(prob_fire_prog, threshold, title="Probabilistic Fire Spread"):

    # Visualize the fire progression over time based on probability values
    #   Walls (1) = Black
    #   Fire (-1) = Red
    #   Empty cells (0) = White
    #   Probabilities (0 < x < 1) → Gradient shades of red

    timesteps, d, _ = prob_fire_prog.shape # Extract the number of time steps and grid dimensions

    for t in range(timesteps): # iterate through each time stamp
        if t == 45:
            break
        fig, ax = plt.subplots() # set up subplots
        img = np.zeros((d, d, 3))  # RGB image

        # looop through each cell in the grid
        for i in range(d):
            for j in range(d):

                val = prob_fire_prog[t, i, j] # get probability of fire at that time at that cell
                
                # Wall
                if val == 1:      
                    img[i, j] = mcolors.to_rgb('black')
                
                # Fire
                elif val == -1:   
                    img[i, j] = mcolors.to_rgb('red')
                
                # Empty
                elif val == 0:  
                    img[i, j] = mcolors.to_rgb('white')
                
                # Potential Fire

                else:  # Fire probability (-1 < val < 0) (Fire = -1)
                    
                    intensity = abs(val)  # 1 at -1, 0 at 0
                    red = mcolors.to_rgb('red') # (1, 0, 0)
                    white = mcolors.to_rgb('white') # (1, 1, 1)
                    
                    # if cell < threshold - it is dangerous and will be avoided by Bot4
                    if val < threshold:
                        img[i, j] = tuple((intensity) * red[k] + (1-intensity) * white[k] for k in range(3)) # display as gradient of red
                    #if cell >= threshold - bot is willing to gamble on that cell - show as orange
                    else:
                        img[i, j] = mcolors.to_rgb('orange')

        # show all the plots at all the times
        ax.imshow(img, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} - Step {t}")
        plt.show()

# helper function for Bot4 that takes in:
# Parameters:
#   info - hashmap containing data about ship - 2D array representation, coords of bot, button, initial fire
#   prob_fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
#   the prob_fire_prog is different than normal fire_progs; 
#   created after running 50 simulations, with each cells is a probability the fire will spread there at time t. 
#
# Performs a probabalistic breadth first search based on prob_fire_prog and returns a path it thinks will reach the goal without catching on fire
def probabilistic_search(info, prob_fire_prog):
    
    d = len(info['ship'])

    # extract necessary information from info - bot start position and button position tuples
    button = info['button']
    bot = info['bot']

    # internal helper function that finds a path from start to button using BFS but it only considers safe cells
    # safe cells determined by whether or not they are greater than a cutoff threshold
    def create_path(threshold, offset):
        
        # initialize necessary variables for probabalistic BFS - queue, visited, set, prev hashmap for path reconstruction
        visited = set()
        queue = deque()
        prev = dict()
        prev[bot] = None
        level = 0

        # add initial bot position to queue
        queue.append(info['bot'])
        
        # loop through each probabailistic fire map in prob_fire_prog for each time
        while level<len(prob_fire_prog):

            x = len(queue)
            
            for i in range(x):
                r,c = queue.popleft() # pop everything off current layer in queue
                
                # if current pos is equal to button, we found a solution, so we reconstruct path and return it
                if (r,c) == button:
                    curr_p = (r,c)
                    path = deque()
                    while curr_p != bot:
                        path.appendleft(curr_p)
                        curr_p = prev[curr_p]
                    return list(path).copy()

                # if we have not arrived at button, visit all adjacent cells if they have not already been visited and they are safe enough based on our threshold
                # offset represents how many time steps in the future we consider in order to manage risk aversion
                for dr,dc in directions:
                    tr,tc = r+dr,c+dc
                    if 0<=tr<d and 0<=tc<d and ((threshold < prob_fire_prog[min(len(prob_fire_prog)-1,level + offset)][tr][tc] <= 0) or prob_fire_prog[level][tr][tc] == -2) and (tr,tc) not in visited:
                        queue.append((tr,tc)) # add this new cell to queue so we visit its neighbors in next step
                        visited.add((tr,tc)) # mark this cell as visited
                        prev[(tr,tc)] = (r,c) # store how we got here using prev dictionary / hashmap

            # increment level by 1
            level += 1

    # set hyperparameters for probabilistic BFS
    threshold = -0.4
    offset = 20

    result = []

    # repeatedly try to find a path, but increasing risk tolerance if we cannot find one
    while not result and threshold >= -1:

        result = create_path(threshold, offset)
        threshold -= .05


    # return result if we find a path

    return result if result else astar(bot,info['ship'],button)

# Bot 4: Based on initial map state and q value, runs many simulations of fire progression and comes up with an average estimated fire_progression
# based on that estimated fire progression, it tries to find a path using BFS such that each cell it takes is reasonably safe at the time it takes that cell
# if its threshold is too conservative, it tries again 
# Parameters:
#   info - hashmap containing data about ship - 2D array representation, coords of bot, button, initial fire
#   fire_prog - a 3D array that is a list of 2D arrays such that represents ships at each time in a specific fire progression
#   visualize - boolean flag that is set to True if we want to visualize ship at different points in Bot1's process; it is False by default
# Returns "success" if Bot 3 succeeds with its approach on the specific fire progression; "failure" otherwise
def bot4(info, fire_prog, q, visualize):

    button = info['button']
    
    # samples will store several 3D arrays representing fire progressions
    samples = []

    # number of fire progressions to simulate
    num_samples = 50

    # stores minimum length of a sample fire progression
    min_fp_len = float('inf')

    # determines a reasonable cap on how many time steps to simulate a fire progression proportional to the length of the initial A* path from start to end
    approx_len = len(astar(info['bot'], info['ship'], info['button'])) + 25

    # generate num_samples sample fire progressions - and add to samples
    for i in range(num_samples):
        sample_fire_prog = create_fire_prog(copy.deepcopy(info), q, approx_len)
        min_fp_len = min(min_fp_len, len(sample_fire_prog)) # keeping track of smallest length fire progression
        samples.append(sample_fire_prog) 

    

    # Truncate all fire progressions to min_fp_len and convert to NumPy array
    samples = np.array([fp[:min_fp_len] for fp in samples])

    # Replace negative values with 0
    samples = np.where(samples < -1, 0, samples)
    samples = np.where(samples > 1, 0, samples)

    # Define probabilistic fire progression as average of 50 fire progressions
    prob_fire_prog = np.mean(samples, axis=0)
    firer,firec = info['fire']
    # print(firer,firec)
    for i in prob_fire_prog:
        i[firer,firec] = 1
    
    # visualize_ship(info['ship'],None,"visualizing ship in bot4")
    # visualize_probabilistic_fire(prob_fire_prog,0.0)
    # get path from probabilistic search method
    path = probabilistic_search(info, prob_fire_prog)

    # testing if path successful or not

    # initialize time to t = 0
    t = 0
    
    # res variable to determine if successful or not
    res = "success"

    if not path:
        return "failure", [], prob_fire_prog
    
    # loop through each position in the path
    while t<len(path):    
        
        if visualize: visualize_ship(fire_prog[t], list(path)[:t+1]) # visualize path if requested

        tr, tc = path[t] # current position stored as tr, tc

        # if we are on fire at the current time, we set res to "failure", break out of loop and return it
        if fire_prog[t][tr][tc] == -1:
            res = "failure"
            break

        # if button on fire, return failure
        if fire_prog[t][button[0]][button[1]] == -1:
            return "failure", [], prob_fire_prog

        # increment t by 1 to show that 1 time unit has passed
        t+=1

    # return both result (success / failure) and path
    return res, path, prob_fire_prog

# Helpful testing function to visualize ship and path
# Parameters:
#   ship - 2D array represnting ship
#   path - list of tuples representing path from start pos to end pos - typically current bot pos to button
#   title - title for desired plot
def visualize_ship(ship, path, title = ""): 

    # hashmap that maps item in 2D ship array representation to corresponding color for visualization
    color_map = {
        1: 'black',  # Wall
        0: 'white',  # Empty space
        -1: 'red',   # Fire
        2: 'blue',   # Bot
        -2: 'green'  # Button
    }
    
    d = len(ship)

    # set up a numpy array to represent the img
    img = np.zeros((d, d, 3))
    
    # loop through the ship 2D array and set the corresponding color based on the value in the array and the color_map
    for i in range(d):
        for j in range(d):
            img[i, j] = mcolors.to_rgb(color_map[ship[i][j]])  
    
    # display the path by coloring in all cells from start of path to end of path orange
    if path is not None:
        for i in range(len(path)):
            r, c = path[i]
            img [r, c] = mcolors.to_rgb('orange')

    # display the graph
    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    # if a title is requested, set it
    if title != "":
        plt.title(title)
    
    # show the visualization
    plt.show()

# Main for testing
def main():

    random.seed(26)

    # Sample test with 40x40 ship, q = 0.65, bot4
    ship_info = init_ship(40)
    visualize_ship(ship_info['ship'],None)
    # ship_info = init_ship(40)
    # visualize_ship(ship_info['ship'],None)
    # visualize_ship(ship_info['ship'], None, "Bot 2 Failure Case")
    
    q = 0.50
    fire_prog = create_fire_prog(copy.deepcopy(ship_info),.45)
    # print(bot4(ship_info, fire_prog, q=.45, visualize=True))[0]
    # print(res)  

# Run Main
if __name__ == "__main__":
    main()