import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque

global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal

random.seed(4)

# Initializes the maze with bot, button, fire, and open and closed cells based on an input dimension d - number of rows and columns
def init_ship(dimension):
        
    d = dimension
    
    # 0 = closed
    # 1 = open 
    # 2 = bot
    # -2 = button
    # -1 = fire

    # initialize ship to size dimension
    ship = [[0] * d for _ in range(d)] 

    # open up a random cell on interior
    to_open = random.sample(range(1,d-1), 2) # range is set from 1 to d-1 to ensure we select from interior
    row, col = to_open
    ship[row][col] = 1

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
        ship[row][col] = 1 # open it up

        for dr,dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < d and 0 <= c < d and ship[r][c] == 0 and (r,c) not in closed:
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
            if ship[r][c] == 1: # open cell
                open_n_count = 0
                closed_neighbors = []
                for dr,dc in directions:
                    nr,nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        if ship[nr][nc] == 1: # open neighbor
                            open_n_count += 1
                        elif ship[nr][nc] == 0:
                            closed_neighbors.append((nr,nc))
                if open_n_count == 1:
                    deadends[(r,c)] = closed_neighbors

    # for ~ 1/2 of deadend cells, pick 1 of their closed neighbors at random and open it
    for i in range(len(deadends)//2):
        list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys()))) # retrieve a random deadend cell's list of closed neighbors
        r,c = random.choice(list_closed_neighbors) # choose a random closed neighbor
        ship[r][c] = 1 # open it

    # create sets that  store coordinates of all the open cells and all the closed cells
    open_cells = set()
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 1:
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
        if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == 1:
            fire_q.append((nr,nc))

    info = dict()

    info['ship'] = ship
    info['bot'] = (bot_r, bot_c)
    info['fire'] = (fire_r, fire_c)
    info['button'] = (button_r, button_c)
    info['fire_q'] = fire_q

    return info

def heuristic(cell1, button):
    return abs(cell1[0] - button[0]) + abs(cell1[1]-button[1])

def astar(tempr,tempc,map,button):

    d = len(map)

    print("tempr,tempc",tempr,tempc)
    heap = []
    heapq.heappush(heap, (0,tempr,tempc))
    prev = {}
    totalCost = {}
    prev[(tempr,tempc)] = None
    totalCost[(tempr, tempc)] = 0

    print("running a*", heap, prev,tempr,tempc)
    sol = None
    visited = set()
    while heap:
        # print("heap", heap)
        cost, r, c =  heapq.heappop(heap)
        if map[r][c] == -2:
            sol = (r,c)
            break
        visited.add((r,c))
        for dr,dc in directions:
            row = r + dr
            col = c + dc
            if 0<=row<d and 0<=col<d and (map[row][col] != 0 and map[row][col] != -1) and (row,col) not in visited:
                estCost = cost + heuristic((row,col),button)
                heapq.heappush(heap, (estCost,row,col))
                prev[(row,col)] = (r,c)
                visited.add((r,c))

    if sol != None:
        curr = sol
        path = deque()
        # #print(prev, curr)

        while curr != None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        print("path inside a* func", path)
        visualize_ship(map,path)
        return path
    else:
        return None

# find path from bot to goal using A* algorithm
def bot1(info, q):

    ship = info['ship']
    bot_r, bot_c = info['bot']
    button = info['button']
    fire_q = info['fire_q']
    
    d = len(ship)

    path = astar(bot_r, bot_c, ship, button)
    inQueue = set(fire_q)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]

    while fire_q and t<len(path):

        tr, tc = path[t]

        if fire[tr][tc] == -1:
            res = "failure"
            break

        visualize_ship(fire,list(path)[0:t+1])

        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # #print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            #print("result ", result, "iteration ", t)
            
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 0 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))

        #print("fireq, ", fire_q)
        t+=1

    print("path1",path)
    return res,fire, path, t

def bot2(info, q):
    
    #find path from bot to goal
    ship = info['ship']
    bot_r, bot_c = info['bot']
    button = info['button']
    fire_q = info['fire_q']
    
    d = len(ship)

    inQueue = set(fire_q)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]
    tr, tc = bot_r,bot_c
    final_path = []
    while fire_q:
        final_path.append((tr,tc))

        if (tr, tc) == button:
            break

        path = astar(tr,tc,fire, button)

        if path == None:
            res = "failure"
            break
    
        if fire[tr][tc] == -1:
            res = "failure"
            break
        
        # visualize_ship(fire,list(final_path)[0:t+1])

        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # #print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            #print("result ", result, "iteration ", t)
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 0 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))


        if path:
            path.popleft()
        if path:
            tr,tc = path.popleft()



        t+=1
        
        #print("path ", path)
        print("tr,tc",tr,tc)

    #print(res)
    return res, fire, final_path, t

def bot3(info, q):
    #find path from bot to goal
    ship = info['ship']
    bot_r, bot_c = info['bot']
    button = info['button']
    fire_q = info['fire_q']
    
    d = len(ship)

    inQueue = set(fire_q)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]
    tr, tc = bot_r,bot_c
    final_path = []
    while fire_q:

        final_path.append((tr,tc))

        if (tr, tc) == button:
            print("tr==tc, breaking")
            break
        

        inflation_map = [row.copy() for row in fire]

        for row in range(d):
            for col in range(d):
                if fire[row][col] == -1:
                    for dr,dc in directions:
                        r = row + dr
                        c = col + dc
                        if 0 <= r < d and 0 <= c < d and (fire[r][c] != 0 and fire[r][c] != -1):
                            inflation_map[r][c] = -1


        path = astar(tr,tc,inflation_map, button)

        if path == None:
            path = astar(tr,tc,fire, button)


        print("path", path)

        if path == None:
            res = "failure"
            break
    
        if fire[tr][tc] == -1:
            res = "failure"
            break

        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # #print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            #print("result ", result, "iteration ", t)
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 0 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))


        # visualize_ship(fire,list(final_path)[:t+1])

        if path:
            path.popleft()
        if path:
            tr,tc = path.popleft()



        t+=1
        
        #print("path ", path)
        print("tr,tc",tr,tc)

    #print(res)
    return res, fire, final_path, t


def bot4(info, q):
    ship = info['ship']
    bot_r, bot_c = info['bot']
    button = info['button']
    fire_q = info['fire_q']
    
    d = len(ship)

    inQueue = set(fire_q)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]
    tr, tc = bot_r,bot_c
    final_path = []
    print("before, ")
    visualize_ship(fire, None)

    #calculate probability map with depth = 5 (5 can be changed with testing)
    random.seed(23) 

    for iteration in range(5):
        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # #print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            #print("result ", result, "iteration ", t)
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 0 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))
    print("after, ")
    visualize_ship(fire, None)
    random.seed(4)




    while fire_q:
        break
        final_path.append((tr,tc))

        if (tr, tc) == button:
            print("tr==tc, breaking")
            break
        

        inflation_map = [row.copy() for row in fire]

        for row in range(d):
            for col in range(d):
                if fire[row][col] == -1:
                    for dr,dc in directions:
                        r = row + dr
                        c = col + dc
                        if 0 <= r < d and 0 <= c < d and fire[r][c] != 1:
                            inflation_map[r][c] = -1


        path = astar(tr,tc,inflation_map, button)

        if path == None:
            path = astar(tr,tc,fire, button)


        print("path", path)

        if path == None:
            res = "failure"
            break
    
        if fire[tr][tc] == -1:
            res = "failure"
            break

        for x in range(len(fire_q)):
            row, col = fire_q.popleft()

            if (row, col) in inQueue:  
                inQueue.remove((row, col))

            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and fire[nr][nc] == -1:
                    k += 1
            
            # #print("k", k)
            prob_f = (1 - ((1-q) ** k))
            
            result = random.choices(["fire", "no fire"], weights=[prob_f, 1-prob_f])[0]
            #print("result ", result, "iteration ", t)
            if result=="fire":
                fire[row][col] = -1
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < d and 0 <= nc < d and (fire[nr][nc] != 0 and fire[nr][nc] != -1) and (nr,nc) not in inQueue:
                        fire_q.append((nr,nc))
                        inQueue.add((nr,nc))
            else:
                fire_q.append((row,col))
                inQueue.add((row,col))


        # visualize_ship(fire,list(final_path)[:t+1])

        if path:
            path.popleft()
        if path:
            tr,tc = path.popleft()



        t+=1
        
        #print("path ", path)
        print("tr,tc",tr,tc)

    #print(res)
    return res, fire, final_path, t


def visualize_ship(ship, path):
    color_map = {
        0: 'black',  # Wall
        1: 'white',  # Empty space
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
        for i in range(1,len(path)-1):
            r, c = path[i]
            img [r, c] = mcolors.to_rgb('orange')

    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()



def main():

    ship_info = init_ship(40)
    q = .73

    # bot1(ship_info.copy(), q)
    bot3(ship_info.copy(), q)



   

if __name__ == "__main__":
    main()

