import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque

random.seed(23) # 23 for lebron

def initShip(d):
        
    dimension = d
    
    # 0 = closed
    # 1 = open 
    # 2 = bot
    # -2 = button
    # -1 = fire

    # initialize ship to size dimension
    ship = [[0] * dimension for t in range(dimension)] 

    for row in ship:
        print(row)
    
    return createShip(ship)

def createShip(ship):    
    d = len(ship)
    toOpen = random.sample(range(1,d-1), 2) # range is set from 1 to d-1 to ensure we select from interior

    row,col= toOpen
    ship[row][col] = 1

    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    singleNeighbor = set()
    closed = set()

    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < d and 0 <= c < d:
            singleNeighbor.add((r,c))

    while singleNeighbor:
        print(singleNeighbor)
        for row in ship:
            print(row)
        rc = (random.choice(list(singleNeighbor)))
        singleNeighbor.remove(rc)
        row, col = rc
        ship[row][col] = 1
        for dr,dc in directions:
            r = row+dr
            c = col+dc
            if 0<=r<d and 0<=c<d and ship[r][c] == 0 and (r,c) not in closed:
                if (r,c) in singleNeighbor:
                    singleNeighbor.remove((r,c))
                    closed.add((r,c))
                else:
                    singleNeighbor.add((r,c))

    deadends = dict()

    for r in range(d):
        for c in range(d):
            if ship[r][c] == 1:
                count = 0
                closedN = []
                for dr,dc in directions:
                    tr = r+dr 
                    tc = c+dc
                    if 0<=tr<d and 0<=tc<d:
                        if ship[tr][tc] == 1:
                            count+=1
                        elif ship[tr][tc] == 0:
                            closedN.append((tr,tc))
                if count == 1:
                    deadends[(r,c)] = closedN

    print("dead ends of ship ", deadends)

    open = set()
    closed = set()
    for r in range(len(ship)):
        for c in range(len(ship)):
            if ship[r][c] == 1:
                open.add((r,c))
            elif ship[r][c] == 0:
                closed.add((r,c))

    for i in range(len(deadends)//2):
        listNeighbors = deadends.pop(random.choice(list(deadends.keys())))
        r,c = random.choice(listNeighbors)
        ship[r][c] = 1
    
    
    print("dead ends after popping ", deadends)

    bot_r,bot_c = (random.choice(list(open)))
    open.remove((bot_r,bot_c))
    fire_r,fire_c = (random.choice(list(open)))
    open.remove((fire_r,fire_c))

    button_r,button_c = (random.choice(list(open)))
    open.remove((button_r,button_c))

    ship[bot_r][bot_c] = 2
    ship[fire_r][fire_c] = -1
    ship[button_r][button_c] = -2

    fire_set = set()

    for dr, dc in directions:
        nr, nc = fire_r + dr, fire_c + dc
        if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == 1:
            fire_set.add((nr,nc))

    items = [bot_r,bot_c,fire_r,fire_c,button_r,button_c, fire_set]
    return ship, open, closed, items

def bot1(ship,open,closed,items, q):
    #find path from bot to goal
    bot_r,bot_c,fire_r,fire_c,button_r,button_c, fire_set = items
    d = len(ship)

    def heuristic(cell1):
        cell2 = (button_r,button_c)
        return abs(cell1[0] - cell2[0]) + abs(cell1[1]-cell2[1])

    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    heap = []
    heapq.heappush(heap, (0,bot_r,bot_c))
    prev = {}
    totalCost = {}
    prev[(bot_r,bot_c)] = None
    totalCost[(bot_r, bot_c)] = 0

    # print("running a*", heap, prev)
    sol = None
    visited = set()
    while heap:
        cost, r, c =  heapq.heappop(heap)
        if (r,c) == (button_r,button_c):
            sol = (r,c)
            break
        for dr,dc in directions:
            row = r + dr
            col = c + dc
            if 0<=row<d and 0<=col<d and (ship[row][col] != 0 and ship[row][col] != -1 )and (row,col) not in visited:
                estCost = cost + heuristic((row,col))
                heapq.heappush(heap, (estCost,row,col))
                prev[(row,col)] = (r,c)
                visited.add((r,c))

    if sol != None:
        curr = sol
        path = []
        print(prev, curr)

        while curr != None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        print("final path ",path)
    
    
    fire_q = deque()
    for i in fire_set:
        deque.append(i)
    
    count = 1
    while deque:
        curr_r, curr_c = path[t]
        for _ in list(fire_set):
            row, col = fire_set.pop()
            k = 0
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < d and 0 <= nc < d and ship[nr][nc] == -1:
                    k += 1
            # prob_f = 1 - (1-q) ** k
            # result = random.choices(["fire", "no fire"], weights=[prob_f, 1- prob_f])[0]



    return path


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
            img[i, j] = mcolors.to_rgb(color_map[ship[i][j]])  # Use correct function
    
    for i in range(1,len(path)-1):
        r, c = path[i]
        img [r, c] = mcolors.to_rgb('yellow')

    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()



def main():
    ship, open, closed, items = initShip(40)
    q = 0.1
    path = bot1(ship.copy(),open,closed,items, q)
    visualize_ship(ship, path)
    

if __name__ == "__main__":
    main()

