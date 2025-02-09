import random
import numpy as np
import heapq

random.seed(23) #23 for lebron


def initShip(d):
        
    dimension = d

    #initialize ship to size dimension
    #0 will be closed, 1 will be open, 
    #2 will be bot, -2 will be button, -1 will be fire

    ship = [[0] * dimension for t in range(dimension)] 


    for row in ship:
        print(row)
    
    return createShip(ship)
        


def createShip(ship):    
    d = len(ship)
    toOpen = random.sample(range(d), 2)

    row,col= toOpen
    ship[row][col] = 1

    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    singleNeighbor = set()
    closed = set()

    for dr,dc in directions:
        r = row+dr
        c = col+dc
        if 0<=r<d and 0<=c<d:
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


    open = set()
    closed = set()
    for r in range(len(ship)):
        for c in range(len(ship)):
            if ship[r][c] == 1:
                open.add((r,c))
            elif ship[r][c] == 0:
                closed.add((r,c))

    bot_r,bot_c = (random.choice(list(open)))
    open.remove((bot_r,bot_c))
    
    fire_r,fire_c = (random.choice(list(open)))
    open.remove((fire_r,fire_c))

    button_r,button_c = (random.choice(list(open)))
    open.remove((button_r,button_c))

    ship[bot_r][bot_c] = 2
    ship[fire_r][fire_c] = -1
    ship[button_r][button_c] = -2
    print("bot1 initialized ship ")
    for row in ship:
        print(row)

    items = [bot_r,bot_c,fire_r,fire_c,button_r,button_c]
    return ship, open, closed, items

def bot1(ship,open,closed,items):
    #find path from bot to goal
    bot_r,bot_c,fire_r,fire_c,button_r,button_c = items
    d = len(ship)

    def heuristic(cell1):
        cell2 = (button_r,button_c)
        return abs(cell1[0] - cell2[0]) + abs(cell1[1]-cell2[1])

    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    heap = []
    heapq.heappush(heap, (0,bot_r,bot_c,))
    prev = {}
    totalCost = {}
    prev[(bot_r,bot_c)] = None
    totalCost[(bot_r, bot_c)] = 0

    print("running a*", heap, prev)
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
  

    



    return




def main():
    ship, open, closed, items = initShip(10)

    bot1(ship.copy(),open,closed,items)
    



if __name__ == "__main__":
    main()

