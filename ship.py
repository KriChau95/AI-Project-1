import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque

random.seed(4) # 23 for lebron

def initShip(d):
        
    dimension = d
    
    # 0 = closed
    # 1 = open 
    # 2 = bot
    # -2 = button
    # -1 = fire

    # initialize ship to size dimension
    ship = [[0] * dimension for t in range(dimension)] 

    #for row in ship:
        #print(row)
    
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
        #print(singleNeighbor)
        #for row in ship:
            #print(row)
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

    #print("dead ends of ship ", deadends)

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
    
    
    #print("dead ends after popping ", deadends)

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

    # #print("running a*", heap, prev)
    sol = None
    visited = set()
    while heap:
        cost, r, c =  heapq.heappop(heap)
        if ship[r][c] == -2:
            sol = (r,c)
            break
        visited.add((r,c))
        for dr,dc in directions:
            row = r + dr
            col = c + dc
            if 0<=row<d and 0<=col<d and (ship[row][col] != 0 and ship[row][col] != -1 )and (row,col) not in visited:
                estCost = cost + heuristic((row,col))
                heapq.heappush(heap, (estCost,row,col))
                prev[(row,col)] = (r,c)

    if sol != None:
        curr = sol
        path = []
        #print(prev, curr)

        while curr != None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        
        #print("final path ",path)
    
    inQueue = set()

    fire_q = deque()
    for i in fire_set:
        fire_q.append(i)
        inQueue.add(i)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]

    while fire_q and t<len(path):

        tr, tc = path[t]

        if fire[tr][tc] == -1:
            res = "failure"
            break

        visualize_ship(fire,path[0:t+1],None)

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

def bot2(ship,open,closed,items, q):
    #find path from bot to goal
    bot_r,bot_c,fire_r,fire_c,button_r,button_c, fire_set = items
    d = len(ship)
    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    def heuristic(cell1):
        cell2 = (button_r,button_c)
        return abs(cell1[0] - cell2[0]) + abs(cell1[1]-cell2[1])

    def astar(tempr,tempc,map):
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
                    estCost = cost + heuristic((row,col))
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
            return sol, path
        else:
            return None, None
    
    # for i in ship:
    #     print(i)
    # print(astar(12,5,ship))

    inQueue = set()
    fire_q = deque()

    for i in fire_set:
        fire_q.append(i)
        inQueue.add(i)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]
    tr, tc = bot_r,bot_c
    final_path = []
    while fire_q:
        final_path.append((tr,tc))

        if (tr, tc) == (button_r, button_c):
            break

        sol, path = astar(tr,tc,fire)

        print("path", path)

        if path == None:
            res = "failure"
            break
    
        if fire[tr][tc] == -1:
            res = "failure"
            break
        
        visualize_ship(fire,final_path,None)

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

def bot3(ship,open,closed,items, q):
    #find path from bot to goal
    bot_r,bot_c,fire_r,fire_c,button_r,button_c, fire_set = items
    d = len(ship)
    directions = [[0,1],[1,0],[-1,0],[0,-1]]

    def heuristic(cell1):
        cell2 = (button_r,button_c)
        return abs(cell1[0] - cell2[0]) + abs(cell1[1]-cell2[1])

    def astar(tempr,tempc,map):
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
                    estCost = cost + heuristic((row,col))
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
            return sol, path
        return None, None
    
    # for i in ship:
    #     print(i)
    # print(astar(12,5,ship))

    inQueue = set()
    fire_q = deque()

    for i in fire_set:
        fire_q.append(i)
        inQueue.add(i)

    t = 1
    res = "success"
    fire = [row.copy() for row in ship]
    tr, tc = bot_r,bot_c
    final_path = []
    while fire_q:

        final_path.append((tr,tc))

        if (tr, tc) == (button_r, button_c):
            print("tr==tc, breaking")
            break
        

        inflation_map = [row.copy() for row in fire]

        for row in range(d):
            for col in range(d):
                if fire[row][col] == -1:
                    for dr,dc in directions:
                        r = row + dr
                        c = col + dc
                        if 0 <= r < d and 0 <= c < d:
                            inflation_map[r][c] = -1


        sol, path = astar(tr,tc,inflation_map)

        if path == None:
            sol, path = astar(tr,tc,fire)


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


        visualize_ship(fire,final_path,None)

        if path:
            path.popleft()
        if path:
            tr,tc = path.popleft()



        t+=1
        
        #print("path ", path)
        print("tr,tc",tr,tc)

    #print(res)
    return res, fire, final_path, t


def visualize_ship(sol_ship, sol_path,t):
    plt.clf()
    plt.close()
    plt.figure()
    color_map = {
        0: 'black',  # Wall
        1: 'white',  # Empty space
        -1: 'red',   # Fire
        2: 'blue',   # Bot
        -2: 'green'  # Button
    }
    
    d = len(sol_ship)
    img = np.zeros((d, d, 3))
    
    for i in range(d):
        for j in range(d):
            img[i, j] = mcolors.to_rgb(color_map[sol_ship[i][j]])  
    if t:
        i = 0
        while i<len(sol_path)-1 and i < t+1:
            r, c = sol_path[i]
            img [r, c] = mcolors.to_rgb('orange')
            i+=1
    else:
        for i in range(1,len(sol_path)-1):
            r, c = sol_path[i]
            img [r, c] = mcolors.to_rgb('orange')

    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.show()




def main():
    ship, open, closed, items = initShip(40)
    q = .73
    start_ship1 = [row.copy() for row in ship]
    start_ship2 = [row.copy() for row in ship]
    start_ship3 = [row.copy() for row in ship]

    res1,ship1, path1, t1 = bot1(start_ship1,open,closed,items,q)
    print(res1,"bot1")
    visualize_ship(ship1, path1,t1)

    res2, ship2, path2,t2 = bot2(start_ship2,open,closed,items,q)
    print(res2,"bot2")
    visualize_ship(ship2, path2,t2)
    
    res3,ship3, path3,t3 = bot3(start_ship3,open,closed,items,q)
    print(res3,"bot3")
    visualize_ship(ship3, path3,t3)

   

if __name__ == "__main__":
    main()

