import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque

class Ship:
    
    def __init__(self, dimension):

        self.dimension = dimension
        directions = [(0,1),(1,0),(-1,0),(0,-1)] # array to store adjacent directions needed during traversal
        
        d = dimension

        # initialize ship 2D list to size dimension
        self.ship = [[0] * d for _ in range(d)] 
        
        # open up a random cell on interior
        to_open = random.sample(range(1,d-1), 2) # range is set from 1 to d-1 to ensure we select from interior
        row, col = to_open

        self.ship[row][col] = 1

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
            self.ship[row][col] = 1 # open it up

            for dr,dc in directions:
                r = row + dr
                c = col + dc
                if 0 <= r < d and 0 <= c < d and self.ship[r][c] == 0 and (r,c) not in closed:
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
                if self.ship[r][c] == 1: # open cell
                    open_n_count = 0
                    closed_neighbors = []
                    for dr,dc in directions:
                        nr,nc = r + dr, c + dc
                        if 0 <= nr < d and 0 <= nc < d:
                            if self.ship[nr][nc] == 1: # open neighbor
                                open_n_count += 1
                            elif self.ship[nr][nc] == 0:
                                closed_neighbors.append((nr,nc))
                    if open_n_count == 1:
                        deadends[(r,c)] = closed_neighbors

        # for ~ 1/2 of deadend cells, pick 1 of their closed neighbors at random and open it
        for i in range(len(deadends)//2):
            list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys()))) # retrieve a random deadend cell's list of closed neighbors
            r,c = random.choice(list_closed_neighbors) # choose a random closed neighbor
            self.ship[r][c] = 1 # open it

        # create sets that  store coordinates of all the open cells and all the closed cells
        open_cells = set()
        closed = set()
        for r in range(d):
            for c in range(d):
                if self.ship[r][c] == 1:
                    open_cells.add((r,c))
                elif self.ship[r][c] == 0:
                    closed.add((r,c))

        # randomly place bot in one of the remaining open cells
        self.bot_r, self.bot_c = (random.choice(list(open_cells)))
        open_cells.remove((self.bot_r, self.bot_c))
        
        # randomly place fire in one of the remaining open cells
        self.fire_r, self.fire_c = (random.choice(list(open_cells)))
        open_cells.remove((self.fire_r, self.fire_c))

        # randomly place button in one of the remaining open cels
        self.button_r,self.button_c = (random.choice(list(open_cells)))
        open_cells.remove((self.button_r, self.button_c))

        self.ship[self.bot_r][self.bot_c] = 2
        self.ship[self.fire_r][self.fire_c] = -1
        self.ship[self.button_r][self.button_c] = -2
    
    def visualize(self):
        
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
        
        d = self.dimension
        img = np.zeros((d, d, 3))
        
        for i in range(d):
            for j in range(d):
                img[i, j] = mcolors.to_rgb(color_map[self.ship[i][j]])  

        plt.imshow(img, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.show()

my_ship = Ship(dimension=40)
my_ship.visualize()