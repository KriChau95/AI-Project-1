import random
import numpy as np


#np.random.seed(23) #23 for lebron


def initShip(d):
        
    dimension = d

    #initialize ship to size dimension
    #0 will be closed, 1 will be open, -1 will be fire

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
        row,col = singleNeighbor.pop()
        
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


    return ship



def main():
    ship = initShip(10)
    



if __name__ == "__main__":
    main()

