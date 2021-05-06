import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt
import math

cell_size_orig = 1 # cell size aka each pixel size in m
cell_size_curr = 0.05 # cell size we are going to use the algorithm for
animate = False
class Node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.g = 99999 # initially it will be infinity
        self.h = 0
        self.f = 99999 # initially it will be infinity

    def calculate_heuristics(self, end):
        self.h = self.distanceH(self, end)
    def move(self,step,end):
        n = Node(self.x+step[0], self.y + step[1])
        n.calculate_heuristics(end)        
        return n
    
    def distanceH(self, n1, end):
        return abs(n1.x - end[0]) + abs(n1.y - end[1])
    def distance(self, n1, end):
        return round(math.hypot(n1.x - end[0], n1.y - end[1]),2)
    def __eq__(self, o) -> bool:
        res = False
        if(o.x == self.x and o.y == self.y):
            res= True
        return res
    def __str__(self) -> str:
        return "[" + str(self.x) + ", " + str(self.y)  + "]" 
    def __repr__(self):
        return "[" + str(self.x) + ", " + str(self.y)  +"]" 

def check_collision(node:Node, map)->bool:
    if(node.x < 0 or node.x >= map.shape[0]):
        return True
    if(node.y < 0 or node.y >= map.shape[1]):
        return True
    if(map[node.x, node.y] < 100):
        return True
    return False

def movements():
    # if our robot's direction is towards north
    return  [
        (0, 1, 1),#  east        
        (1, 1, 1.14),# south east
        (1, 0, 1), # south
        (1,-1, 1.14), # south west
        (0,-1, 1), # west
        (-1,-1, 1.14), # north west
        (-1, 0, 1), # north
        (-1, 1, 1.14) # north east
    ]
def calculate_index(node, shape):
    return node.x * shape[0] + node.y

def djirska(map, start=(10,28), end=(270,293)):
    costmat = np.zeros_like(map)
    visited = {}
    queue = {}
    snode =Node(start[0],start[1])
    snode.g = 0
    snode.f = snode.g + snode.h
    snode.calculate_heuristics(end)
    queue[calculate_index(snode, map.shape)] = snode
    
    if(map[end[0],end[1]] < 100):
        print('target has an obstacle')
        return
    
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        print('steps = ', steps)
        
        # search for the best node in the queue
        keys =list(queue.keys())
        dist = queue[keys[0]].f
        in_curr = keys[0]
        for tmpnk in keys:
            if(queue[tmpnk].f < dist):
                dist  = queue[tmpnk].f
                in_curr = tmpnk

        curr = queue[in_curr]
        del queue[in_curr]

        if(curr.x == end[0] and curr.y == end[1]): 
            print("found the target")
            break

        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step, end)
            g = curr.g + step[2]

            if(g < adjacentnode.g):
                adjacentnode.g = g
                adjacentnode.f = adjacentnode.g + adjacentnode.h
                ind_tmp = calculate_index(adjacentnode, map.shape)
                if(not (ind_tmp in visited or ind_tmp in queue) ):
                    if(not check_collision(adjacentnode, map)):                    
                        queue[ind_tmp] = adjacentnode
            
        # add the current to visited
        visited[in_curr] = curr
        map[curr.x, curr.y] = 50
        
        if(animate):
            plt.imshow(map_new)
            plt.waitforbuttonpress(0.1)
            
        print("curr", curr, map.shape, in_curr)
    
def reconstruct_map(map_img):
    w,h,_ = map_img.shape
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)
    return map_new

if __name__ == "__main__":
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new[:,:,0]
    djirska(map_in)
    plt.imshow(map_new)
    plt.show()
    # plt.waitforbuttonpress()
