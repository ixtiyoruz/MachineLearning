import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt

cell_size_orig = 1 # cell size aka each pixel size in m
cell_size_curr = 0.2 # cell size we are going to use the algorithm for
class Node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
    def move(self,dx,dy):
        return Node(self.x+dx, self.y + dy)
    def __eq__(self, o) -> bool:
        if(o.x == self.x and o.y == self.y):
            return True
        return False
    def __str__(self) -> str:
        return "[" + str(self.x) + ", " + str(self.y) + "]" 
    def __repr__(self):
        return "[" + str(self.x) + ", " + str(self.y) + "]" 

def check_collision(node:Node, map)->bool:
    if(node.x < 0 or node.x >= map.shape[0]):
        return True
    if(node.y < 0 or node.y >= map.shape[1]):
        return True
    print(node.x, node.y, map.shape)
    if(map[node.x, node.y] < 100):
        return True
    return False

def movements():
    # if our robot's direction is towards north
    return  [
        (0, 1),#  east        
        (1, 1),# south east
        (1, 0), # south
        (1,-1), # south west
        (0,-1), # west
        (-1,-1), # north west
        (-1, 0), # north
        (-1, 1) # north east
    ]

def djirska(map, start=(0,0), end=(20,20)):
    costmat = np.zeros_like(map)
    visited = []
    queue = []
    queue.append(Node(start[0],start[1]))
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        print('steps = ', steps)
        curr = queue.pop(-1)
        if(curr.x == end[0] and curr.y == end[1]): 
            print("found the target")
            break
        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step[0],step[1])
            if(adjacentnode in visited):
                continue
            # check for collision
            # to do
            if(check_collision(adjacentnode, map)):
                continue
            # add to the queue
            queue.append(adjacentnode)

        # add the current to visited
        visited.append(curr)
        # print("queue", queue)
        print("curr", curr, map.shape)
    
def reconstruct_map(map_img):
    # print(map_img.shape)
    w,h,_ = map_img.shape
    # print(np.max(map_img[:,:,2]), np.mean(map_img[:,:,1]))
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)

    # for i in range(map_img.shape[0]):
        # for j in range(map_img.shape[1]):
            # map_new[i,j] = map[]
    return map_new

if __name__ == "__main__":

    print("hello world")
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new[:,:,0]
    djirska(map_in)
    plt.imshow(map_new)
    plt.show()
    # plt.waitforbuttonpress()
