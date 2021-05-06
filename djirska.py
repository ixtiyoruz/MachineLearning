import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt
import math

cell_size_orig = 1 # cell size aka each pixel size in m
cell_size_curr = 0.1 # cell size we are going to use the algorithm for
animate = False
class Node:
    def __init__(self, x, y, dist, shape) -> None:
        self.x = x
        self.y = y
        self.dist = dist
        self.index = self.x * shape[0] + self.y
    def move(self,dx,dy ,end, shape):
        n = Node(self.x+dx, self.y + dy, 99999999, shape)
        n.dist = self.distance(n,end)
        return n
    def distance(self, n1, end):
        return round(math.hypot(n1.x - end[0], n1.y - end[1]),2)
    def __eq__(self, o) -> bool:
        res = False
        if(o.x == self.x and o.y == self.y):
            res= True
        # print(o, self, res)
        return res
    def __str__(self) -> str:
        return str(self.index)#"[" + str(self.x) + ", " + str(self.y) + ", " + str(self.dist) + "]" 
    def __repr__(self):
        return str(self.index)#"[" + str(self.x) + ", " + str(self.y) + ", " + str(self.dist) +"]" 

def check_collision(node:Node, map)->bool:
    if(node.x < 0 or node.x >= map.shape[0]):
        return True
    if(node.y < 0 or node.y >= map.shape[1]):
        return True
    # print(node.x, node.y, map.shape)
    if(map[node.x, node.y] < 100):
        return True
    return False

def movements():
    # if our robot's direction is towards north
    return  [
        (0, 1),#  east        
        (1, 1,),# south east
        (1, 0), # south
        (1,-1), # south west
        (0,-1), # west
        (-1,-1), # north west
        (-1, 0), # north
        (-1, 1) # north east
    ]

def djirska(map, start=(10,28), end=(270,293)):
    costmat = np.zeros_like(map)
    visited = []
    queue = []
    queue.append(Node(start[0],start[1], 0,map.shape))
    if(map[end[0],end[1]] < 100):
        print('target has an obstacle')
        return
    
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        print('steps = ', steps)
        
        # search for the best node in the queue
        dist = queue[0].dist
        in_curr = 0
        for i in range(1,len(queue)):
            if(queue[i].dist < dist):
                dist  = queue[i].dist
                in_curr = i

        curr = queue[in_curr]
        del queue[in_curr]

        if(curr.x == end[0] and curr.y == end[1]): 
            print("found the target")
            break

        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step[1],step[0], end, map.shape)
            if(not (adjacentnode.index in visited or adjacentnode in queue) ):
                if(not check_collision(adjacentnode, map)):
                    queue.append(adjacentnode)
            
        # add the current to visited
        visited.append(curr.index)
        map[curr.x, curr.y] = 150
        
        if(animate):
            plt.imshow(map_new)
            plt.waitforbuttonpress(1)
            
        # print("curr", curr, map.shape, in_curr)
    
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
