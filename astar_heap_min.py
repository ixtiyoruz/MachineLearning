import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt
import math
import heapq
# cell_size_orig = 1 # cell size aka each pixel size in m
# cell_size_curr = 0.15 # cell size we are going to use the algorithm for
animate = False
class Node:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.g = 99999 # initially it will be infinity
        self.h = 0
        self.f = 99999 # initially it will be infinity
        self.prev = None

    def calculate_heuristics(self, end):
        self.h = self.distanceH(self, end)
    def move(self,step,end=None):
        n = Node(self.x+step[0], self.y + step[1])
        if(end is not None):
            n.calculate_heuristics(end)        
        return n
    def new(coor):
        return Node(coor[0],  coor[1])
    def distanceH(self, n1, end):
        return (abs(n1.x - end[0])+ abs(n1.y - end[1])) *100

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
    if(np.any(map[node.x-2:node.x + 2, node.y-2:node.y + 2,0]< 100)):
        return True
    return False

def movements():
    # if our robot's direction is towards north
    return  [
        (0, 1, 1),#  east        
        (1, 1, 1.14),# south east
        (1, 0, 1), # south
        (1,-1, 1.14), # south west
        (0,-1, 1), # westcell_size_orig
        (-1,-1, 1.14), # north west
        (-1, 0, 1), # north
        (-1, 1, 1.14) # north east
    ]
def calculate_index(node, shape):
    return node.x * shape[0] + node.y

def astar_heap(costmap, start, end):
    print(start, end)
    visited = {}
    queue = {}
    snode =Node(start[0],start[1])
    snode.g = 0
    snode.calculate_heuristics(end)
    snode.f = snode.g + snode.h    
    in_curr = calculate_index(snode, costmap.shape)
    queue[in_curr] = snode
    heap = []
    heapq.heappush(heap, (snode.f, in_curr))
    if(costmap[end[0],end[1],0] < 100):
        print('target has an obstacle')
        return None
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        # print('steps = ', steps)
        
        # search for the best node in the queue
        keys =list(queue.keys())
        _,in_curr = heapq.heappop(heap)

        curr = queue[in_curr]
        del queue[in_curr]

        if(curr.x == end[0] and curr.y == end[1]): 
            print("found the target")
            res = []
            while(True):
                res.append([curr.x, curr.y, curr.g])
                curr = curr.prev
                if(curr is None):
                    break
            # res.reverse()
            return res
        

        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step, end)
            adjacentnode.prev = curr
            g = curr.g + 100*step[2] + 100 *(255-costmap[int(adjacentnode.x), int(adjacentnode.y), 0])

            # if(g < adjacentnode.g):
            adjacentnode.g = g
            adjacentnode.f = adjacentnode.g + adjacentnode.h
            ind_tmp = calculate_index(adjacentnode, costmap.shape)
            if(not (ind_tmp in visited or ind_tmp in queue) ):
                if(not check_collision(adjacentnode, costmap)):
                    queue[ind_tmp] = adjacentnode
                    heapq.heappush(heap, (adjacentnode.f, ind_tmp))
            
        # add the current to visited
        visited[in_curr] = curr
        # print(np.shape(costmap))
        # costmap[curr.x, curr.y] = [127,0,127]
        
        # if(animate):
        #     plt.imshow(costmap)
        #     # plt.waitforbuttonpress(0.1)
        #     plt.show()
def astar_heap_find_open_place(costmap, start, end):
    visited = {}
    queue = {}
    snode =Node(start[0],start[1])
    snode.g = 0
    # snode.calculate_heuristics(end)
    snode.f = snode.g+snode.h 
    in_curr = calculate_index(snode, costmap.shape)
    print(in_curr)
    queue[in_curr] = snode
    heap = []
    heapq.heappush(heap, (snode.f, in_curr))
    
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        # print('steps = ', steps)
        
        # search for the best node in the queue
        keys =list(queue.keys())
        _,in_curr = heapq.heappop(heap)

        curr = queue[in_curr]
        del queue[in_curr]
        
        # find endnode
        if(not check_collision(curr, costmap)): 
            print("found the target")
            return [curr.x, curr.y, math.atan2(curr.y, curr.x)]

        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step, None)
            adjacentnode.prev = curr
            g = curr.g + step[2]
            
            adjacentnode.g = g
            adjacentnode.f = adjacentnode.g#+ adjacentnode.h
            ind_tmp = calculate_index(adjacentnode, costmap.shape)
            if(not (ind_tmp in visited or ind_tmp in queue) ):
                queue[ind_tmp] = adjacentnode
                heapq.heappush(heap, (adjacentnode.f, ind_tmp))
            
        visited[in_curr] = curr
    
def reconstruct_map(map_img, cell_size_curr = 1, cell_size_orig = 1.0):
    w,h,_ = map_img.shape
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)
    return map_new

if __name__ == "__main__":
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new
    import time 
    start = time.time()
    astar_heap(map_in, (10,10),(30, 25))
    print(time.time() - start)
    plt.imshow(map_new)
    plt.show()
    # plt.waitforbuttonpress()
