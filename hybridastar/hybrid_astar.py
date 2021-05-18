import ast
import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt
import math
import heapq
from dubins import dubins_heuristics
cell_size_orig = 2 # cell size aka each pixel size in m
cell_size_curr = 1 # cell size we are going to use the algorithm for
animate = False
n_cell_fractions = 5 # it divides each cell by 5 x 5
movement_step = 1.1 # 
STEER_ANGLES= [-6.5*3.14/180, 0, 6.5*3.14/180] # turning angles for each move
WB = 0.7 # rear to front wheel
astar_history = {}

class Node:
    
    def __init__(self, x, y, yaw=0.0) -> None:
        self.x = x
        self.y = y
        self.g = 99999 # initially it will be infinity
        self.h = 0
        self.f = 99999 # initially it will be infinity
        self.yaw = yaw
        self.h1 = 0.0
        self.h2 = 0.0

    def calculate_heuristics(self, end):
        self.h = self.distanceH(self, end)
    def move(self,step,end):
        n = Node(self.x+step[0], self.y + step[1])
        n.calculate_heuristics(end)        
        return n
    
    def distanceH(self, n1, end):
        return abs(n1.x - end[0])+ abs(n1.y - end[1]) 

    def distance(self, n1, end):
        return round(math.hypot(n1.x - end[0], n1.y - end[1]),2)
    def __eq__(self, o) -> bool:
        res = False
        if(o.x == self.x and o.y == self.y and o.yaw == self.yaw):
            res= True
        return res
    def __str__(self) -> str:
        return "[" + str(self.x) + ", " + str(self.y)  + "]" 
    def __repr__(self):
        return "[" + str(self.x) + ", " + str(self.y)  +"]" 

def check_collision(node:Node, map)->bool:
    if(node.x < 0 or node.x >= map.shape[0]):
        return 99999999
    if(node.y < 0 or node.y >= map.shape[1]):
        return 99999999
    if(map[int(node.x), int(node.y),0] < 100):
        return 99999999
    return False

def movements():
    # if our robot's direction is towards north
    return  [
        (0, 1, 1),#  east        
        # (1, 1, 1.14),# south east
        (1, 0, 1), # south
        # (1,-1, 1.14), # south west
        (0,-1, 1), # west
        # (-1,-1, 1.14), # north west
        (-1, 0, 1), # north
        # (-1, 1, 1.14) # north east
    ]
def movements_hybrid(x,y,yaw):
    movement = []
    for i in range(len(STEER_ANGLES)):
        dx,dy,dyaw = move(x,y,yaw, movement_step, STEER_ANGLES[i])
        STEERING_PINALTY = 0.00
        dcost = movement_step + STEERING_PINALTY * dyaw

        movement.append([round(dx,2),round(dy,2),round(dyaw,2), dcost])
    return movement

def calculate_index(node, shape):
    # cell_fractions_all = n_cell_fractions**2
    return (node.x * shape[0] + node.y) * 360 + node.yaw

def djirska(map, start, end):
    visited = {}
    queue = {}
    snode =Node(start[0],start[1])
    snode.g = 0
    snode.calculate_heuristics(end)
    snode.f = snode.g + snode.h    
    in_curr = calculate_index(snode, map.shape)
    in_goal = calculate_index(Node(end[0],end[1]), map.shape)
    max_size = map.shape[0] * map.shape[1]
    queue[in_curr] = snode
    heap = []
    heapq.heappush(heap, (snode.f, in_curr))
    if(map[int(end[0]),int(end[1]),0] < 100):
        print('target has an obstacle')
        return 99999
    
    steps = 0
    while(len(queue) >0):
        steps = steps + 1
        
        # search for the best node in the queue
        keys =list(queue.keys())
        _,in_curr = heapq.heappop(heap)

        curr = queue[in_curr]
        del queue[in_curr]

        if(int(curr.x) == int(end[0]) and int(curr.y) == int(end[1])): 
            # print("found the target")
            # print(curr)
            return curr.f
        
        # find the neighbors
        for step in movements():
            adjacentnode = curr.move(step, end)
            g = curr.g + step[2]

            # if(g < adjacentnode.g):
            adjacentnode.g = g
            adjacentnode.f = adjacentnode.g + adjacentnode.h + check_collision(adjacentnode, map)
            ind_tmp = calculate_index(adjacentnode, map.shape)
            
            if(not (ind_tmp in visited or ind_tmp in queue) ):      
                    astar_history[ind_tmp*max_size + in_goal] = adjacentnode.g
                    queue[ind_tmp] = adjacentnode
                    heapq.heappush(heap, (adjacentnode.f, ind_tmp))
            
        # add the current to visited
        visited[in_curr] = curr
        # map[curr.x, curr.y] = [255,0,0]
        
        # if(animate):
        #     plt.imshow(map)
        #     plt.waitforbuttonpress(0.1)
    return 99999

def pi_2_pi(angle):
    """
    this turns the angle between -p to p
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# this is a method based on dubins car model
def move(x,y,yaw, distance, steer, L=WB):
    # print("move")
    x = x + distance * math.cos(yaw)# + distance * math.sin(yaw * 3.14/180)
    y =y + distance * math.sin(yaw)#+ distance * math.cos(yaw * 3.14/180)
    yaw =yaw +  pi_2_pi(distance * math.tan(steer) / L)  # distance/2

    return x, y, yaw


def hybrid_astar(map, start, end, radius=0.2):
    visited = {}
    queue = {}
    
    dubins_history = {}
    startNode = Node(start[0], start[1], start[2])
    endNode = Node(end[0], end[1], end[2])
    curvature = 1./radius
    

    startNode.g = 0
    # here we have to find the initial heuristics
    # 1st heuristics is dubins path cost
    dubin_cost = dubins_heuristics(startNode, endNode, curvature)
    # print('dubins_cost=',dubin_cost)
    # 2nd heuristics is a start cost
    a_star_cost = djirska(map, [int(startNode.x),int(startNode.y)], [int(endNode.x), int(endNode.y)])
    print("astar cost=", a_star_cost)

    startNode.h = dubin_cost + a_star_cost

    startNode.f = startNode.g + startNode.h

    
     
    in_curr = calculate_index(startNode, map.shape)
    queue[in_curr] = startNode
    
    in_goal = calculate_index(Node(end[0],end[1]), map.shape)
    max_size = map.shape[0] * map.shape[1]

    astar_history[in_curr* max_size + in_goal] = a_star_cost
    heap = []
    heapq.heappush(heap, (startNode.f, in_curr))
    while(len(queue) >0):
        # search for the best node in the queue
        keys =list(queue.keys())
        _,in_curr = heapq.heappop(heap)

        curr = queue[in_curr]
        del queue[in_curr]

        if(abs(int(curr.x) -int(end[0])) == 0 and abs(int(curr.y) - int(end[1])) == 0): 
            print("found the target in hb")
            break
        else:
            print(curr)
        # get the movements here
        for movement in movements_hybrid(curr.x, curr.y, curr.yaw):
            tmpNode = Node(movement[0], movement[1], movement[2])
            in_tmp = calculate_index(tmpNode, map.shape)
            if(not (in_tmp in visited or in_tmp in queue or check_collision(tmpNode, map) >= 999) ):
                # now we need to calculage heuristics and g
                # 1st heuristics dubins path
                # print('starting to calculate dubins cost')
                
                if(in_tmp in dubins_history):
                    dubin_cost = dubins_history[in_tmp]
                else:
                    dubin_cost = dubins_heuristics(tmpNode, endNode, curvature)
                    dubins_history[in_tmp] = dubin_cost

                # 2nd heuristics a star cost
                # print('starting to calculate astar cost')
                if(in_tmp in astar_history):
                    a_star_cost = astar_history[in_tmp* max_size + in_goal ]
                else:
                    a_star_cost = djirska(map, [tmpNode.x,tmpNode.y], [endNode.x, endNode.y])
                    astar_history[in_tmp* max_size + in_goal] = a_star_cost
    

                # print(a_star_cost, dubin_cost)
                tmpNode.h = dubin_cost + a_star_cost
                tmpNode.g = curr.g + movement[3]
                tmpNode.f = tmpNode.g + tmpNode.h
                queue[in_tmp]= tmpNode
                heapq.heappush(heap, (tmpNode.f, in_tmp))
        visited[in_curr] = curr
        map[int(curr.x), int(curr.y)] = [255,0,0]
        
        if(animate):
            plt.imshow(map)
            plt.waitforbuttonpress(0.1)
def reconstruct_map(map_img):
    w,h,_ = map_img.shape
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)
    print("new map grid size", wn, hn)
    return map_new# + 255

if __name__ == "__main__":
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new
    import time 
    start = time.time()
    hybrid_astar(map_new, [0,0,0],[55/cell_size_curr,15/cell_size_curr,10 * 3.14/180], 5)
    print(time.time() - start)
    plt.imshow(map_new)
    plt.show()
    # plt.waitforbuttonpress()
