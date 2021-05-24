import numpy as np
from numpy import ma
import cv2
import matplotlib.pyplot as plt
import math
import heapq
from dubins import dubins_heuristics,dubins_path_planning, dubins_path_planning_costmap
from astar_heap_min import astar_heap, astar_heap_find_open_place
cell_size_orig = 1 # cell size aka each pixel size in m
cell_size_curr = 0.15 # cell size we are going to use the algorithm for
STEER_ANGLES = [-66*3.14/180, 0, 66 * 3.14/180]
movement_step =1.0
animate =False
cell_segment = 2 # divide each grid by 
WB = 2.1 # rear to front wheel
radius = 1.6
curvature = 1./radius
STEERING_PINALTY =10
HEURISTICS1_PINALTY=10
HEURISTICSASTAR_PINALTY = 100
DUBINS_PINALTY=5
start=(46, 73., np.deg2rad(90.0))
end=(57, 160., np.deg2rad(-0.0))
steps2expand = 7 # apply dubins curve to each steps2expand steps after finding the result # takes around 0.01 sec to accomplish

class Node:
    def __init__(self, x, y, yaw) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.g = 99999 # initially it will be infinity
        self.h = 0
        self.f = 99999 # initially it will be infin
        self.prev = None
        
    def new(self, pos):
        return Node(pos[0], pos[1], pos[2])

    def calculate_heuristics(self, end):
        self.h =  self.distanceH(self,end) +  DUBINS_PINALTY * dubins_heuristics(self, self.new(end), curvature)
    def move(self,step):
        n = Node(step[0], step[1],step[2])
        return n
    
    def distanceH(self, n1, end):
        return abs(n1.x - end[0])+ abs(n1.y - end[1]) 

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
    if(np.any(map[max(0,int(node.x)-3):int(node.x)+3, int(node.y),0] < 100)):
        return True
    return False


def pi_2_pi(angle):
    """
    this turns the angle between -p to p
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def move(x,y,yaw, distance, steer, L=WB):
    # print("move")
    # print((distance * math.cos(yaw* 3.14/180)))
    dx = distance * math.cos(yaw)# - distance * math.sin(yaw) 
    # dx = math.copysign(max()
    dy =  distance * math.sin(yaw) #+ distance * math.cos(yaw)
    dyaw = pi_2_pi(distance * math.tan(steer) / L) # distance/2
    return x + dx, y + dy, yaw + dyaw

def movements(x,y,yaw):
    # if our robot's direction is towards north
    dm =np.array([
        (0, 1,0, 1),#  east        
        (1, 1,0, 1.14),# south east
        (1, 0,0, 1), # south
        (1,-1,0, 1.14), # south west
        (0,-1,0, 1), # west
        (-1,-1,0, 1.14), # north west
        (-1, 0,0, 1), # north
        (-1, 1,0, 1.14) # north east
    ])
    dm[:,0] += x
    dm[:,1] += y
    return  dm
def movements_hybrid(x,y,yaw):
    movement = []
    for i in range(len(STEER_ANGLES)):
        # print(x,y,yaw, '--')
        dx,dy,dyaw = move(x,y,yaw, movement_step, STEER_ANGLES[i])
        # print(x,y,yaw,dx,dy,dyaw, '++')
        
        dcost = movement_step + STEERING_PINALTY * abs(yaw-dyaw)

        movement.append([dx,dy,dyaw, dcost])
    return movement

def calculate_index(node, shape):
    x_id = int(node.x * cell_segment)
    y_id = int(node.y * cell_segment)
    yaw_id = int(pi_2_pi(node.yaw) / (2 * 3.14/ 72))
    return yaw_id * shape[0] * shape[1]   +  x_id * shape[1] + y_id #+ int(node.yaw * 2 * 180/72)

     
def reconstruct_map(map_img):
    w,h,_ = map_img.shape
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)
    return map_new

def analytic_expansion(current, goal, costmap):
    if(current.x == goal.x and current.y == goal.y):
        return None, None, None
    start_x = current.x
    start_y = current.y
    start_yaw = current.yaw

    goal_x = goal.x
    goal_y = goal.y
    goal_yaw = goal.yaw

    max_curvature = math.tan(STEER_ANGLES[2]) / WB
    x_list, y_list, yaw_list, mode, lengths = dubins_path_planning_costmap(start_x, start_y, start_yaw, 
                                                                   goal_x, goal_y, goal_yaw, 
                                                                   curvature, 1.0, costmap)   
    # print("rc_calc paths finished")
    if(len(x_list) == 0):
        return  None, None, None
    
    # for i in range(len(x_list)):
    #     x = x_list[i]
    #     y = y_list[i]
    #     # yaw = yaw_list[i]
        
    #     if(x > costmap.shape[0] or x < 0 or y > costmap.shape[1] or y < 0):
    #         return None, None, None
    #     tt = costmap[int(x-1):int(x+1), int(y-1):int(y+1),0]   
    #     if(np.any(tt < 255)):
    #         return None, None, None
    #     else:
    #         if(y > 78 and x > 45 and y < 81 and x < 48):
    #             print(tt)
    #     #
    #     # if(np.any(tt < 255)):
    #     #     print(tt)
    return  x_list, y_list, yaw_list

def calculate_heuristics_astar(node, path):
    if(type(path) == list):
        path = np.array(path)
    if(len(path) == 0):
        return None # if a star couldnt find the path then we expect error
    if(not len(path[0]) == 3):
        path = path.T
    point = [node.x, node.y]
    diff = path[:,:-1] -  point
    dist = np.hypot(diff[:,0], diff[:,1])
    dist= dist[np.argmin(dist)]
    # print(np.argmin(dist))
    # print('ash', node, path[np.argmin(dist)])
    return dist#path[np.argmin(dist)][-1]

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
def hybrid_astar(map_new, start, end):
    
    visited = {}
    queue = {}
    snode =Node(start[0],start[1], start[2])
    snode.calculate_heuristics(end)
    snode.g = 0.0
    snode.f = snode.g + snode.h
    if(check_collision(snode.new(end),map_new)):
        print("target is in collision mode searching new target")
        end_ = astar_heap_find_open_place(map_new,np.int32(end[:-1]),np.int32(start[:-1]))
        end = [end_[0], end_[1], end_[2]]
    goal = snode.new(end)
    
    # plt.xticks(np.arange(0,map_new.shape[1], 0.5))
    # plt.yticks(np.arange(0,map_new.shape[0], 0.5))
    # initial astar search 
    astar_path = astar_heap(map_new, np.int32(start[:-1]), np.int32(end[:-1]))
    if(astar_path is None):
        print("astar could not find the path")
        return None, None, None
    if(animate):
        plt.plot(start[1], start[0], 'bo')  
        plt.plot(end[1], end[0], 'ro')
        xs_, ys_,_ = zip(*astar_path)
        plt.plot(ys_, xs_, "-r", label="astar path")
    
    in_curr = calculate_index(snode, map_new.shape)
    queue[in_curr] = snode
    heap = []
    heapq.heappush(heap, (snode.f, in_curr))
    # return None, None, None
        
    res_xs, res_ys, res_yaws =  None, None, None
    steps = 0
    max_cost = 0
    while(len(queue) >0):
        steps = steps + 1
        # print('steps = ', steps)
        
        # search for the best node in the queue
        keys =list(queue.keys())
        _,in_curr = heapq.heappop(heap)

        curr = queue[in_curr]
        del queue[in_curr]
    
        
        # print(curr, map_new[int(curr.x), int(curr.y)])
        # break
        
        if(animate):          
            
            plt.imshow(map_new)
            
            # plt.pause(0.01)
            # map_new[int(curr.x), int(curr.y), 1] = 100
            # ax.wait(0.1)
            plt.waitforbuttonpress(0.01)
        # break    
    
        res_xs, res_ys, res_yaws = analytic_expansion(curr, goal, map_new)    
        if(res_xs is not None and res_ys is not None and res_yaws is not None or steps >= 2000): 
            res = [] 
            
            while(True):
                res.append([curr.x, curr.y, curr.yaw])
                curr = curr.prev
                if(curr is None):
                    break
            res.reverse()            
            
            # res.append([end[0], end[1], end[2]])
            
            rxs,rys,ryaws = [], [], []
            for i in np.arange(0, len(res) - steps2expand, steps2expand):                
                ind_end = min(i + steps2expand, len(res)-1)
                res_xs_, res_ys_, res_yaws_ = analytic_expansion(goal.new(res[i]), goal.new(res[ind_end]), map_new)    
                
                if(res_xs_ is not None and res_ys_ is not None and res_yaws_ is not None):
                    rxs.extend(res_xs_)
                    rys.extend(res_ys_)
                    ryaws.extend(res_yaws_)
                else:                
                    restmp = res[i:ind_end+1]
                    rxs_tmp, rys_tmp, ryaws_tmp = zip(*restmp)
                    rxs.extend(rxs_tmp)
                    rys.extend(rys_tmp)
                    ryaws.extend(ryaws_tmp)
            if(res_xs is not None):
                rxs.extend(res_xs)
                rys.extend(res_ys)
                ryaws.extend(res_yaws)
            return rxs, rys, ryaws
        
        
        all_movements = movements_hybrid(curr.x, curr.y , curr.yaw)
        
        for step in all_movements:
            adjacentnode = curr.move(step)
            adjacentnode.prev = curr
            heuristics_astar = calculate_heuristics_astar(adjacentnode, astar_path)
            g = curr.g + step[-1]     
            adjacentnode.calculate_heuristics(end)       
            # if(g > curr.g):
            adjacentnode.g = g
            adjacentnode.f = adjacentnode.g +HEURISTICS1_PINALTY*adjacentnode.h +\
                                HEURISTICSASTAR_PINALTY*heuristics_astar + \
                                100 * (255-map_new[int(adjacentnode.x), int(adjacentnode.y), 0])
            ind_tmp = calculate_index(adjacentnode, map_new.shape)
            # if(adjacentnode.h <= curr.h):
            # print("index",ind_tmp, step)
            if(not (ind_tmp in visited or ind_tmp in queue) ):
               
                # print(adjacentnode, "not visited")
                if(not check_collision(adjacentnode, map_new)):
                    if(animate):  
                        if(max_cost < adjacentnode.f):
                            max_cost = adjacentnode.f
                            #add rectangle to plot
                        plt.gca().add_patch(plt.Rectangle((int(adjacentnode.y * cell_segment)/cell_segment,  
                                                int(adjacentnode.x * cell_segment)/cell_segment), 
                                                1/cell_segment, 1/cell_segment,
                                      edgecolor = [0,(max_cost -adjacentnode.f)/max_cost,  (max_cost -adjacentnode.f)/max_cost],
                                      facecolor = [0,(max_cost -adjacentnode.f)/max_cost, (max_cost -adjacentnode.f)/max_cost],
                                      fill=True,
                                      lw=5))
                        plt.text(int(adjacentnode.y * cell_segment)/cell_segment, 
                                  int(adjacentnode.x * cell_segment)/cell_segment, 
                                  str(round(adjacentnode.f)), fontsize=8)
                    queue[ind_tmp] = adjacentnode
                    heapq.heappush(heap, (adjacentnode.f, ind_tmp))
                    
        # add the current to visited
        visited[in_curr] = curr
        
    print("end returning none")
    return None, None, None

def plot_arrow(x, y, yaw, length=2.0, width=0.8, fc="r",
               ec="k"):  # pragma: no cover
    """
    Plot arrow10
    """

    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * math.sin(yaw), length * math.cos(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)



if __name__ == "__main__":
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new
    import time 
    
    print("input map shape", map_in.shape)
    map_in = cv2.blur(map_in, (3, 3))   
    start_time = time.time()    
    res_xs, res_ys, res_yaws = hybrid_astar(map_in ,start,end)
    print(time.time() - start_time)
    plot_arrow(start[1], start[0],start[2], fc='g')
    plot_arrow(end[1], end[0],end[2], fc='g')
    # plot_arrow(res_ys, res_xs,res_yaws, fc='r')
   
    # fig, ax = plt.subplots()
    # plt.cla()
    if(res_xs is not None):
        # for i in range(len(res_ys)):
            # plt.text(res_ys[i],  res_xs[i], str(i), fontsize=8)
        plt.plot(res_ys, res_xs, "b-",markersize=1,  label="Hybrid A* path")
    else:
        print("hybrid start could not find the object")
    plt.grid(True)
    plt.axis("equal")
    plt.imshow(map_in[:,:,0])
    plt.plot()
    
    plt.legend()
    plt.show()
