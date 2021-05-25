import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import heapq
from dubins import dubins_heuristics, dubins_path_planning_costmap
from astar_heap_min import astar_heap, astar_heap_find_open_place
from hybrid_astar_utils import draw_car,check_car_collision, car_constraints,pi_2_pi, WB, Node, analytic_expansion,STEER_ANGLES,plot_arrow

cell_size_orig = 1 # cell size aka each pixel size in m
cell_size_curr = 0.2 # cell size we are going to use the algorithm for
movement_step =1.01
animate =False
debug = False
cell_segment = 2 # divide each grid by 
STEERING_PINALTY =200
HEURISTICS1_PINALTY=10
HEURISTICSASTAR_PINALTY = 3000 # because the astar error is too low we have to put high threshold to avoid our path 100 % folliwing the a star path
start=(39, 26., np.deg2rad(-19.0))
end=(19, 47., np.deg2rad(-90.0))
steps2expand = 10 # apply dubins curve to each steps2expand steps after finding the result # takes around 0.01 sec to accomplish
MAX_ITERATIONS = 10000
YAW_RESOLUTION=5 
YAW_SEGMENT = 360/YAW_RESOLUTION


def check_collision(node:Node, map)->bool:
    if(node.x < 0 or node.x >= map.shape[0]):
        return True
    if(node.y < 0 or node.y >= map.shape[1]):
        return True
    if(map[int(node.x), int(node.y),0] < 100):
        return True
    return False
    
def move(x,y,yaw, distance, steer, L=WB):
    dx = distance * math.cos(yaw)
    dy =  distance * math.sin(yaw)
    dyaw = pi_2_pi(distance * math.tan(steer) / L) 
    return x + dx, y + dy, yaw + dyaw


def movements_hybrid(x,y,yaw):
    movement = []
    for i in range(len(STEER_ANGLES)):
        # print(x,y,yaw, '--')
        dx,dy,dyaw = move(x,y,yaw, movement_step, STEER_ANGLES[i])
        # print(x,y,yaw,dx,dy,dyaw, '++')
        # dx = 0.7 * dx + 0.3 * x
        # dy = 0.7 * dy + 0.3 * y
        # dyaw = 0.7 * dyaw + 0.3 * yaw
        
        dcost = movement_step + STEERING_PINALTY *( abs(yaw-dyaw) ) 

        movement.append([dx,dy,dyaw, dcost])
    return movement

def calculate_index(node, shape):
    x_id = int(node.x * cell_segment)
    y_id = int(node.y * cell_segment)
    yaw_id = int(pi_2_pi(node.yaw) / (2 * np.pi/ YAW_SEGMENT))
    return yaw_id * shape[0] * shape[1]   +  x_id * shape[1] + y_id

     
def reconstruct_map(map_img):
    w,h,_ = map_img.shape
    wn,hn = int(w * cell_size_orig/cell_size_curr),int(h * cell_size_orig/cell_size_curr)
    map_new = cv2.resize(map_img,(wn, hn), interpolation = cv2.INTER_NEAREST)
    return map_new


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
    return dist#path[np.argmin(dist)][-1]

def hybrid_astar(map_new, start, end):
    
    visited = {}
    queue = {}
    snode =Node(start[0],start[1], start[2])
    snode.calculate_heuristics(end)
    snode.g = 0.0
    snode.f = snode.g + snode.h
    if(check_collision(snode.new(end),map_new)):
        print("target is in collision mode searching new target")
        return None, None,None
        # end_ = astar_heap_find_open_place(map_new,np.int32(end[:-1]),np.int32(start[:-1]))
        # end = [end_[0], end_[1], end_[2]]
    goal = snode.new(end)
    
    # initial astar search 
    astar_path = astar_heap(map_new, np.int32(start[:-1]), np.int32(end[:-1]))
    if(astar_path is None):
        print("astar could not find the path")
        return None, None, None
    if(debug):
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
    
        
        if(debug): 
            if(animate):
                plt.waitforbuttonpress(0.01)
    
        res_xs, res_ys, res_yaws = analytic_expansion(curr, goal, map_new)    
        if(res_xs is not None and res_ys is not None and res_yaws is not None or steps >= MAX_ITERATIONS): 
            res = [] 
            
            while(True):
                res.append([curr.x, curr.y, curr.yaw])
                curr = curr.prev
                if(curr is None):
                    break
            res.reverse()        
            rxs,rys,ryaws = [], [], []
            for i in np.arange(0, len(res), steps2expand):                
                ind_end = min(i + steps2expand, len(res)-1)
                res_xs_, res_ys_, res_yaws_ = analytic_expansion(goal.new(res[i]), goal.new(res[ind_end]), map_new)    
                
                if(res_xs_ is not None and res_ys_ is not None and res_yaws_ is not None ):
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
                                HEURISTICSASTAR_PINALTY*heuristics_astar
                               
            ind_tmp = calculate_index(adjacentnode, map_new.shape)
            if(not (ind_tmp in visited or ind_tmp in queue) ):
               
                if(not check_collision(adjacentnode, map_new)):
                    adjacentnode.f +=  50 * (255-map_new[int(adjacentnode.x), int(adjacentnode.y), 0])
                    if(debug):  
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
                        if(animate):
                            plt.text(int(adjacentnode.y * cell_segment)/cell_segment, 
                                      int(adjacentnode.x * cell_segment)/cell_segment, 
                                      str(round(adjacentnode.f)), fontsize=8)
                    queue[ind_tmp] = adjacentnode
                    heapq.heappush(heap, (adjacentnode.f, ind_tmp))
                    
        # add the current to visited
        visited[in_curr] = curr
        
    print("end returning none")
    return None, None, None


if __name__ == "__main__":
    map_path = "/home/essys/catkin_ws/src/path_planner/maps/map_basic.png"
    map_img = cv2.imread(map_path, -1)
    map_new = reconstruct_map(map_img)
    map_in = map_new
    import time 
    
    print("input map shape", map_in.shape)
    map_in = cv2.blur(map_in, (3, 3))   
    for i in range(1):
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
        for i in range(len(res_ys)):
            draw_car(res_xs[i], res_ys[i], res_yaws[i])
        plt.plot(res_ys, res_xs, "b-",markersize=1,  label="Hybrid A* path")
    else:
        print("hybrid start could not find the object")
    plt.grid(True)
    plt.axis("equal")
    plt.imshow(map_in[:,:,0])
    plt.plot()
    
    plt.legend()
    plt.show()
