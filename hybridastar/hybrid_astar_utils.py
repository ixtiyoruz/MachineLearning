#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:34:52 2021

@author: essys
"""
from scipy.spatial.transform import Rotation as Rot
import math
import numpy as np
import matplotlib.pyplot as plt
from dubins import dubins_heuristics, dubins_path_planning_costmap
WB = 2 # rear to front wheel
DUBINS_PINALTY=10
radius = 0.539606813
curvature = 1./radius
STEER_ANGLES = [-36*np.pi/180, 0, 36 * np.pi/180]
DUBIN_STEP = 0.9

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
    
def draw_car(y,x, yaw, length=WB):    
    car_outline_x, car_outline_y,c,s = car_constraints(x, y, yaw)
    arrow_x, arrow_y, arrow_yaw = c * 0.1 + y, s * 0.1 + x, yaw
    plot_arrow(arrow_y, arrow_x, arrow_yaw, length=0.5, width=0.2)
    car_color = '-k'
    
    # print("car_outline_shape\n", np.stack([car_outline_x, car_outline_y]).T)
    plt.plot(car_outline_x, car_outline_y, car_color)
    

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



def check_car_collision(node:Node, costmap)->bool:
    # we should get the 4 corners of the car 
    car_outline_x, car_outline_y,c,s = car_constraints(node.x, node.y, node.yaw)
    car_outline_x = np.int32(car_outline_x)
    car_outline_y = np.int32(car_outline_y)
    # print(np.any((car_outline_x < 0) |(car_outline_x >= costmap.shape[1]) ))
    if(np.any((car_outline_x < 0) | (car_outline_x >= costmap.shape[0]) | (car_outline_y < 0) | (car_outline_y >= costmap.shape[1]))):
        return True
    if(np.any(costmap[car_outline_x, car_outline_y] < 100)):
        return True
    return False



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
                                                                   curvature, DUBIN_STEP, costmap)   
        
    if(len(x_list) == 0):
        return  None, None, None
    return  x_list, y_list, yaw_list


def car_constraints(x, y, yaw):
    WB = 0.8  # rear to front wheel
    W = 0.7  # width of car
    LF = 0.2  # distance from rear to vehicle front end
    LB = 0.2  # distance from rear to vehicle back end
    # vehicle rectangle vertices
    VRX = [LF, LF, -LB, -LB, LF]
    VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

    
    c, s = math.cos(yaw), math.sin(yaw)
    rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)
    return car_outline_x, car_outline_y, c, s
def pi_2_pi(angle):
    """
    this turns the angle between -p to p
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

