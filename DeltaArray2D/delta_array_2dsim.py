import numpy as np
import cv2
import matplotlib.pyplot as plt

import pygame
import pymunk
from pymunk.vec2d import Vec2d

class DeltaArray2DSim:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.iterations = 10
        static_body = self.space.static_body

        self.object = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.space.add(self.object)
        self.object.position = 540, 960
        self.object.angle = 0.0
        shape = pymunk.Poly.create_box(self.object, (278, 137), 0.0)
        shape.mass = 0.113
        shape.friction = 0.4
        self.space.add(shape)

        self.robot_positions = np.zeros((8,8,2))
        for i in range(8):
            for j in range(8):
                if i%2!=0:
                    finger_pos = np.array((i*37.5, j*43.301 - 21.65))
                else:
                    finger_pos = np.array((i*37.5, j*43.301))

                finger_pos[0] = (finger_pos[0] - plane_size[0][0])/(plane_size[1][0]-plane_size[0][0])*1080 - 0
                finger_pos[1] = 1920 - (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920
                finger_pos = finger_pos.astype(np.int32)
                

                self.robot_positions[i,j] = finger_pos


        
    def update(self, space, dt, surface):
