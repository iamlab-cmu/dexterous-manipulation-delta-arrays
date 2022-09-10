#!/usr/bin/env python

import time
import serial
import math
import numpy as np
import ipdb

PI = math.pi

class Prismatic_Delta:
    def __init__(self, s_p, s_b, l):
        # p: platform
        # b: base
        # u: distance to vertex from center of triangle (base or end effector)
        # w: distance to side from center of triangle
        # s: side length
        # L: upper legs (attached to base)
        # l: lower legs (attached to platform)

        self.s_p = s_p
        self.s_b = s_b

        self.l = l

        # geometry of equilateral triangle
        self.u_p = math.sqrt(3) * s_p / 3
        self.u_b = math.sqrt(3) * s_b / 3

        self.w_p = math.sqrt(3) * s_p / 6
        self.u_b = math.sqrt(3) * s_b / 3

        u_b = math.sqrt(3) * s_b / 3
        self.base1 = np.array([[0, u_b, 0]])
        a2 = -math.pi / 6
        self.base2 = np.array([[u_b * math.cos(a2), u_b * math.sin(a2), 0]])
        a3 = 7 * math.pi / 6
        self.base3 = np.array([[u_b * math.cos(a3), u_b * math.sin(a3), 0]])

    def get_base_verteces(self, offset, rotation):
        bv1 = np.transpose((rotation * np.transpose(self.base1))) + offset
        bv2 = np.transpose((rotation * np.transpose(self.base1))) + offset
        bv3 = np.transpose((rotation * np.transpose(self.base1))) + offset
        return [bv1, bv2, bv3]

    def get_knee_joints(self, heights, offset, rotation):
        [bv1, bv2, bv3] = self.get_base_verteces(offset, rotation)
        kj1 = bv1 + np.array([[0, 0, heights[0]]])
        kj2 = bv2 + np.array([[0, 0, heights[1]]])
        kj3 = bv3 + np.array([[0, 0, heights[2]]])
        return [kj1, kj2, kj3]

    def get_platform_verteces(self, pt, offset, rotation):
        u_p = self.u_p
        c1 = pt + np.array([[0, u_p, 0]])
        a2 = -math.pi / 6
        c2 = pt + np.array([[u_p * math.cos(a2), u_p * math.sin(a2), 0]])
        a3 = 7 * math.pi / 6
        c3 = pt + np.array([[u_p * math.cos(a3), u_p * math.sin(a3), 0]])

        pv1 = np.tranpose(rotation * np.transpose(c1)) + offset
        pv2 = np.tranpose(rotation * np.transpose(c1)) + offset
        pv3 = np.tranpose(rotation * np.transpose(c1)) + offset

        return [pv1, pv2, pv3]

    def validate_pts(self, pts, knee_joint_lim, par_joint_lim):
        # validate that joint limits are satisfied for each point in
        # trajectory and return corrected heights that are valid

        # A corrected point is one that is acheived with every knee
        # joint below the base platform, which is not a constraint for
        # FK and workspace sampling.  This is simply done by calling IK

        valid_pts = np.zeros((len(pts), 3))
        valid_heights = np.zeros((len(pts), 3))
        i = 0
        b1 = self.base1
        b2 = self.base2
        b3 = self.base3
        bases = np.concatenate((b1, b2, b3))

        for row in range(len(pts)):
            pt = pts[row]
            [pv1, pv2, pv3] = self.get_platform_verteces(pt, np.array([[0, 0, 0]]), np.identity(
                3))  # was rotz(0), but this is same as identity
            plat_verts = np.concatenate((pv1, pv2, pv3))

            floor_pt = np.concatenate((plat_verts[:, 0:2], np.zeros((3, 1))), axis=1)
            normr_bases = bases / np.linalg.norm(bases, axis=1,
                                                 keepdims=True)  # https://stackoverflow.com/questions/37914795/normalising-rows-in-numpy-matrix
            normr_floor_pt = (floor_pt - bases) / np.linalg.norm((floor_pt - bases), axis=1, keepdims=True)
            par_angles = math.asin(
                math.sqrt(sum(np.power(np.cross(normr_bases, normr_floor_pt), 2))))  # sum of each row
            if sum(par_angles > par_joint_lim) == 0:
                heights = self.IK(pt)
                if sum(sum(heights < 0)) > 0:
                    continue
                knee_angles = math.pi / 2 - math.asin(
                    (np.ones((3, 1)) * pt[2] - np.transpose(heights)) / self.l)  # FIXME
                if sum(knee_angles > knee_joint_lim) == 0:
                    valid_heights[i] = heights
                    valid_pts[i] = pt
                    i = i + 1

        valid_pts = valid_pts[0:i + 1]
        valid_heights = valid_heights[0:i + 1]

        return [valid_heights, valid_pts]

    def is_valid(self, pt, max_height):
        min_height = 0
        h = self.IK(pt)

        # not imaginary and not above max height
        is_valid = sum(h < min_height) + sum(h > max_height) == 0 and sum(h == abs(h)) == 3
        return is_valid

    def IK(self, position):
        # find corners of ee platform
        u_p = self.u_p
        c1 = position + np.array([[0, u_p, 0]])
        a2 = -math.pi / 6
        c2 = position + np.array([[u_p * math.cos(a2), u_p * math.sin(a2), 0]])
        a3 = 7 * math.pi / 6
        c3 = position + np.array([[u_p * math.cos(a3), u_p * math.sin(a3), 0]])

        # squared dist to prismatic actuator axis
        d1 = np.sum(np.power(c1[0][0:2] - self.base1[0][0:2], 2))
        d2 = np.sum(np.power(c2[0][0:2] - self.base2[0][0:2], 2))
        d3 = np.sum(np.power(c3[0][0:2] - self.base3[0][0:2], 2))

        l = self.l ** 2
        if d1 > l or d2 > l or d3 > l:
            return [-42., -42., -42.]
        h1 = position[2] - math.sqrt(l - d1)
        h2 = position[2] - math.sqrt(l - d2)
        h3 = position[2] - math.sqrt(l - d3)

        heights = [h1, h2, h3]
        return heights

    def IK_Traj(self, trajectory):
        traj = np.zeros((len(trajectory), 3))
        for i in range(len(trajectory)):
            traj[i] = self.IK(trajectory[i])

        return traj

    def bound_workspace(self, max_height):
        height_sample = np.arange(0, max_height, 0.3)
        pts = np.zeros((height_sample.shape[0]**3, 3))
        index = -1
        for h1 in height_sample:
            for h2 in height_sample:
                for h3 in height_sample:
                    test_pt = self.FK([h1,h2,h3])
                    if(np.sum(np.isnan(test_pt)) == 0):
                        index = index + 1
                        pts[index,:] = test_pt
                        # print('sum of NaN loop')
        pts = pts[0:index+1,:]
        # ipdb.set_trace()
        return pts

    def bound_constrained_workspace(self, z_vals, max_height):
        height_sample = np.arange(0,max_height,0.1)
        if z_vals[1] < 0:
            h1s = height_sample
        else:
            h1s = z_vals[1]
        if z_vals[2] < 0:
            h2s = height_sample
        else:
            h2s = z_vals[2]
        if z_vals[3] < 0:
            h3s = height_sample
        else:
            h3s = z_vals[3]
        pts = np.zeros((height_sample.shape[0]**3,3))
        index = -1
        for h1 in h1s:
            for h2 in h2s:
                for h3 in h3s:
                    test_pt = self.FK(np.array([h1,h2,h3]))
                    if(np.sum(test_pt) == 0):
                        index = index + 1
                        pts[index,:] = test_pt

        pts = pts[0:index,:]

        return pts


    def FK(self, heights):

        c1 = self.base1 + np.array([0,0,heights[0]])
        c2 = self.base2 + np.array([0,0,heights[1]])
        c3 = self.base3 + np.array([0,0,heights[2]])

        # shift sphere centers by platform offset
        u_p = self.u_p
        ic1 = c1 - np.array([0,u_p,0])
        a2 = -PI/6
        ic2 = c2 - np.array([u_p*math.cos(a2),u_p*math.sin(a2),0])
        a3 = 7*PI/6
        ic3 = c3 - np.array([u_p*math.cos(a3),u_p*math.sin(a3),0])

        l = self.l
        position = self.interx(ic1,ic2,ic3,l,l,l,1)
        position = np.transpose(position[0:3])

        return position

    def FK_Traj(self, heights):
        # FK on trajectory
        traj = np.zeros((heights.shape[0],3))
        for i in np.arange(heights.shape[0]):
            traj[i,:] = self.FK(heights[i,:])

        return traj

    def find_workspace_shape(self, max_height):
        step = .1
        Z = np.arange(0, max_height,step) + self.l
        max_rad = 0.0
        max_rad_z = 0.0
        low_z = max_height + self.l
        high_z = 0.0
        for z in Z:
            if self.is_valid(np.array([0,0,z]),max_height):
                if low_z > z:
                    low_z = z
                if high_z < z:
                    high_z = z

        x = 0.0
        y = max_rad
        pt = np.array([x,y,z])
        if self.is_valid(pt,max_height):
            while True:
                y = y + step
                pt = np.array([x,y,z])
                if self.is_valid(pt,max_height):
                    max_rad_z = z
                    max_rad = y
                else:
                    break

        return np.array([low_z,high_z,max_rad_z,max_rad])

    def find_workspace_slice(self, z, step, max_height):

        # returns workspace slice for a fixed z
        directions = np.arange(0, 2*PI, step)
        border_pts = np.zeros(math.ceil((self.l * 2.0 / step) ** 2, 3))
        index = 0
        for deg in directions:
            pt = np.array([0, 0, z])
        while True:
            if self.is_valid(pt, max_height):
                index = index + 1
                slice_pts[index,:] = pt
            else:
                break
            pt = pt + step * [math.cos(deg), math.sin(deg), 0]

        slice_pts = slice_pts[0:index,:]

        return slice_pts

    def find_workspace_edge(self, z, step, max_height):
        # returns points on the border of the workspace for a fixed value of z
        # step is the rotation in radians to the next point
        directions = np.arange(0, 2*PI,step)
        border_pts = zeros(size(directions, 2), 3)
        index = 0
        for deg in directions:
            pt = np.array([0, 0, z])
        while True:
            next_pt = pt + step * np.array([math.cos(deg), math.sin(deg), 0])
            if not self.is_valid(next_pt, max_height):
                index = index + 1
                border_pts[index,:] = pt
                break
            pt = next_pt
        return border_pts

    def interx(self, X1=None,X2=None,X3=None,r1=None,r2=None,r3=None,pos=None):
        if X1 is None or X2 is None or X3 is None or r1 is None or r2 is None or r3 is None or pos is None:
            pos = 1 # default value
        # import ipdb
        # ipdb.set_trace()
        x1 = X1[0,0]
        y1 = X1[0,1]
        z1 = X1[0,2]
        x2 = X2[0,0]
        y2 = X2[0,1]
        z2 = X2[0,2]
        x3 = X3[0,0]
        y3 = X3[0,1]
        z3 = X3[0,2]
        # convert in coord sys at [x1 y1 z1] oriented same as global
        # x2 = 1
        # y2 = 1
        # z2 = 0.2
        # x3 = 2
        # y3 = 1
        # z3 = 1
        # r1 = 2.5
        # r2 = 2.6
        # r3 = 2.7 # TEST VALUES

        x2 = x2 - x1
        y2 = y2 - y1
        z2 = z2 - z1
        x3 = x3 - x1
        y3 = y3 - y1
        z3 = z3 - z1

        a=(16.0*y2**2*z3*y3**2*z2*x3*r1**2*x2-4*y2**3*z3*y3*z2*x3*r1**2*x2+4*y2**3*z3*y3*z2*x3*x2*r3**2-4*y2*y3**3*z2*x2*z3*r1**2*x3+4*y2*y3**3*z2*x2*z3*r2**2*x3+16*z2*x3**2*x2**2*r1**2*y3*y2*z3-4*z2*x3**3*x2*r1**2*y3*y2*z3+4*z2*x3**3*x2*y2*z3*r2**2*y3-4*x2**3*z3*x3*y2*r1**2*z2*y3+4*x2**3*z3*x3*y2*r3**2*z2*y3-4*y2*z3*z2**3*y3*x3*r1**2*x2+4*y2*z3*z2**3*y3*x3*x2*r3**2+8*y2*z3**2*z2**2*y3*x2*r1**2*x3-4*y2*z3**2*z2**2*y3*x2*r2**2*x3+4*x2**2*y3**2*z2*y2**2*z3*x3**2-4*x2**4*z3**2*x3**2*y3*y2-2*z2**2*x3**4*x2**2*y2**2+2*z2**2*x3**3*y2**2*r1**2*x2-2*z2**2*x3**3*y2**2*x2*r3**2+2*z2**2*x3**5*x2*y2*y3+2*x2**5*z3**2*x3*y3*y2+2*z2**3*y3**2*y2**2*z3*x3**2+2*z2**3*y3**2*x2**2*z3*x3**2+2*z2**3*y3**2*x2**2*z3*r1**2-2*z2**3*y3**2*x2**2*z3*r3**2+2*x2**2*y3**4*z2*y2**2*z3-4*x2**2*y3**2*z2**2*x3**2*y2**2+2*x2**4*y3**2*z2*z3*x3**2-2*x2**2*y3**3*z2**2*y2*x3**2+2*x2**2*y3**3*z2**2*y2*z3**2+2*x2**3*y3**2*z2**2*x3*z3**2+2*x2**2*y3**2*z2*y2**2*z3**3+2*x2**2*y3**3*z2**2*y2*r1**2+2*x2**2*y3**2*z2**2*x3**2*r2**2+2*x2**4*y3**2*z2*z3*r1**2-2*x2**4*y3**2*z2*z3*r3**2-2*x2**2*y3**3*z2**2*y2*r3**2+2*x2**3*y3**2*z2**2*x3*r1**2-2*x2**3*y3**2*z2**2*x3*r3**2+2*y2**4*z3*x3**2*y3**2*z2+2*y2**2*z3*x3**4*z2*x2**2-4*y2**2*z3**2*x3**2*x2**2*y3**2+2*y2**3*z3**2*x3**2*z2**2*y3+2*y2**2*z3**2*x3**3*x2*z2**2-2*y2**3*z3**2*x3**2*x2**2*y3+2*y2**3*z3**2*x3**2*r1**2*y3+2*y2**2*z3*x3**4*z2*r1**2-2*y2**2*z3*x3**4*z2*r2**2+2*y2**2*z3**2*x3**2*x2**2*r3**2-2*y2**3*z3**2*x3**2*r2**2*y3+2*y2**2*z3**2*x3**3*x2*r1**2-2*y2**2*z3**2*x3**3*x2*r2**2-2*y2**2*z3**2*y3**2*x2**3*x3-4*y2**4*z3**2*y3**2*x2*x3+2*y2**2*z3**2*y3**2*x2**2*r3**2+4*y2**3*z3**2*y3*x2**3*x3+2*y2**5*z3**2*y3*x2*x3+4*y2*y3**3*z2**2*x3**3*x2+2*y2*y3**5*z2**2*x3*x2-2*y2**2*y3**2*z2**2*x3**3*x2-4*y2**2*y3**4*z2**2*x3*x2+2*y2**2*y3**2*z2**2*x3**2*r2**2-4*z2**2*x3**4*x2**2*y2*y3+2*z2*x3**2*x2**2*y2**2*z3**3+2*z2*x3**2*y2**4*z3*r1**2+2*z2**2*x3**2*y2**3*r1**2*y3-2*z2*x3**2*y2**4*z3*r3**2-2*z2**2*x3**2*y2**3*r3**2*y3-z2**4*y3**2*x3**2*x2**2-z2**4*y3**2*x3**2*y2**2+2*z2**3*y3**4*x2**2*z3+2*z2**3*y3**2*x2**2*z3**3+2*x2**2*y3**5*z2**2*y2-2*x2**2*y3**4*z2**2*y2**2-2*x2**4*y3**2*z2**2*x3**2+2*x2**3*y3**2*z2**2*x3**3+2*x2**4*y3**4*z2*z3+2*x2**3*y3**4*z2**2*x3+2*x2**2*y3**4*z2**2*r2**2-2*y2**4*z3**2*x3**2*y3**2+2*y2**5*z3**2*x3**2*y3+2*y2**4*z3*x3**4*z2+2*y2**2*z3**2*x3**3*x2**3-2*y2**2*z3**2*x3**4*x2**2+2*y2**4*z3**2*x3**3*x2+2*y2**2*z3*x3**4*z2**3-y2**2*z3**4*x3**2*x2**2+2*y2**4*z3**2*x3**2*r3**2-2*y2**2*z3**2*y3**4*x2**2+2*y2**3*z3**2*y3**3*x2**2-y2**2*z3**4*y3**2*x2**2-y2**4*z3**2*y3**2*x2**2+2*y2**3*y3**3*z2**2*x3**2-y2**2*y3**4*z2**2*x3**2-2*y2**4*y3**2*z2**2*x3**2-z2**4*y3**4*x2**2-2*x2**4*y3**4*z2**2-2*y2**4*z3**2*x3**4-y2**4*z3**4*x3**2-2*z2**2*x3**4*y2**4-z2**4*x3**4*y2**2-2*x2**4*z3**2*y3**4-x2**4*z3**4*y3**2-z3**2*y2**6*x3**2-x3**6*z2**2*y2**2-z3**2*x2**6*y3**2-2*y2**4*x3**4*y3**2+2*y2**4*x3**4*r3**2+2*y2**5*x3**4*y3-y2**4*x3**2*r1**4-y2**4*x3**2*y3**4+2*y2**5*x3**2*y3**3-y2**4*x3**2*r3**4-y2**6*x3**2*y3**2-y2**2*x3**4*r1**4-y2**2*x3**4*x2**4+2*y2**2*x3**5*x2**3-y2**2*x3**4*r2**4-y2**2*x3**6*x2**2-2*y2**4*x3**4*x2**2+2*y2**4*x3**4*r2**2+2*y2**4*x3**5*x2+2*x2**4*y3**5*y2-2*x2**4*y3**4*y2**2+2*x2**4*y3**4*r2**2-x2**2*y3**4*r1**4-x2**2*y3**6*y2**2+2*x2**2*y3**5*y2**3-x2**2*y3**4*y2**4-x2**2*y3**4*r2**4-x2**4*y3**2*r1**4-x2**6*y3**2*x3**2+2*x2**5*y3**2*x3**3-x2**4*y3**2*x3**4-x2**4*y3**2*r3**4+2*x2**5*y3**4*x3-2*x2**4*y3**4*x3**2+2*x2**4*y3**4*r3**2-y3**6*z2**2*x2**2-y2**4*x3**6-y2**6*x3**4-x2**6*y3**4-x2**4*y3**6-2*z2*x3**2*r1**2*y2**2*z3*r3**2-2*z2*y3**2*r2**2*x2**2*z3*r1**2+2*z2*y3**2*r2**2*x2**2*z3*r3**2+2*x2**4*y3**2*z2*z3**3+2*y2*r1**4*z2**2*y3*x3*x2+2*x2**2*y3**2*z2*y2**2*z3*r1**2-8*x2**2*y3**3*z2*r1**2*y2*z3-2*x2**2*y3**2*z2*y2**2*z3*r3**2-8*x2**3*y3**2*z2*z3*r1**2*x3+2*y2**2*z3*x3**2*r1**2*y3**2*z2-8*y2**3*z3*x3**2*r1**2*z2*y3-2*y2**2*z3*x3**2*z2*y3**2*r2**2-8*y2**2*z3*x3**3*z2*r1**2*x2-4*y2**2*z3**2*y3**2*x2*z2**2*x3-4*y2**2*z3**2*y3**2*x2*r1**2*x3+4*y2**2*z3**2*y3**2*x2*r2**2*x3-4*y2**3*z3*y3*z2*x3**3*x2-4*y2**3*z3*y3**3*z2*x3*x2-4*y2**3*z3**3*y3*z2*x3*x2+4*y2**3*z3**2*y3*x2*z2**2*x3-4*y2**3*z3**2*y3*x2*r2**2*x3-4*y2*y3**3*z2*x2**3*z3*x3+4*y2*y3**3*z2**2*x3*x2*z3**2-4*y2*y3**3*z2**3*x2*z3*x3-4*y2*y3**3*z2**2*x3*x2*r3**2-4*y2**2*y3**2*z2**2*x3*r1**2*x2+4*y2**2*y3**2*z2**2*x3*x2*r3**2-4*z2**2*x3**2*x2**2*y2*z3**2*y3+2*z2*x3**2*x2**2*y2**2*z3*r1**2-4*z2**2*x3**2*x2**2*y2*r1**2*y3-2*z2*x3**2*x2**2*y2**2*z3*r3**2+4*z2**2*x3**2*x2**2*y2*r3**2*y3-4*z2**3*x3**3*x2*y2*z3*y3+4*z2**2*x3**3*x2*y2*z3**2*y3-4*z2*x3**3*x2**3*y3*y2*z3-4*z2**2*x3**3*x2*y2*r3**2*y3+4*x2**3*z3**2*x3*y2*z2**2*y3-4*x2**3*z3**3*x3*y2*z2*y3-4*x2**3*z3**2*x3*y2*r2**2*y3+2*x2**2*z3*x3**2*r1**2*y3**2*z2-4*x2**2*z3**2*x3**2*r1**2*y3*y2-2*x2**2*z3*x3**2*z2*y3**2*r2**2+4*x2**2*z3**2*x3**2*y2*r2**2*y3-4*y2*z3**3*z2**3*y3*x3*x2+2*y2*z3**2*z2**4*y3*x2*x3+2*y2*z3**4*z2**2*y3*x3*x2-2*r1**2*y3**2*z2*x2**2*z3*r3**2-2*y2**2*z3*r1**2*z2*x3**2*r2**2+2*r1**4*y3*y2*z3**2*x2*x3+2*z2**2*x3**5*y2**2*x2+2*z2**2*x3**4*y2**3*y3+2*z2*x3**2*y2**4*z3**3+2*z2**2*x3**4*y2**2*r2**2-z2**2*x3**4*x2**2*y3**2+2*x2**5*z3**2*x3*y3**2-x2**4*z3**2*x3**2*y2**2-2*x2**4*z3**2*x3**2*y3**2+2*x2**4*z3**2*y3**3*y2+2*x2**4*z3**2*y3**2*r3**2-2*y2**2*x3**4*z2**2*y3**2-2*z2**2*x3**2*x2**2*y3**4-2*x2**2*z3**2*y2**4*x3**2-2*x2**4*y3**2*y2**2*z3**2+2*y2**2*z3**3*z2**3*x3**2-z3**2*y2**2*r1**4*x3**2-z3**2*y2**2*z2**4*x3**2-z3**2*y2**2*r2**4*x3**2-2*z3**2*y2**4*x3**2*z2**2+2*z3**2*y2**4*x3**2*r2**2-2*x3**4*z2**2*y2**2*z3**2+2*x3**4*z2**2*y2**2*r3**2-x3**2*z2**2*y2**2*r1**4-x3**2*z2**2*y2**2*z3**4-x3**2*z2**2*y2**2*r3**4-2*z3**2*x2**4*y3**2*z2**2+2*z3**2*x2**4*y3**2*r2**2-z3**2*x2**2*r1**4*y3**2-z3**2*x2**2*z2**4*y3**2-z3**2*x2**2*r2**4*y3**2+2*y2**3*x3**4*r1**2*y3-2*y2**3*x3**4*x2**2*y3-2*y2**3*x3**4*r2**2*y3+2*y2**4*x3**3*r1**2*x2-2*y2**4*x3**3*x2*y3**2-2*z2**2*x3**2*x2**2*y3**2*z3**2+2*z2**2*x3**2*x2**2*y3**2*r3**2-2*x2**2*z3**2*y2**2*x3**2*z2**2+2*x2**2*z3**2*y2**2*x3**2*r2**2+2*x2**2*y3**2*y2**2*z3**2*r2**2+2*y2**2*z3**3*z2*x3**2*r1**2-2*y2**2*z3**3*z2*x3**2*r2**2+2*z2**3*x3**2*y2**2*z3*r1**2-2*z2**3*x3**2*y2**2*z3*r3**2+2*x2**2*z3**3*r1**2*y3**2*z2-2*x2**2*z3**3*z2*y3**2*r2**2+2*r1**4*y3**2*z2*x2**2*z3+2*y2**2*z3*r1**4*z2*x3**2+2*x2**2*z3*y3**4*r1**2*z2+2*x2**2*z3**2*y3**3*r1**2*y2-2*x2**2*z3*y3**4*z2*r2**2-2*x2**2*z3**2*y3**3*y2*r2**2+2*x2**3*z3**2*y3**2*r1**2*x3-2*x2**3*z3**2*y3**2*r2**2*x3-2*y2**2*z3**2*z2**2*y3**2*x2**2-2*y2**2*x3**2*z2**2*y3**2*z3**2+2*y2**2*x3**2*z2**2*y3**2*r3**2-4*y2*z3**2*z2**2*y3*x3*x2*r3**2-4*y2*z3**3*z2*y3*x2*r1**2*x3+4*y2*z3**3*z2*y3*x2*r2**2*x3-4*r1**4*y3*y2*z3*z2*x3*x2-4*r1**2*y3*y2*z3**2*x2*r2**2*x3-4*y2*r1**2*z2**2*y3*x3*x2*r3**2+4*r1**2*y3*y2*z3*z2*x3*x2*r3**2+4*y2*r1**2*z2*y3*x2*z3*r2**2*x3+2*z2*x3**2*r2**2*y2**2*z3*r3**2+2*y2*z3**2*r2**4*y3*x2*x3+2*y2*r3**4*z2**2*y3*x3*x2+4*y2**2*x3*x2*y3**2*r1**2*r3**2+4*y2**2*x3*x2*y3**2*r1**2*r2**2-4*y2**2*x3*x2*y3**2*r3**2*r2**2+4*y2*x3**2*x2**2*y3*r1**2*r2**2+4*y2*x3**2*x2**2*y3*r1**2*r3**2-4*y2*x3**2*x2**2*y3*r2**2*r3**2-4*y2**3*x3*x2*y3*z3**2*r3**2-4*y2*x3*x2*y3**3*r1**2*r2**2-4*y2**3*x3*x2*y3*r1**2*r3**2-4*y2*x3*x2*y3**3*z2**2*r2**2-4*y2*x3*x2**3*y3*r1**2*r3**2-4*y2*x3**3*x2*y3*r1**2*r2**2-4*y2*x3**3*x2*y3*z2**2*r2**2-4*y2*x3*x2**3*y3*z3**2*r3**2-4*z3**2*y2**2*r1**2*x3**2*z2**2+2*z3**2*y2**2*r1**2*x3**2*r2**2+2*z3**2*y2**2*z2**2*x3**2*r2**2+2*x3**2*z2**2*y2**2*z3**2*r3**2+2*x3**2*z2**2*y2**2*r1**2*r3**2-4*z3**2*x2**2*r1**2*y3**2*z2**2+2*z3**2*x2**2*r1**2*y3**2*r2**2+2*z3**2*x2**2*z2**2*y3**2*r2**2-2*y2**3*x3**2*r1**2*y3*r3**2-2*y2**3*x3**2*x2**2*y3*r1**2+2*y2**3*x3**2*x2**2*y3*r3**2-2*y2**3*x3**2*r1**2*r2**2*y3+2*y2**3*x3**2*r3**2*r2**2*y3-2*y2**2*x3**3*r1**2*x2*r2**2-2*y2**2*x3**3*r1**2*x2*y3**2-2*y2**2*x3**3*r1**2*x2*r3**2+2*y2**2*x3**3*r2**2*x2*y3**2+2*y2**2*x3**3*r2**2*x2*r3**2+2*y2**2*x3**2*r1**2*y3**2*r2**2+4*y2**2*x3**2*x2**2*y3**2*r2**2+2*y2**2*x3**2*r1**2*x2**2*r3**2+4*y2**2*x3**2*x2**2*y3**2*r3**2-8*y2**3*x3**3*r1**2*x2*y3-2*x2**2*y3**3*r1**2*y2*x3**2-2*x2**2*y3**3*r1**2*y2*r3**2-2*x2**2*y3**3*y2*r1**2*r2**2+2*x2**2*y3**3*y2*x3**2*r2**2+2*x2**2*y3**3*y2*r3**2*r2**2-2*x2**3*y3**2*r1**2*y2**2*x3-2*x2**3*y3**2*r1**2*r2**2*x3-2*x2**3*y3**2*r1**2*x3*r3**2+2*x2**3*y3**2*y2**2*x3*r3**2+2*x2**3*y3**2*r2**2*x3*r3**2+2*x2**2*y3**2*y2**2*r1**2*r3**2-4*y2*z3*r2**2*y3*z2*x3*x2*r3**2-4*x2**4*y3**2*r1**2*x3**2+2*x2**4*y3**2*r1**2*r3**2+2*x2**3*y3**2*r1**2*x3**3+2*x2**4*y3**2*x3**2*r2**2-2*x2**5*y3**2*x3*r3**2-2*x2**3*y3**2*r2**2*x3**3+2*x2**4*y3**2*x3**2*r3**2-y3**2*z2**2*r1**4*x2**2-y3**2*z2**2*x2**2*z3**4-y3**2*z2**2*x2**2*r3**4-2*y3**4*z2**2*x2**2*z3**2+2*y3**4*z2**2*x2**2*r3**2+4*y2**3*x3*x2**3*y3**3+4*y2**3*x3**3*x2*y3**3+2*y2*x3*x2**5*y3**3+2*y2**3*x3**5*x2*y3+2*y2**3*x3*x2*y3**5-4*y2**4*x3*x2*y3**4+2*y2**5*x3*x2*y3**3+2*y2*x3**3*x2**5*y3-4*y2*x3**4*x2**4*y3+2*y2**5*x3**3*x2*y3+2*y2*x3**5*x2**3*y3+2*y2*x3*x2**3*y3**5+4*y2**3*x3**3*x2**3*y3+4*y2*x3**3*x2**3*y3**3-2*y2**4*x3**3*x2*r3**2-2*y2**5*x3**2*r3**2*y3+2*y2**4*x3**2*y3**2*r2**2+2*y2**3*x3**2*r1**2*y3**3+2*y2**3*x3**2*r1**4*y3-4*y2**4*x3**2*r1**2*y3**2-3*y2**4*x3**2*x2**2*y3**2+2*y2**4*x3**2*r1**2*r3**2+2*y2**5*x3**2*r1**2*y3+2*y2**4*x3**2*y3**2*r3**2-2*y2**3*x3**2*y3**3*r2**2-y2**2*x3**2*r1**4*y3**2-3*y2**2*x3**2*x2**4*y3**2-y2**2*x3**2*r2**4*y3**2-y2**2*x3**2*r1**4*x2**2-3*y2**2*x3**2*x2**2*y3**4-y2**2*x3**2*x2**2*r3**4+2*y2**2*x3**3*r1**4*x2+2*y2**2*x3**3*r1**2*x2**3-4*y2**2*x3**4*r1**2*x2**2+2*y2**2*x3**4*r1**2*r2**2+2*y2**2*x3**5*r1**2*x2+2*y2**2*x3**4*x2**2*r2**2-2*y2**2*x3**3*x2**3*r3**2-2*y2**2*x3**5*r2**2*x2-3*y2**2*x3**4*x2**2*y3**2+2*y2**2*x3**4*x2**2*r3**2+2*x2**4*y3**3*y2*r1**2-2*x2**4*y3**3*y2*x3**2-2*x2**4*y3**3*y2*r3**2+2*x2**3*y3**4*r1**2*x3-2*x2**3*y3**4*y2**2*x3-2*x2**3*y3**4*r2**2*x3-2*x2**2*y3**3*y2**3*r3**2+2*x2**2*y3**4*y2**2*r2**2+2*x2**2*y3**5*r1**2*y2+2*x2**2*y3**3*r1**4*y2-4*x2**2*y3**4*r1**2*y2**2+2*x2**2*y3**4*r1**2*r2**2+2*x2**2*y3**3*y2**3*r1**2+2*x2**2*y3**4*y2**2*r3**2-2*x2**2*y3**5*y2*r2**2-x2**2*y3**2*y2**2*r1**4-x2**2*y3**2*y2**2*r3**4-x2**2*y3**2*r1**4*x3**2-x2**2*y3**2*r2**4*x3**2+2*x2**3*y3**2*r1**4*x3+2*x2**5*y3**2*r1**2*x3+2*x2**2*y3**2*r1**2*x3**2*r2**2-8*x2**3*y3**3*r1**2*y2*x3+2*y3**2*z2**2*r1**2*x2**2*r3**2+2*y3**2*z2**2*x2**2*z3**2*r3**2+4*y2**4*x3*x2*y3**2*r3**2+4*y2**3*x3*x2*y3**3*z2**2-4*y2**3*x3*x2*y3**3*r2**2-4*y2**2*x3*x2*y3**4*r1**2-4*y2**2*x3*x2*y3**2*r1**4+8*y2**3*x3*x2*y3**3*r1**2+4*y2*x3*x2**3*y3**3*z2**2-4*y2*x3*x2**3*y3**3*r2**2-4*y2**4*x3*x2*y3**2*r1**2+4*y2**3*x3**3*x2*y3*z3**2-4*y2**3*x3**3*x2*y3*r3**2+4*y2**3*x3*x2*y3**3*z3**2-4*y2**3*x3*x2*y3**3*r3**2+4*y2**2*x3*x2*y3**4*r2**2+2*y2*x3*x2*y3**3*r1**4+2*y2**3*x3*x2*y3*r1**4+2*y2**3*x3*x2*y3*z3**4+2*y2**3*x3*x2*y3*r3**4+2*y2*x3*x2*y3**3*z2**4+2*y2*x3*x2*y3**3*r2**4+2*y2*x3*x2**3*y3*r1**4+2*y2*x3**3*x2*y3*r1**4+2*y2*x3**3*x2*y3*z2**4+2*y2*x3**3*x2*y3*r2**4+2*y2*x3*x2**3*y3*z3**4+2*y2*x3*x2**3*y3*r3**4-4*y2*x3**2*x2**2*y3*r1**4-4*y2*x3**2*x2**4*y3*r1**2+8*y2*x3**3*x2**3*y3*r1**2-4*y2*x3**4*x2**2*y3*r1**2+4*y2*x3**3*x2**3*y3*z2**2-4*y2*x3**3*x2**3*y3*r2**2+4*y2*x3**2*x2**4*y3*r3**2+4*y2**3*x3**3*x2*y3*z2**2-4*y2**3*x3**3*x2*y3*r2**2+4*y2*x3**4*x2**2*y3*r2**2+4*y2*x3**3*x2**3*y3*z3**2-4*y2*x3**3*x2**3*y3*r3**2+4*y2*x3*x2**3*y3**3*z3**2-4*y2*x3*x2**3*y3**3*r3**2+16*y2**2*x3**2*x2**2*y3**2*r1**2)
        b=(-z2**3*y3**2-x2**2*y3**2*z2-y2**2*z3*x3**2-y2**2*z3*y3**2+y2**3*z3*y3+y2*y3**3*z2-y2**2*y3**2*z2-z2*x3**2*x2**2-z2*x3**2*y2**2+z2*x3**3*x2+x2**3*z3*x3-x2**2*z3*x3**2-x2**2*z3*y3**2+y2*z3*z2**2*y3+y2*x3**2*z2*y3+y2*z3**2*z2*y3+z2*x3*x2*y3**2+z2*x3*x2*z3**2+x2*z3*y2**2*x3+x2*z3*z2**2*x3+x2**2*y3*y2*z3-y2**2*z3**3-z2**3*x3**2-x2**2*z3**3-r1**2*y3**2*z2-y2**2*z3*r1**2+r1**2*y3*y2*z3+y2*r1**2*z2*y3+z2*y3**2*r2**2-z2*x3**2*r1**2+z2*x3**2*r2**2-x2**2*z3*r1**2+x2**2*z3*r3**2+y2**2*z3*r3**2-y2*z3*r2**2*y3-y2*r3**2*z2*y3+z2*x3*r1**2*x2-z2*x3*x2*r3**2+x2*z3*r1**2*x3-x2*z3*r2**2*x3)
        c=(-2*y2*z3*z2*y3-2*z2*x3*x2*z3+z3**2*y2**2+x3**2*z2**2+z3**2*x2**2+y2**2*x3**2+x2**2*y3**2+y3**2*z2**2-2*y2*x3*x2*y3)
        if a<0 or c==0:
            result = np.array([np.nan,np.nan,np.nan]).reshape((3,1))
            return result # coz c is the denominator and a is under a root
        #     error('Error in interx.m at z');
        za=-0.5*(b-a**(0.5))/c
        zb=-0.5*(b+a**(0.5))/c
        if za>zb:
            if(pos):
                z=za
            else:
                z=zb
        else:
            if(pos):
                z=zb
            else:
                z=za
        a=(2*z*z2*x3-2*x2*z*z3+r1**2*x2-r1**2*x3-x2**2*x3-y2**2*x3-z2**2*x3+r2**2*x3+x2*x3**2+x2*y3**2+x2*z3**2-x2*r3**2)
        b=(-2*y2*x3+2*x2*y3)
        if b==0:
            result = np.array([np.nan,np.nan,np.nan]).reshape((3,1))
            return result # coz b is the denominator in the expression
        y=a/b
        if x2 == 0:
            result = np.array([np.nan,np.nan,np.nan]).reshape((3,1))
            return result
        x = 0.5*(r1**2+x2**2-2*y*y2+y2**2-2*z*z2+z2**2-r2**2)/x2
        # convert result back to global
        result = np.array([[x1],[y1],[z1],[1]]) + np.array([[x],[y],[z],[0]])

        return result