#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:38:18 2022

@author: cyberguli
"""

import scipy
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import binom
np.random.seed(0)
import meshio


def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(points,triangles):
    mesh=points[triangles]
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume



def MultiBSpline(t1,t2,t3):
        spline1=BSpline.basis_element(t1, False)
        spline2=BSpline.basis_element(t2, False)
        spline3=BSpline.basis_element(t3, False)
        def function(x):
            return np.nan_to_num(spline1(x[0]))*np.nan_to_num(spline2(x[1]))*np.nan_to_num(spline3(x[2]))
        return function
        


class FFD():
    def __init__(self,box_origin,box_length,n_control,k=5,n=10):
        self.box_origin=np.array(box_origin)
        self.box_length=np.array(box_length)
        self.n_control=np.array(n_control, dtype=np.int64)
        self.n=n
        self.k=k
        self.generate_splines()
        self.control_points=np.zeros([n_control[0],n_control[1],n_control[2],3])
        for i in range(n_control[0]):
            for j in range(n_control[1]):
                for k in range(n_control[2]):
                    self.control_points[i,j,k]=1/(self.n_control-1)*np.array([i,j,k])
        
    
    def apply_to_point(self,x):
        y=0
        x=(x-self.box_origin)/self.box_length
        for i in range(self.n_control[0]):
            for j in range(self.n_control[1]):
                for k in range(self.n_control[2]):
                    y=y+self.control_points[i,j,k]*self.splines[i,j,k](x)
        return (y*self.box_length)+self.box_origin
        
    def apply_to_mesh(self,mesh_points):
        new_mesh=np.zeros(mesh_points.shape)
        for i in range(len(mesh_points)):
            if self.check_point_in_box(mesh_points[i]):
                new_mesh[i]=self.apply_to_point(mesh_points[i])
            else:
                new_mesh[i]=mesh_points[i]
        return new_mesh
    

    def apply_to_mesh_pres(self,mesh_points,triangles):
        deftriangles=[]
        for T in triangles:
            if self.check_point_in_box(mesh_points[T[0]]) and self.check_point_in_box(mesh_points[T[1]]) and self.check_point_in_box(mesh_points[T[2]]):
                deftriangles.append(T)
        alpha=np.zeros(self.n_control)
        beta=np.zeros(self.n_control)
        gamma=np.zeros(self.n_control)
        Vref=volume(mesh_points,triangles)
        new_mesh=self.apply_to_mesh(mesh_points)
        mesh_points=(mesh_points-self.box_origin)/self.box_length
        VM=volume(new_mesh, triangles)
        a=VM+1/3*(Vref-VM)
        new_mesh=self.apply_to_mesh(mesh_points)
        for i in range(self.n_control[0]):
            for j in range(self.n_control[1]):
                for k in range(self.n_control[2]):
                    print("x ",i," ",j," ",k)
                    for T in deftriangles:
                        print((((self.apply_to_point(mesh_points[T[1]])[1]-self.apply_to_point(mesh_points[T[0]])[1])*(self.apply_to_point(mesh_points[T[2]])[2]-self.apply_to_point(mesh_points[T[0]])[2])-(self.apply_to_point(mesh_points[T[2]])[1]-self.apply_to_point(mesh_points[T[0]])[1])*(self.apply_to_point(mesh_points[T[1]])[2]-self.apply_to_point(mesh_points[T[0]])[2]))))
                        alpha[i,j,k]=alpha[i,j,k]+(self.splines[i,j,k](mesh_points[T[0]])+self.splines[i,j,k](mesh_points[T[1]])+self.splines[i,j,k](mesh_points[T[2]]))/6*((self.apply_to_point(mesh_points[T[1]])[1]-self.apply_to_point(mesh_points[T[0]])[1])*(self.apply_to_point(mesh_points[T[2]])[2]-self.apply_to_point(mesh_points[T[0]])[2])-(self.apply_to_point(mesh_points[T[2]])[1]-self.apply_to_point(mesh_points[T[0]])[1])*(self.apply_to_point(mesh_points[T[1]])[2]-self.apply_to_point(mesh_points[T[0]])[2]))
        print(alpha)
        delta_x=a/np.sum(alpha**2)
        self.control_points=self.control_points+delta_x
        mesh_points=mesh_points*self.box_length+self.box_origin
        new_mesh=self.apply_to_mesh(mesh_points)
        mesh_points=(mesh_points-self.box_origin)/self.box_length
        for i in range(self.n_control[0]):
            for j in range(self.n_control[1]):
                for k in range(self.n_control[2]):
                    print("y ",i," ",j," ",k)
                    for T in triangles:
                        beta[i,j,k]=beta[i,j,k]+(self.splines[i,j,k](mesh_points[T[0]])+self.splines[i,j,k](mesh_points[T[1]])+self.splines[i,j,k](mesh_points[T[2]]))/6*((self.apply_to_point(mesh_points[T[1]])[2]-self.apply_to_point(mesh_points[T[0]])[2])*(self.apply_to_point(mesh_points[T[2]])[0]-self.apply_to_point(mesh_points[T[0]])[0])-(self.apply_to_point(mesh_points[T[2]])[2]-self.apply_to_point(mesh_points[T[0]])[2])*(self.apply_to_point(mesh_points[T[1]])[0]-self.apply_to_point(mesh_points[T[0]])[0]))
        delta_y=a/np.sum(beta**2)
        self.control_points=self.control_points+delta_y
        mesh_points=mesh_points*self.box_length+self.box_origin
        new_mesh=self.apply_to_mesh(mesh_points)
        mesh_points=(mesh_points-self.box_origin)/self.box_length
        for i in range(self.n_control[0]):
            for j in range(self.n_control[1]):
                for k in range(self.n_control[2]):
                    print("z ",i," ",j," ",k)
                    for T in triangles:
                        gamma[i,j,k]=gamma[i,j,k]+(self.splines[i,j,k](mesh_points[T[0]])+self.splines[i,j,k](mesh_points[T[1]])+self.splines[i,j,k](mesh_points[T[2]]))/6*((self.apply_to_point(mesh_points[T[1]])[0]-self.apply_to_point(mesh_points[T[0]])[0])*(self.apply_to_point(mesh_points[T[2]])[1]-self.apply_to_point(mesh_points[T[0]])[0])-(self.apply_to_point(mesh_points[T[2]])[0]-self.apply_to_point(mesh_points[T[0]])[0])*(self.apply_to_point(mesh_points[T[1]])[1]-self.apply_to_point(mesh_points[T[0]])[1]))
        delta_z=a/np.sum(gamma**2)
        self.control_points=self.control_points+delta_z
        mesh_points=mesh_points*self.box_length+self.box_origin
        new_mesh=self.apply_to_mesh(mesh_points)
        return new_mesh

        
        
    def generate_splines(self):
        maxcontrol=np.max(self.n_control)
        c=np.zeros((maxcontrol,self.n))
        for i in range(maxcontrol):
            p=np.random.rand(1)
            c[i]=binom.pmf(range(self.n),self.n-1,p)
        t=np.random.rand(self.n+self.k+1)
        lst=[]
        t.sort()
        t=t-t[0]
        t=t/t[-1]
        for i in range(maxcontrol):
            for j in range(maxcontrol):
                for k in range(maxcontrol):
                    lst.append(MultiBSpline(t[i:i+self.k+2],t[j:j+self.k+2],t[k:self.k+2]))
    
        self.splines=np.array(lst).reshape((maxcontrol,maxcontrol,maxcontrol))
        
    def check_point_in_box(self,x):
        return (x[0]< self.box_origin[0]+self.box_length[0]) and (x[0]>self.box_origin[0]) and (x[1]< self.box_origin[1]+self.box_length[1]) and (x[1]>self.box_origin[1]) and  (x[2]< self.box_origin[2]+self.box_length[2]) and (x[2]>self.box_origin[2])

mesh=meshio.read("DTMB_per_giovanni_front.stl")
M=mesh.points
triangles=mesh.cells_dict['triangle'].astype(np.int64)
ffd=FFD([-2.7, 0, -0.4],[0.7, 0.15, 0.5],[3, 3, 3])
new_mesh=ffd.apply_to_mesh_pres(M,triangles)
meshio.write_points_cells('test.stl',new_mesh,[("triangle", triangles)])
