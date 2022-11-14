#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:13:12 2022

@author: cyberguli
"""

import meshio
import numpy as np
import scipy
import sympy
np.random.seed(0)
k=30
mesh=meshio.read("hullrotated.stl")
M=mesh.points.copy()
triangles=mesh.cells_dict["triangle"].copy()
neigh_list=[]
for i in range(len(M)):
    s=set({})
    for T in triangles:
        if i in T:
            for j in T:
                if j!=i:
                    s.add(j)
    neigh_list.append(s)
    
start=np.argmin(np.abs(M[:,2]))
point=start
point_old=-1
next_point=-1
ring=[]
traversed=[]
while next_point!=start:
    ring.append(point)
    temp=neigh_list[point]-set(traversed+[point_old])
    temp=list(temp)
    next_point=temp[np.argmin(np.abs(M[temp,2]))]
    traversed.append(next_point)
    point_old=point
    point=next_point
    
    
M_new=M.copy()
M_new[ring,2]=0
meshio.write_points_cells('hullpreprocessed.stl', M_new, [("triangle", triangles)])