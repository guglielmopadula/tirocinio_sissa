#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 21:27:23 2022

@author: cyberguli
"""

from stl import mesh
import stl
file="square.stl"
your_mesh = mesh.Mesh.from_file(file)
M=your_mesh.vectors
import numpy as np
a=np.array([])
for i in range(len(M)):
    if M[i,:,2].any()==np.zeros(3).any():
        a=np.append(a,M[i]).reshape(-1,3,3)
        
        
        
def area_triangle(a,b,c):
    M=np.zeros([3,3])
    M[0:2,0]=a
    M[0:2,1]=b
    M[0:2,2]=c
    M[2,:]=1
    return abs(np.linalg.det(M))/2

def area(mesh):
    area=0
    for i in range(len(mesh)):
        area=area+area_triangle(mesh[i,0,0:2],mesh[i,1,0:2],mesh[i,2,0:2])
    return area

print(area(a))
M=M/area(a)**(1/2)
data = np.zeros(len(M), dtype=mesh.Mesh.dtype)
data['vectors'] = M
mymesh = mesh.Mesh(data.copy())
mymesh.save(file, mode=stl.Mode.ASCII)
