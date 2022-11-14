#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:36:12 2022

@author: cyberguli
"""
import meshio
from scipy.integrate import solve_ivp
from sympy.vector import Vector, CoordSys3D
from sympy.matrices import Matrix,ImmutableDenseMatrix
from sympy import *
import numpy as np
import scipy
import math
import sympy
np.random.seed(0)
k=30
mesh=meshio.read("hullpreprocessed.stl")
M=mesh.points.copy()
newmesh_indices=np.arange(len(mesh.points))[mesh.points[:,2]>0].tolist()
triangles=mesh.cells_dict['triangle'].astype(np.int64)
newtriangles=[]
for T in triangles:
    if T[0] in newmesh_indices and T[1] in newmesh_indices and T[2] in newmesh_indices:
        newtriangles.append([newmesh_indices.index(T[0]),newmesh_indices.index(T[1]),newmesh_indices.index(T[2])])

newtriangles=np.array(newtriangles).astype(np.int64)
newmesh=mesh.points[newmesh_indices]

def vec_def(alpha,beta,a,c):
    def fun(t,x):
        u=np.array([np.sin(alpha)*np.cos(beta),np.cos(alpha)*np.cos(beta),np.sin(beta)])
        w=np.array([np.cos(alpha),np.sin(alpha),0])
        return (u.dot(w)+np.cross(a,np.cross(-(2*a),np.cross(a,x-c))))*(x[2]>0).astype(int)*(x[1]>0).astype(int) 
    return fun


def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume



        
        
        
def deform_stl(mesh_orig,mesh_def,newmesh_indices,index):
    newmesh=mesh_orig.points.copy()
    triangles=mesh.cells_dict['triangle'].astype(np.int64)
    newmesh[newmesh_indices]=mesh_def
    meshio.write_points_cells('hull_{}.stl'.format(index), newmesh, [("triangle", triangles)])

    
    


t=[0,1]
defmesh=newmesh.copy()
for j in range(500):
    print(j)
    alpha=2*np.pi*np.random.rand()
    beta=2*np.pi*np.random.rand()
    a=np.random.randn(3)
    c=np.random.randn(3)
    v=vec_def(alpha,beta,a,c)
    for i in range(len(newmesh)):
        defmesh[i]=solve_ivp(v,t,newmesh[i].astype(np.single),dense_output=True).sol(1)
    deform_stl(mesh,defmesh,newmesh_indices,j)
    

meshio.write_points_cells('test.stl', defmesh, [("triangle", newtriangles)])
    
    
    
    