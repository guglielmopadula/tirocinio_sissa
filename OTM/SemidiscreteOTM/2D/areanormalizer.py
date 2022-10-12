#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.interpolate import lagrange
import sympy
from scipy.spatial import Delaunay
import itertools
import meshio
import sys
name=sys.argv[1]

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    triangles=mesh.cells_dict['triangle']
    return points,triangles

def area_triangle(a,b,c):
    M=np.zeros([3,3])
    M[0:2,0]=a
    M[0:2,1]=b
    M[0:2,2]=c
    M[2,:]=1
    return abs(np.linalg.det(M))/2

def area(mesh,F):
    area=0
    for i in range(F.shape[0]):
        area=area+area_triangle(mesh[F[i,0]],mesh[F[i,1]],mesh[F[i,2]])
    return area

points,triangle=getinfo(name)
indices=np.arange(len(points))[points[:,2]==0].tolist()

points_zero=points[points[:,2]==0,:][:,0:2]
triangles_zero=[x for x in triangle if points[x[0]][2]==0 and points[x[1]][2]==0 and points[x[2]][2]==0]
triangles_zero_reformat=[]


for x in triangles_zero:
    triangles_zero_reformat.append([indices.index(x[0]),indices.index(x[1]),indices.index(x[2])])

triangles_zero_reformat=np.array(triangles_zero_reformat)
A=area(points_zero,triangles_zero_reformat)
points[:,0]=points[:,0]/(A)**2
points[:,1]=points[:,1]/(A)**2
meshio.write_points_cells(name,points.tolist(),[("triangle", triangle)])


