#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:33:37 2022

@author: cyberguli
"""

import triangle as tr
import trimesh
import numpy as np
import meshio 
from scipy.spatial import Delaunay
mesh=meshio.read("test_new.stl")
num_vertices=len(mesh.points)
points=mesh.points
indici=np.arange(num_vertices)[np.logical_or(points[:,0]==0,points[:,2]==0)]
print(indici)
listone= [set() for index in range(len(indici))]
faces=mesh.cells_dict["triangle"]
tot_indici=np.arange(num_vertices)
print(np.linalg.norm(points[22803]-points[22808]))



for face in faces:
    for i in indici:
        if i in face:
            for j in face:
                listone[np.where(indici==i)[0][0]].add(j)
                

    

tol=0.005
a=set({})
for i in indici:
    print(i)
    for j in listone[np.where(indici==i)[0][0]]:
        if j not in indici:
            if np.linalg.norm(points[i]-points[j])<tol:
                print("DELETED")
                a.add(j)





for m in range(10):
    flag=1
    k=0
    while flag!=0:
        index=indici[k]
        for j in indici:
            if j!=index:
                if np.linalg.norm(points[index]-points[j])<tol:
                    print("DELETED")
                    a.add(j)
                    indici=indici[indici!=j]
                    print(m)
        k=k+1
        if k>=len(indici):
            flag=0                


a=set({})
for i in indici:
    print(i)
    for j in range(num_vertices):
        if j not in indici:
            if np.linalg.norm(points[i]-points[j])<tol:
                print("DELETED")
                a.add(j)

    
            

true=list(set(np.arange(num_vertices).tolist()).difference(a)) 
newvertices=points[true]
meshio.write_points_cells("points_only.ply", newvertices,[])

'''
tri=Delaunay(newvertices,qhull_options="QJ")

def get_surface(tetra):
    envelope = set()
    for tet in tetra:
        for face in    ((tet[0], tet[1], tet[2]), 
                       (tet[0], tet[2], tet[3]), 
                       (tet[0], tet[3], tet[2]),
                       (tet[1], tet[3], tet[2]) ):
            if face in envelope:    
                envelope.remove(face)
            else:                  
                envelope.add((face[2], face[1], face[0]))
    
    # there is now only faces encountered once (or an odd number of times for paradoxical meshes)
    l=[]
    for i in envelope:
        l.append(list(i))
    return l
T=get_surface(tri.simplices)

meshio.write_points_cells("test_preprocessed.stl", newvertices, [("triangle", T)])
'''