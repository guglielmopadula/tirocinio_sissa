#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:16:55 2022

@author: cyberguli
"""

import numpy as np
import cvxpy as cp
np.random.seed(0)
import meshio


def volume_prism_x(M):
    return np.sum(M[:,0])*(np.linalg.det(M[1:,1:]-M[0,1:])/6)

def volume_prism_y(M):
    return np.sum(M[:,1])*(np.linalg.det(M[np.ix_((0,2),(0,2))]-M[1,[0,2]]))/6

def volume_prism_z(M):
    return np.sum(M[:,2])*(np.linalg.det(M[:2,:2]-M[2,:2])/6)


def volume_2_x(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_x(mesh[i,:,:])
    return volume

def volume_2_y(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_y(mesh[i,:,:])
    return volume

def volume_2_z(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_prism_z(mesh[i,:,:])
    return volume

def get_coeff_z(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,:2,:2]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,2,:2,np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])

def get_coeff_x(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,1:,1:]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,0,1:,np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])

def get_coeff_y(vertices_face,points_zero,newtriangles_zero):
    return np.array([np.sum(np.linalg.det(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][np.ix_(np.arange(len(vertices_face[i])),(0,2),(0,2))]-np.repeat(points_zero[np.array(newtriangles_zero)[vertices_face[i]]][:,1,[0,2],np.newaxis],2,axis=0).reshape(-1,2,2))/6) for i in range(len(vertices_face))])



def getinfo(stl,flag):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points_old=mesh.points.astype(np.float32)
    points=points_old[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
    points_zero=points_old[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)].tolist()
        triangles=mesh.cells_dict['triangle'].astype(np.int64)
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        newmesh_indices_global_zero=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)].tolist()
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)].tolist()
        newtriangles_local_3=[]
        newtriangles_local_2=[]
        newtriangles_local_1=[]
        edge_matrix=np.zeros([np.max(newtriangles_zero)+1,np.max(newtriangles_zero)+1])
        vertices_face=[set({}) for i in range(len(newmesh_indices_local))]
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])
        
        for i in range(len(newtriangles_zero)):
            T=newtriangles_zero[i]
            if T[0] in newmesh_indices_local:
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
                vertices_face[newmesh_indices_local.index(T[0])].add(i)
            else:
                edge_matrix[T[0],T[0]]=1
                
            if T[1] in newmesh_indices_local:
                edge_matrix[T[1],T[2]]=1
                edge_matrix[T[1],T[0]]=1
                vertices_face[newmesh_indices_local.index(T[1])].add(i)
            else:
                edge_matrix[T[1],[T[1]]]=1
                
                
            if T[2] in newmesh_indices_local:
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
                vertices_face[newmesh_indices_local.index(T[2])].add(i)
            else:
                edge_matrix[T[2],[T[2]]]=1
        vertices_face=[list(t) for t in vertices_face]

    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        vertices_face=0
        edge_matrix=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face


def volume_norm(points_zero,newtriangles_zero, vertices_face,volume_const):
    points_zero_2=points_zero.copy()
    a=1/3*(volume_const-volume_2_x(points_zero[newtriangles_zero]))


    alpha_z=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero).copy()
    hz=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]
    Az=alpha_z
    bz=a
    x = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(x)),
                      [-x <= (1-0.05)*hz,
                       Az @ x == bz])
    prob.solve()
    def_z=x.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]+def_z
    

    
    alpha_y=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero).copy()
    hy=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]
    Ay=alpha_y
    by=a
    x = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(x)),
                      [-x <= (1-0.05)*hy,
                       Ay @ x == by])
    prob.solve()
    def_y=x.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]+def_y
    alpha_x=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero).copy()
    hx=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]
    Ax=alpha_x
    bx=a
    var = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(var)),
                  [-var <= (1-0.05)*hx,
                   Ax @ var == bx])
    prob.solve()
    def_x=var.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]+def_x
    return points_zero_2


points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face=getinfo("/home/cyberguli/newhullrotatedhalveremesheddirty.stl",True)
volume_const=volume_2_z(points_zero[newtriangles_zero])
points_zero_2=volume_norm(points_zero,newtriangles_zero,vertices_face,volume_const)

