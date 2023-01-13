#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:58:15 2022

@author: cyberguli
"""
import time
import meshio
from numba import jit
from scipy.integrate import solve_ivp
from sympy.vector import Vector, CoordSys3D
from sympy.matrices import Matrix,ImmutableDenseMatrix
from sympy import *
from scipy.optimize import NonlinearConstraint,Bounds
from scipy.optimize import BFGS
import numpy as np
import scipy
import sympy
from scipy.optimize import minimize,root,basinhopping,brute,differential_evolution
np.random.seed(0)
from smithers.io import STLHandler
from pygem import FFD
import meshio
import scipy
import numpy as np
np.random.seed(0)

lapl_up_counter=0

def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume



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
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])
        for T in newtriangles_zero:
            if T[0] in newmesh_indices_local:
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
            else:
                edge_matrix[T[0],T[0]]=1
                
            if T[1] in newmesh_indices_local:
                edge_matrix[T[1],T[2]]=1
                edge_matrix[T[1],T[0]]=1
            else:
                edge_matrix[T[1],[T[1]]]=1
                
                
            if T[2] in newmesh_indices_local:
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
            else:
                edge_matrix[T[2],[T[2]]]=1
 

    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        edge_matrix=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix



def mysqrt(x):
    return np.sqrt(np.maximum(x,np.zeros_like(x)))

def myarccos(x):
    return np.arccos(np.minimum(0.9999*np.ones_like(x),np.maximum(x,-0.9999*np.ones_like(x))))

def myarccosh(x):
    return np.arccosh(np.maximum(1.00001*np.ones_like(x),x))

def multi_cubic(a, b, c, d):
    p=(3*a*c-b**2)/(3*a**2)
    q=(2*b**3-9*a*b*c+27*a**2*d)/(27*a**3)
    temp1=(p>=0).astype(int)*(-2*np.sqrt(np.abs(p)/3)*np.sinh(1/3*np.arcsinh(3*q/(2*np.abs(p))*np.sqrt(3/np.abs(p)))))
    temp2=(np.logical_and(p<0,(4*p**3+27*q**2)>0).astype(int)*(-2*np.abs(q)/q)*np.sqrt(np.abs(p)/3)*np.cosh(1/3*myarccosh(3*np.abs(q)/(2*np.abs(p))*np.sqrt(3/np.abs(p)))))
    temp3=np.logical_and(p<0,(4*p**3+27*q**2)<0).astype(int)*2*mysqrt(np.abs(p)/3)*np.max(np.stack((np.cos(1/3*myarccos(3*q/(2*p)*np.sqrt(3/np.abs(p)))-2*np.pi*0/3),np.cos(1/3*myarccos(3*q/(2*p)*np.sqrt(3/np.abs(p)))-2*np.pi*1/3),np.cos(1/3*myarccos(3*q/(2*p)*np.sqrt(3/np.abs(p)))-2*np.pi*2/3))),axis=0)
    return temp1+temp2+temp3-b/(3*a)



points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix=getinfo("hullpreprocessed.stl",True)
alls=np.zeros([100,points.shape[0],3])

temp=points_old[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)]
temp1=points_old[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
volume_const=volume(temp[newtriangles_zero])

print(volume_const)
t=[0,1]
temp1_ffd=temp1.copy()


voltrue=volume(temp[newtriangles_zero])



@jit
def objective_function(param):
    return np.sum(param**2)
@jit
def objective_function_der(param):
    return 2*param

@jit
def objective_function_hess(param):
    return np.diag(2*np.ones(len(param)))


for i in range(100,100):
    print(i)
    np.random.seed(i)
    init_param=np.random.uniform(0.2,0.5,24)

    def constraint(param):
        ffd=FFD([3,3,3])    
        ffd.array_mu_x[1:,1:,1:]+=param[0:len(param)//3].reshape(2,2,2)+init_param[0:len(param)//3].reshape(2,2,2)
        ffd.array_mu_y[1:,1:,1:]+=param[len(param)//3:2*len(param)//3].reshape(2,2,2)+init_param[len(param)//3:2*len(param)//3].reshape(2,2,2)
        ffd.array_mu_z[1:,1:,1:]+=param[2*len(param)//3:].reshape(2,2,2)+init_param[2*len(param)//3:].reshape(2,2,2)
        ffd.box_origin = np.array([0.01, 0.01, 0.01])
        ffd.box_length = np.array([0.62, 0.10, 0.23])
        temp_new=temp.copy()
        temp_new[np.logical_and(temp_new[:,2]>0,temp_new[:,0]>0)]=ffd(temp1)
        return (volume(temp_new[newtriangles_zero])-voltrue)/voltrue

    

    nonlinear_constraint = NonlinearConstraint(constraint, -0.01, 0.01, jac='2-point', hess=BFGS())
    bounds=Bounds(-3/4*init_param, init_param)
    x0=np.zeros(24)
        
    
    start=time.time()
    
    res = minimize(objective_function, x0, method='trust-constr', jac=objective_function_der, hess=objective_function_hess,
    
                   constraints=[nonlinear_constraint],
                   bounds=bounds,
                   options={'verbose': 1})
    
    
    
    end=time.time()
    print("Time elapsed is",end-start)
    param=res.x
    ffd=FFD([3,3,3])    
    points_new=points_old.copy()
    ffd.array_mu_x[1:,1:,1:]+=param[0:len(param)//3].reshape(2,2,2)+init_param[0:len(param)//3].reshape(2,2,2)
    ffd.array_mu_y[1:,1:,1:]+=param[len(param)//3:2*len(param)//3].reshape(2,2,2)+init_param[len(param)//3:2*len(param)//3].reshape(2,2,2)
    ffd.array_mu_z[1:,1:,1:]+=param[2*len(param)//3:].reshape(2,2,2)+init_param[2*len(param)//3:].reshape(2,2,2)
    ffd.box_origin = np.array([0.01, 0.01, 0.01])
    ffd.box_length = np.array([0.62, 0.10, 0.23])
    temp_new=temp.copy()
    temp1_new=ffd(temp1)
    temp_new[np.logical_and(temp_new[:,2]>0,temp_new[:,0]>0)]=ffd(temp1)
    points_new[np.logical_and(points_new[:,2]>0,points_new[:,0]>0)]=temp1_new
    meshio.write_points_cells("hull_{}.stl".format(i), points_new, [("triangle", triangles)])

    



    
