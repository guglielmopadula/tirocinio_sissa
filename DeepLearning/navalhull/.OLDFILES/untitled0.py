#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:41:16 2022

@author: cyberguli
"""

import meshio
import trimesh
from scipy.integrate import solve_ivp
from sympy.vector import Vector, CoordSys3D
from sympy.matrices import Matrix,ImmutableDenseMatrix
from sympy import *
from sklearn.metrics.pairwise import pairwise_kernels

import numpy as np
import scipy
import sympy
np.random.seed(0)
from smithers.io import STLHandler
from pygem import FFD
import meshio
import scipy
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def getinfo(stl,flag):
    mesh=meshio.read(stl)
    points_old=mesh.points.astype(np.float32)
    points=points_old[points_old[:,2]>0]
    points_zero=points_old[points_old[:,2]>=0]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[mesh.points[:,2]>0].tolist()
        triangles=mesh.cells_dict['triangle'].astype(np.int64)
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        newmesh_indices_global_zero=np.arange(len(mesh.points))[mesh.points[:,2]>=0].tolist()
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[points_zero[:,2]>0].tolist()
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
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,edge_matrix

def relativemmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))/(np.sqrt(1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian')))+np.sqrt(1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))))
    
def mmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))

def kl_divergence_sampled(X,Y):
   n=len(X)
   m=len(Y)
   d=len(X[0])
   tempX=np.linalg.norm(X[:,None,:]-X,axis=2)
   tempY=np.linalg.norm(X[:,None,:]-Y,axis=2)
   p=np.sort(tempX)[:,1]
   v=np.sort(tempY)[:,0]
   eps=np.max([p,v],axis=0).reshape(n,1)
   l=np.sum(tempX<=eps,axis=1)
   k=np.sum(tempY<=eps,axis=1)
   pi=np.sort(tempX)[np.arange(len(tempX)),l-1]
   vi=np.sort(tempY)[np.arange(len(tempY)),k-1]
   return d/n*np.sum(np.log(vi/pi))+1/n*np.sum(scipy.special.digamma(l)-scipy.special.digamma(k))+np.log(m/(n-1))

def sym_kl(X,Y,k=1):
    return scipy_estimator(X, Y,k)+scipy_estimator(X,Y,k)


def scipy_estimator(s1, s2, k=1):
    """ KL-Divergence estimator using scipy's KDTree
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """
    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))
    
    rho_d, rhio_i = KDTree(s1).query(s1, k+1)
    nu_d,  nu_i   = KDTree(s2).query(s1, k)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d/n)*np.sum(np.log(nu_d[::, -1]/rho_d[::, -1]))
    else:
        D += (d/n)*np.sum(np.log(nu_d/rho_d[::, -1]))

    return D

points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,edge_matrix=getinfo("hull_0.stl",True)

a=points_zero.shape
temp1=np.zeros([500,681,3])
temp2=np.zeros([500,681,3])
temp3=np.zeros([500,681,3])


names=["LAPLACE","AEE"]
j=0


for i in range(500):
    temp1[i]=getinfo("hull_{}.stl".format(i),False)[1]
    temp2[i]=getinfo("hull_negative_{}.stl".format(i),False)[1]
    temp3[i]=getinfo("test_AE_intPCA_{}.stl".format(i),False)[1]


area_real=np.zeros(500)
curvature_gaussian_real=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_mean_real=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_total_real=np.zeros(500)

curvature_gaussian_sampled_lapl=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_mean_sampled_lapl=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_total_sampled_lapl=np.zeros(500)
area_sampled_lapl=np.zeros(500)
curvature_gaussian_sampled_aae=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_mean_sampled_aae=np.zeros([500,np.max(newtriangles_zero)+1])
curvature_total_sampled_aae=np.zeros(500)
area_sampled_aae=np.zeros(500)



for i in range(500):
    mesh_object=trimesh.base.Trimesh(temp1[i],newtriangles_zero,process=False)
    curvature_gaussian_real[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_mean_real[i]=trimesh.curvature.discrete_mean_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_total_real[i]=np.sum(curvature_gaussian_real[i])
    area_real[i]=mesh_object.area
curvature_total_real=curvature_total_real.reshape(-1,1)
area_real=area_real.reshape(-1,1)






for i in range(500):
    mesh_object=trimesh.base.Trimesh(temp2[i],newtriangles_zero,process=False)
    curvature_gaussian_sampled_lapl[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_mean_sampled_lapl[i]=trimesh.curvature.discrete_mean_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_total_sampled_lapl[i]=np.sum(curvature_gaussian_sampled_lapl[i])
    area_sampled_lapl[i]=mesh_object.area
curvature_total_sampled_lapl=curvature_total_sampled_lapl.reshape(-1,1)
area_sampled_lapl=area_sampled_lapl.reshape(-1,1)


for i in range(500):
    mesh_object=trimesh.base.Trimesh(temp3[i],newtriangles_zero,process=False)
    curvature_gaussian_sampled_aae[i]=trimesh.curvature.discrete_gaussian_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_mean_sampled_aae[i]=trimesh.curvature.discrete_mean_curvature_measure(mesh_object, mesh_object.vertices, 0.05)
    curvature_total_sampled_aae[i]=np.sum(curvature_gaussian_sampled_aae[i])
    area_sampled_aae[i]=mesh_object.area
curvature_total_sampled_aae=curvature_total_sampled_aae.reshape(-1,1)
area_sampled_aae=area_sampled_aae.reshape(-1,1)





def relativemmd_n(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    gamma=1/(len(X[0]))
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian',gamma=gamma))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian',gamma=gamma))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian',gamma=gamma))))/(np.sqrt(1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian',gamma=gamma)))+np.sqrt(1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian',gamma=gamma))))

def relativemmd_n(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    gamma=1/(len(X[0])**(3/2))
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='rbf',gamma=gamma))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='rbf',gamma=gamma))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='rbf',gamma=gamma))))/(np.sqrt(1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='rbf',gamma=gamma)))+np.sqrt(1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='rbf',gamma=gamma))))



print("MMD Gaussian Curvature distance of Laplace", "is", relativemmd_n(curvature_gaussian_real,curvature_gaussian_sampled_lapl))
print("MMD Gaussian Curvature distance of AAE", "is", relativemmd_n(curvature_gaussian_real,curvature_gaussian_sampled_aae))
print("MMD Id distance of Laplace", relativemmd_n(temp1,temp2))
print("MMD Id distance of AAE", relativemmd_n(temp1,temp3))
print("MMD Area distance of Laplace" "is", relativemmd_n(area_real,area_sampled_lapl))
print("MMD Area distance of AAE" "is", relativemmd_n(area_real,area_sampled_aae))
print("MMD Total Curvature distance of Laplace", "is", relativemmd_n(curvature_total_real,curvature_total_sampled_lapl))
print("MMD Total Curvature distance of AAE", "is", relativemmd_n(curvature_total_real,curvature_total_sampled_aae))




