#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:58:15 2022

@author: cyberguli
"""

import meshio
from scipy.integrate import solve_ivp
from sympy.vector import Vector, CoordSys3D
from sympy.matrices import Matrix,ImmutableDenseMatrix
from sympy import *
import numpy as np
import scipy
import sympy
np.random.seed(0)
import time
from smithers.io import STLHandler
from pygem import FFD
import meshio
import scipy
import numpy as np
np.random.seed(0)

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
    
    B=1-np.prod(points_zero[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)]**(1/5),axis=1)
    #B=(1-points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2])**1/2
    alpha_z=get_coeff_z(vertices_face, points_zero_2, newtriangles_zero).copy()
    hz=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]
    Az=alpha_z
    bz=a
    x = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(cp.multiply(B,x))),
                      [-x <= (1-0.05)*hz,
                       Az @ x == bz])
    prob.solve()
    def_z=x.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),2]+def_z
    
    B=1-np.prod(points_zero[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)]**(1/5),axis=1)
    #B=(1-points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1])**1/2
    alpha_y=get_coeff_y(vertices_face, points_zero_2, newtriangles_zero).copy()
    hy=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]
    Ay=alpha_y
    by=a
    x = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(cp.multiply(B,x))),
                      [-x <= (1-0.05)*hy,
                       Ay @ x == by])
    prob.solve()
    def_y=x.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),1]+def_y
    B=1-np.prod(points_zero[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)]**(1/5),axis=1)
    #B=(1-points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0])**1/2
    alpha_x=get_coeff_x(vertices_face, points_zero_2, newtriangles_zero).copy()
    hx=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]
    Ax=alpha_x
    bx=a

    var = cp.Variable(len(vertices_face))
    prob = cp.Problem(cp.Minimize((1/2)*cp.sum_squares(cp.multiply(B,var))),
                  [-var <= (1-0.05)*hx,
                   Ax @ var == bx])
    prob.solve()
    def_x=var.value
    points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]=points_zero_2[np.logical_and(points_zero_2[:,2]>0,points_zero_2[:,0]>0),0]+def_x
    return points_zero_2



points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face=getinfo("/home/cyberguli/newhullrotatedhalveremesheddirty.stl",True)
points_old=points_old+0.0
temp=points_old[np.logical_and(points_old[:,2]>=0.0,points_old[:,0]>=0.0)]
temp1=points_old[np.logical_and(points_old[:,2]>0.0,points_old[:,0]>0.0)]
volume_const=volume_2_x(temp[newtriangles_zero])

t=[0,1]
temp1_ffd=temp1.copy()
def laplace_symb(h):
    x,y,z=sympy.symbols("x,y,z")
    sol=Matrix([[0,0,0]])
    counter=0
    i=1
    j=1
    k=1
    a=0
    s=0
    while counter<h:
        for itemp in range(s+1):
            for jtemp in range(s-itemp+1):
                ktemp=s-itemp-jtemp
                i=itemp+1
                j=jtemp+1
                k=ktemp+1
                a=(np.pi**2*((i**2+j**2+k**2)))**(-3/2)*np.random.randn()
                sol=sol+np.pi*a*Matrix([[0,k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),-j*sympy.sin(i*np.pi*x)*sympy.cos(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
                counter=counter+1
                if counter==h:
                    return sol
                a=(np.pi**2*((i**2+j**2+k**2)))**(-3/2)*np.random.randn()
                sol=sol+np.pi*a*Matrix([[-k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),0,i*sympy.cos(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
                counter=counter+1
                if counter==h:
                    return sol
                a=(np.pi**2*(i**2+j**2+k**2))**(-3/2)*np.random.randn()
                sol=sol+np.pi*a*Matrix([[j*sympy.sin(i*np.pi*x)*sympy.cos(j*np.pi*y)*sympy.sin(k*np.pi*z),-i*sympy.cos(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.sin(k*np.pi*z),0]])
                counter=counter+1
                if counter==h:
                    return sol
        s=s+1
    return sol

def smoother(mesh,edge_matrix):
    mesh_temp=mesh.reshape(1,mesh.shape[0],mesh.shape[1])
    mesh_temp=np.swapaxes(mesh_temp,1,2)
    mesh_temp=np.matmul(mesh_temp,edge_matrix.T)    
    mesh_temp=np.swapaxes(mesh_temp,1,2)
    num=np.sum(edge_matrix,axis=1)
    num=num.reshape(1,-1,1)
    num=np.tile(num,[mesh_temp.shape[0],1,mesh_temp.shape[2]])
    mesh_temp=mesh_temp/num
    return mesh_temp.reshape(mesh_temp.shape[1],mesh_temp.shape[2])

def k_smoother(k,mesh,edge_matrix):
    mesh_temp=mesh
    for i in range(k):
        mesh_temp=smoother(mesh_temp,edge_matrix)
    
    return mesh_temp



def vec_def(alpha,beta,a,c):
    def fun(t,x):
        u=np.array([np.sin(alpha)*np.cos(beta),np.cos(alpha)*np.cos(beta),np.sin(beta)])
        w=np.array([np.cos(alpha),np.sin(alpha),0])
        return (u.dot(w)+np.cross(a,np.cross(-(2*a),np.cross(a,x-c))))*(x[2]>0.001).astype(int)*(x[1]>0.001).astype(int)*(x[0]>0.001).astype(int) 
    return fun

def jac_def(alpha,beta,a,c):
    def fun(t,x):
        jac=np.zeros([3,3])
        jac[0]=np.cross(np.cross(np.cross(np.array([1,0,0]),a),2*a),a)
        jac[1]=np.cross(np.cross(np.cross(np.array([0,1,0]),a),2*a),a)
        jac[2]=np.cross(np.cross(np.cross(np.array([0,0,1]),a),2*a),a)
        return jac
    return fun


param_ffd=np.zeros([100,24])





for i in range(0,100):
    a=3
    b=3
    c=3
    print(i)
    if i%100==0:
        print(i)
    ffd = FFD([a, b, c])
    
    ffd.box_origin = np.array([0.0, 0.0, 0.0])
    ffd.box_length = np.array([0.62, 0.10, 0.23])
    param= (np.random.uniform(-0.30, 0.10, 24))
    param2=np.random.uniform(0.10,0.20,2)
    ffd.array_mu_x[a-1, b-2, c-2] += param[0]
    ffd.array_mu_x[a-1, b-1, c-2] += param[1]
    ffd.array_mu_x[a-1, b-2, c-1] += param[2]
    ffd.array_mu_x[a-1, b-1, c-1] += param[3]
    ffd.array_mu_x[a-2, b-2, c-2] += param[4]
    ffd.array_mu_x[a-2, b-1, c-2] += param[5]
    ffd.array_mu_x[a-2, b-2, c-1] += param[6]
    ffd.array_mu_x[a-2, b-1, c-1] += param[7]
    ffd.array_mu_y[a-1, b-2, c-2] += param[8]
    ffd.array_mu_y[a-1, b-1, c-2] += param[9]
    ffd.array_mu_y[a-1, b-2, c-1] += param[10]
    ffd.array_mu_y[a-1, b-1, c-1] += param[11]
    ffd.array_mu_y[a-2, b-2, c-2] += param[12]
    ffd.array_mu_y[a-2, b-1, c-2] += param[13]
    ffd.array_mu_y[a-2, b-2, c-1] += param[14]
    ffd.array_mu_y[a-2, b-1, c-1] += param[15]
    ffd.array_mu_z[a-1, b-2, c-2] += param[16]
    ffd.array_mu_z[a-1, b-1, c-2] += param[17]
    ffd.array_mu_z[a-1, b-2, c-1] += param[18]
    ffd.array_mu_z[a-1, b-1, c-1] += param[19]
    ffd.array_mu_z[a-2, b-2, c-2] += param[20]
    ffd.array_mu_z[a-2, b-1, c-2] += param[21]
    ffd.array_mu_z[a-2, b-2, c-1] += param[22]
    ffd.array_mu_z[a-2, b-1, c-1] += param[23]
    ffd.array_mu_x[a-1, 0, c-1] += param2[0]
    ffd.array_mu_x[a-1, 0, c-2] += param2[1]


    
    temp1_ffd = ffd(temp1)
    temp_new=temp.copy()
    a=volume_2_z(temp_new[newtriangles_zero])
    points_new=points_old.copy()
    temp1_ffd[temp1_ffd[:,1]<0,1]=0
    temp_new[newmesh_indices_local]=temp1_ffd
    temp_new=volume_norm(temp_new, newtriangles_zero, vertices_face,a)
    points_new[np.logical_and(points_new[:,2]>=0,points_new[:,0]>=0)]=temp_new
    meshio.write_points_cells("hull_new_{}.stl".format(i), points_new, [("triangle", triangles)])

'''    
temp1_lapl=temp1.copy()
for j in range(0,100):
    if j==0:
        start=time.time()
    print(j)
    if j%100==0:
        print(j)
    lapl_up_counter=j
    v_symb=laplace_symb(30)
    x,y,z=sympy.symbols("x,y,z")
    jac_symb=v_symb.jacobian([x,y,z])
    v=sympy.lambdify([x,y,z],v_symb,modules='numpy')
    jac=sympy.lambdify([x,y,z],jac_symb,modules='numpy')
    def jac_vec(t,x):
        return jac(x[0],x[1],x[2]).reshape(3,3)

    def v_vec(t,x):
        return v(x[0],x[1],x[2]).reshape(-1)

    for i in range(len(temp1)):
        temp1_lapl[i]=solve_ivp(v_vec,t,temp1[i].astype(np.single),dense_output=True,jac=jac_vec,method="DOP853").sol(1)
    temp_new=temp.copy()
    temp_new[newmesh_indices_local]=temp1_lapl
    points_new=points_old.copy()
    temp_new=volume_norm(temp_new, newtriangles_zero, vertices_face)
    points_new[np.logical_and(points_new[:,2]>=0,points_new[:,0]>=0)]=temp_new
    if j==0:
        end=time.time()
        print(end-start)
    meshio.write_points_cells("test_{}.stl".format(j), points_new, [("triangle", triangles)])   


print("Total Variance is", np.sum(np.var(alls,axis=0)))

ffd_dict={"params":param_ffd,"output":alls}
np.save("ffd.npy",ffd_dict)
'''