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



points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix=getinfo("hull_negative.stl",True)
alls=np.zeros([100,points.shape[0],3])

temp=points_old[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)]
temp1=points_old[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
volume_const=volume(temp[newtriangles_zero])


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
                a=(np.pi**2*((i**2+j**2+k**2)))**(-7/4)*np.random.randn()
                sol=sol+np.pi*a*Matrix([[0,k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),-j*sympy.sin(i*np.pi*x)*sympy.cos(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
                counter=counter+1
                if counter==h:
                    return sol
                a=(np.pi**2*((i**2+j**2+k**2)))**(-7/4)*np.random.randn()
                sol=sol+np.pi*a*Matrix([[-k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),0,i*sympy.cos(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
                counter=counter+1
                if counter==h:
                    return sol
                a=(np.pi**2*(i**2+j**2+k**2))**(-7/4)*np.random.randn()
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




for counter in range(100):
    if counter%100==0:
        print(counter)
    alpha=1/4*np.pi*np.random.rand()
    beta=1/4*np.pi*np.random.rand()
    a=scipy.stats.truncnorm.rvs(0.2,0.8,size=3)
    c=scipy.stats.truncnorm.rvs(0.2,0.8,size=3)
    v=vec_def(alpha,beta,a,c)
    j=jac_def(alpha,beta,a,c)
    for i in range(len(temp1)):
        temp1_ffd[i]=solve_ivp(v,t,temp1[i].astype(np.single),dense_output=True,jac=j,method="BDF").sol(0.01)
    temp_new=temp.copy()
    points_new=points_old.copy()
    #a1=np.random.uniform(0.8,1.2,1)
    #b1=np.random.uniform(0.8,1.2,1)
    #c1=np.random.uniform(0.8,1.2,1)
    #temp1_new[:,0]=temp1_new[:,0]*a1
    #temp1_new[:,1]=temp1_new[:,1]*b1
    #temp1_new[:,2]=temp1_new[:,2]*c1
    alls[counter]=temp1_ffd
    temp_new[newmesh_indices_local]=temp1_ffd
    a=(np.abs(np.linalg.det(temp_new[newtriangles_local_3])).sum()/6)
    b=(np.abs(np.linalg.det(temp_new[newtriangles_local_2])).sum()/6)
    c=(np.abs(np.linalg.det(temp_new[newtriangles_local_1])).sum()/6)
    d=-volume_const
    k=multi_cubic(a, b, c, d)
    temp1_ffd=temp1_ffd*k
    temp_new[newmesh_indices_local]=temp1_ffd
    print(volume(temp_new[newtriangles_zero]))
    points_new[np.logical_and(points_new[:,2]>0,points_new[:,0]>0)]=temp1_ffd
    alls[counter]=temp1_ffd
    meshio.write_points_cells("hull_negative_{}.stl".format(counter), points_new, [("triangle", triangles)])
   




for i in range(0,0):
    a=3
    b=3
    c=3
    if i%100==0:
        print(i)
    ffd = FFD([a, b, c])
    
    ffd.box_origin = np.array([0.01, 0.01, 0.01])
    ffd.box_length = np.array([0.62, 0.10, 0.23])
    param= (np.random.uniform(-1, 0, 24))
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

    
    

    temp1_ffd = ffd(temp1)
    param_ffd[i]=param
    #temp1_new=((temp1_new)/np.array([0.48,0.42,0.23]))**2*temp1_new+(1-((temp1_new)/np.array([0.48,0.42,0.23])))*temp1
    temp_new=temp.copy()
    points_new=points_old.copy()
    temp_new[newmesh_indices_local]=temp1_ffd
    temp_new=k_smoother(5,temp_new, edge_matrix)
    temp1_new=temp_new[np.logical_and(temp_new[:,2]>0,temp_new[:,0]>0)]
    a=(np.abs(np.linalg.det(temp_new[newtriangles_local_3])).sum()/6)
    b=(np.abs(np.linalg.det(temp_new[newtriangles_local_2])).sum()/6)
    c=(np.abs(np.linalg.det(temp_new[newtriangles_local_1])).sum()/6)
    d=-volume_const
    k=multi_cubic(a, b, c, d)
    temp1_new=temp1_new*k
    variance_vol=np.sum(np.var(alls,axis=0))
    points_new[np.logical_and(points_new[:,2]>0,points_new[:,0]>0)]=temp1_new
    alls[i]=temp1_ffd
    meshio.write_points_cells("hull_{}.stl".format(i), points_new, [("triangle", triangles)])
   
    
   
for j in range(0,0):
    print(j)
    if j%100==0:
        print(j)
    m=np.max(temp1,axis=0)
    #temp2=temp1/m
    lapl_up_counter=j
    v_symb=laplace_symb(30)
    x,y,z=sympy.symbols("x,y,z")
    #v_symb=v_symb.applyfunc(sympy.simplify)
    jac_symb=v_symb.jacobian([x,y,z])
    v=sympy.lambdify([x,y,z],v_symb,modules='numpy')
    jac=sympy.lambdify([x,y,z],jac_symb,modules='numpy')
    def jac_vec(t,x):
        return jac(x[0],x[1],x[2]).reshape(3,3)

    def v_vec(t,x):
        return v(x[0],x[1],x[2]).reshape(-1)

    for i in range(len(temp1)):
        temp1_ffd[i]=solve_ivp(v_vec,t,temp1[i].astype(np.single),dense_output=True,jac=jac_vec,method="BDF").sol(1)
    #temp1_ffd=temp1_ffd*m
    temp_new=temp.copy()
    points_new=points_old.copy()
    temp_new[newmesh_indices_local]=temp1_ffd
    #a1=np.random.uniform(0.8,1.2,1)
    #b1=np.random.uniform(0.8,1.2,1)
    #c1=np.random.uniform(0.8,1.2,1)
    #temp1_new[:,0]=temp1_new[:,0]*a1
    #temp1_new[:,1]=temp1_new[:,1]*b1
    #temp1_new[:,2]=temp1_new[:,2]*c1
    #temp_new=k_smoother(1,temp_new, edge_matrix)
    if np.max(np.isnan(temp_new)):
        print("Found NAN")
    temp1_new=temp_new[np.logical_and(temp_new[:,2]>0,temp_new[:,0]>0)]
    #a=(np.abs(np.linalg.det(temp_new[newtriangles_local_3])).sum()/6)
    #b=(np.abs(np.linalg.det(temp_new[newtriangles_local_2])).sum()/6)
    #c=(np.abs(np.linalg.det(temp_new[newtriangles_local_1])).sum()/6)
    #d=-volume_const
    #k=multi_cubic(a, b, c, d)
    #temp1_new=temp1_new*k
    alls[j]=temp1_new
    points_new[np.logical_and(points_new[:,2]>0,points_new[:,0]>0)]=temp1_new
    print(volume(temp_new[newtriangles_zero]))
    meshio.write_points_cells("hull_{}.stl".format(j), points_new, [("triangle", triangles)])   

print("Total Variance is", np.sum(np.var(alls,axis=0)))

ffd_dict={"params":param_ffd,"output":alls}
np.save("ffd.npy",ffd_dict)