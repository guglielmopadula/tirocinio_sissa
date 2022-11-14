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


def deformationfield(h):
    def function(t,x):
        rs=np.random.RandomState(seed)
        y=np.zeros(3)
        counter=0
        i=0
        j=1
        k=1
        counter1=0
        a=0
        while counter<h:
            counter1=counter1%3        
            if counter1==0:
                if i<=j and i<=k:
                    i=i+1
                if j<i:
                    j=j+1
                if k<j:
                    k=k+1
                a=(np.pi**2*i**2+j**2+k**2)**(-3/2)*rs.randn()/h**(1/5)
                y=y+np.pi*a*np.array([0,k*np.sin(i*np.pi*x[0])*np.sin(j*np.pi*x[1])*np.cos(k*np.pi*x[2]),-j*np.sin(i*np.pi*x[0])*np.cos(j*np.pi*x[1])*np.sin(k*np.pi*x[2])])
            if counter1==1:
                a=i**2+j**2+k**2*rs.random.randn()
                y=y+np.pi*a*np.array([-k*np.sin(i*np.pi*x[0])*np.sin(j*np.pi*x[1])*np.cos(k*np.pi*x[2]),0,i*np.cos(i*np.pi*x[0])*np.sin(j*np.pi*x[1])*np.sin(k*np.pi*x[2])])
            if counter1==2:
                a=i**2+j**2+k**2*rs.random.randn()
                y=y+np.pi*a*np.array([j*np.sin(i*np.pi*x[0])*np.cos(j*np.pi*x[1])*np.sin(k*np.pi*x[2]),-i*np.cos(i*np.pi*x[0])*np.sin(j*np.pi*x[1])*np.sin(k*np.pi*x[2]),0])
            counter=counter+1
            counter1=counter1+1
        return y
    return function

'''
def deformationfield_symb(h):
    x,y,z=sympy.symbols("x,y,z")
    sol=Matrix([[0,0,0]])
    counter=0
    i=0
    j=1
    k=1
    counter1=0
    a=0
    while counter<h:
        counter1=counter1%3        
        if counter1==0:
            if i<=j and i<=k:
                i=i+1
            if j<i:
                j=j+1
            if k<j:
                k=k+1
            a=(np.pi**2*i**2+j**2+k**2)**(-3/2)*np.random.randn()
            sol=sol+np.pi*a*Matrix([[0,k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),-j*sympy.sin(i*np.pi*x)*sympy.cos(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
        if counter1==1:
            a=(np.pi**2*i**2+j**2+k**2)**(-3/2)*np.random.randn()
            sol=sol+np.pi*a*Matrix([[-k*sympy.sin(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.cos(k*np.pi*z),0,i*sympy.cos(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.sin(k*np.pi*z)]])
        if counter1==2:
            a=(np.pi**2*i**2+j**2+k**2)**(-3/2)*np.random.randn()
            sol=sol+np.pi*a*Matrix([[j*sympy.sin(i*np.pi*x)*sympy.cos(j*np.pi*y)*sympy.sin(k*np.pi*z),-i*sympy.cos(i*np.pi*x)*sympy.sin(j*np.pi*y)*sympy.sin(k*np.pi*z),0]])
        counter=counter+1
        counter1=counter1+1
    return sol
'''


def deformationfield_symb(h):
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
    v_symb=deformationfield_symb(30)
    x,y,z=sympy.symbols("x,y,z")
    #v_symb=v_symb.applyfunc(sympy.simplify)
    jac_symb=v_symb.jacobian([x,y,z])
    v=sympy.lambdify([x,y,z],v_symb,modules='numpy')
    jac=sympy.lambdify([x,y,z],jac_symb,modules='numpy')
    def jac_vec(t,x):
        return jac(x[0],x[1],x[2]).reshape(3,3)

    def v_vec(t,x):
        return v(x[0],x[1],x[2]).reshape(-1)

    for i in range(len(newmesh)):
        defmesh[i]=solve_ivp(v_vec,t,newmesh[i].astype(np.single),dense_output=True,jac=jac_vec,method="BDF").sol(1)
    deform_stl(mesh,defmesh,newmesh_indices,j)
    

meshio.write_points_cells('test.stl', defmesh, [("triangle", newtriangles)])
    
    
    
    