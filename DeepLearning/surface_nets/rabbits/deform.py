#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:58:15 2022

@author: cyberguli
"""


import meshio
from scipy.integrate import solve_ivp
from sympy.vector import Vector, CoordSys3D
from sympy.matrices import Matrix
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



def volume_tetra(M):
    return abs(np.linalg.det(M))/6


def volume(mesh):
    volume=0
    for i in range(len(mesh)):
        volume=volume+volume_tetra(mesh[i,:,:])
    return volume

def getinfo(stl,flag):
    mesh=meshio.read(stl)
    points=mesh.points.astype(np.float32)
    if flag==True:
        triangles=mesh.cells_dict['triangle'].astype(np.int64)


    else:
        triangles=0
        
    return points,triangles



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



points,triangles=getinfo("bunny.stl",True)

volume_const=volume(points[triangles])

'''
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





def vec_def(alpha,beta,a,c):
    def fun(t,x):
        u=np.array([np.sin(alpha)*np.cos(beta),np.cos(alpha)*np.cos(beta),np.sin(beta)])
        w=np.array([np.cos(alpha),np.sin(alpha),0])
        return (u.dot(w)+np.cross(a,np.cross(-(2*a),np.cross(a,x-c))))*(x[2]>0.001).astype(int)*(x[1]>0.001).astype(int) 
    return fun

def jac_def(alpha,beta,a,c):
    def fun(t,x):
        jac=np.zeros([3,3])
        jac[0]=np.cross(np.cross(np.cross(np.array([1,0,0]),a),2*a),a)
        jac[1]=np.cross(np.cross(np.cross(np.array([0,1,0]),a),2*a),a)
        jac[2]=np.cross(np.cross(np.cross(np.array([0,0,1]),a),2*a),a)
        return jac
    return fun

            

t=[0,1]
temp_new=points.copy()


for counter in range(150):
    print(counter)
    alpha=2*np.pi*np.random.rand()
    beta=2*np.pi*np.random.rand()
    a=scipy.stats.truncnorm.rvs(-1,1,size=3)
    c=scipy.stats.truncnorm.rvs(-1,1,size=3)
    v=vec_def(alpha,beta,a,c)
    j=jac_def(alpha,beta,a,c)
    for i in range(len(points)):
        temp_new[i]=solve_ivp(v,t,points[i].astype(np.single),dense_output=True,jac=j,method="BDF").sol(0.01)
    points_new=temp_new/volume(temp_new[triangles])**(1/3)*volume_const**(1/3)
    meshio.write_points_cells("hull_{}.stl".format(counter), points_new, [("triangle", triangles)])


for j in range(150,300):
    print(j)
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

    for i in range(len(points)):
        temp_new[i]=solve_ivp(v_vec,t,points[i].astype(np.single),dense_output=True,jac=jac_vec,method="BDF").sol(1)
    points_new=temp_new/volume(temp_new[triangles])**(1/3)*volume_const**(1/3)
    meshio.write_points_cells("hull_{}.stl".format(j), points_new, [("triangle", triangles)])

'''
ffd = FFD([4, 4, 4])
ffd.box_origin = np.array([0, 0, 0])
ffd.box_length = np.array([1, 1, 1])
tempx=ffd.array_mu_x
tempy=ffd.array_mu_y
tempz=ffd.array_mu_z
for i in range(0,500):
    print(i)
    param_x = np.random.uniform(-0.03, 0.05, [4,4,4])
    param_y = np.random.uniform(-0.03, 0.05, [4,4,4])
    param_z = np.random.uniform(-0.03, 0.05, [4,4,4])


    ffd.reset_weights()
    for a in range(4):
        for b in range(4):
            for c in range(4):  
                ffd.array_mu_x[a,b,c] = tempx[a,b,c]+param_x[a][b][c]
                ffd.array_mu_y[a,b,c] = tempy[a,b,c]+param_y[a][b][c]
                ffd.array_mu_z[a,b,c] = tempz[a,b,c]+param_z[a][b][c]
    
    temp_new = ffd(points)
    points_new=temp_new/volume(temp_new[triangles])**(1/3)*volume_const**(1/3)
    print(len(temp_new))
    meshio.write_points_cells("hull_{}.stl".format(i), points_new, [("triangle", triangles)])
    