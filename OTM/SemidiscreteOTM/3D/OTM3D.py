#!/usr/bin/env python
# coding: utf-8
import numpy as np
import meshio
from scipy.interpolate import lagrange
import sympy
import math
import laspy
import itertools
from scipy.spatial import Delaunay
from numpy.polynomial.polynomial import Polynomial
from pyevtk.hl import pointsToVTK


def readtet6(filename):
    counter=0
    counterfaces=0
    countervert=0
    X=[]
    Y=[]
    F=[]
    with open(filename, "r") as a_file:
        for line in a_file:
            stripped_line = line.split()
            if len(stripped_line)==2:
                if counter==0:
                    numverts=int(stripped_line[0])  
                    counter=counter+1;
                if counter==1:
                    numfaces=int(stripped_line[0])  
                    X=np.zeros([numverts,3])
                    Y=np.zeros([numverts,3])
                    F=np.zeros([numfaces,4],dtype=int)
                    
            if len(stripped_line)==6:
                X[countervert,0]=float(stripped_line[0])
                X[countervert,1]=float(stripped_line[1])
                X[countervert,2]=float(stripped_line[2])
                Y[countervert,0]=float(stripped_line[3])
                Y[countervert,1]=float(stripped_line[4])
                Y[countervert,2]=float(stripped_line[5])
                countervert=countervert+1
            if len(stripped_line)==5:
                F[counterfaces,0]=int(stripped_line[1])
                F[counterfaces,1]=int(stripped_line[2])
                F[counterfaces,2]=int(stripped_line[3])
                F[counterfaces,3]=int(stripped_line[4])
                counterfaces=counterfaces+1
    return X,Y,F

def writetet6_3d_all(filename,X,Y,F):
    file=open(filename,"w")
    numverts=X.shape[0]
    numfaces=F.shape[0]
    file.write(str(numverts)+" vertices\n")    
    file.write(str(numfaces)+" tets\n")
    for i in range(numverts):
        file.write(str(X[i,0])+" "+str(X[i,1])+" "+str(X[i,2])+" "+str(Y[i,0])+" "+str(Y[i,1])+" "+str(Y[i,2])+" \n")
    for i in range(numfaces):
        file.write(str(4)+" "+str(F[i,0])+" "+str(F[i,1])+" "+str(F[i,2])+" "+str(F[i,3])+"\n")
    file.close()
    
def writetet6_3d_points(filename,X,Y):
    file=open(filename,"w")
    numverts=X.shape[0]
    numfaces=0
    file.write(str(numverts)+" vertices\n")    
    file.write(str(numfaces)+" tets\n")
    for i in range(numverts):
        file.write(str(X[i,0])+" "+str(X[i,1])+" "+str(X[i,2])+" "+str(Y[i,0])+" "+str(Y[i,1])+" "+str(Y[i,2])+" \n")
    file.close()
    

def volume_tetra(a,b,c,d):
    M=np.zeros([4,4])
    M[0:3,0]=a
    M[0:3,1]=b
    M[0:3,2]=c
    M[0:3,3]=d
    M[3,:]=1
    return abs(np.linalg.det(M))/6


def volume(tetramesh,F):
    volume=0
    for i in range(F.shape[0]):
        volume=volume+volume_tetra(tetramesh[F[i,0]],tetramesh[F[i,1]],tetramesh[F[i,2]],tetramesh[F[i,3]])
    return volume

def get_matrix(X,indices):
    matrix=np.zeros([3,3])
    matrix[:,0]=X[indices[0]]-X[indices[3]]
    matrix[:,1]=X[indices[0]]-X[indices[3]]
    matrix[:,2]=X[indices[0]]-X[indices[3]]
    return matrix

def get_map(filename):
    X,Y,F=readtet6(filename)
    t = sympy.symbols("t")
    denom=0
    for i in range(len(F)):
        matrixX=get_matrix(X,F[i])
        matrixY=get_matrix(X,F[i])
        if np.linalg.det(matrixX)<0:
            temp=matrixX[:,0].copy()
            matrixX[:0]=matrixX[:,1].copy()
            matrixX[:1]=temp.copy()
            temp=matrixY[:,0].copy()
            matrixY[:0]=matrixY[:,1].copy()
            matrixY[:1]=temp.copy()
        matrix=sympy.Matrix(matrixY)+t*(sympy.Matrix(matrixX)-sympy.Matrix(matrixY))
        denom=denom+sympy.Abs(matrix.det()/6)
    denom=denom**(1/3)
    num=(sympy.Matrix(Y)+(X-Y)*t)*denom.subs(t,0)
    fun=num/denom
    fun=sympy.simplify(fun)
    return fun,t,F
    
def get_boundary_3d_unstructured(points):
    T1 = Delaunay(points)
    boundary = set()
    for i in range(len(T1.neighbors)):
        for k in range(3):
            if (T1.neighbors[i][k] == -1):
                nk1,nk2 = (k+1)%3, (k+2)%3 
                boundary.add(T1.simplices[i][nk1])
                boundary.add(T1.simplices[i][nk2])
    return boundary
    
def get_boundary_3d_structured(F):
    l=[]
    for i in range(len(F)):
        temp1=list(itertools.combinations(set(F[i]), 3))
        temp2=[sorted(x) for x in temp1]
        l=l+temp2    
    boundary_set=[x for x in l if l.count(x)==1]
    boundary_list=[list(x) for x in boundary_set]
    temp=np.array(boundary_list).reshape(-1)
    boundary=np.unique(temp).tolist()
    return boundary

def get_map_interpolation(filename):
    X,Y,F=readtet6(filename)
    t = sympy.symbols("t")
    v=np.array([[0,1/3,2/3,1],[volume(Y,F),volume(2/3*Y+1/3*X,F),volume(1/3*Y+2/3*X,F),volume(X,F)]])
    poly = lagrange(v[0], v[1])
    denom=poly[0]+poly[1]*t+poly[2]*(t**2)+poly[3]*(t**3)
    denom=denom**(1/3)
    num=poly[0]**(1/3)*(sympy.Matrix(Y)+(X-Y)*t)
    fun=num/denom
    bound=get_boundary_3d_structured(F)
    return fun,t,F,bound

    
fun,symb,F,bound=get_map_interpolation("morph.tet6") 
pfun=sympy.lambdify(symb,fun)
N=100

#pointA=np.array([0.96199671, 6.42071   , 1.26859406])
#pointB=pointA+np.array([0.0001,0.0001,0.0001])
#pointC=pointA+np.array([0.0001,0.0001,-0.0001])
#pointD=pointA+np.array([-0.0001,0.0001,-0.0001])
#f=int(np.max(F))
#F=np.vstack((F,np.array([f+1,f+2,f+3,f+4])))

#N=100
    
for i in range(N+1):
    xyz=pfun(i*1/(N))
    #xyz=np.vstack((xyz,pointA,pointB,pointC,pointD)) optional for coloring, do not use in production
    #meshio.write_points_cells('tetra.0.time.0'+f'{i:03}'+'.vtk', xyz, {'tetra': F}, {"T": xyz[:,1]})
    meshio.write_points_cells('tetra.0.time.0'+f'{i:03}'+'.vtk', xyz, {'tetra': F})
    #hdr = laspy.LasHeader(version="1.4", point_format=6)
    #mins = np.floor(np.min(xyz,axis=1))
    #hdr.offset = mins
    #hdr.scale = [0.01,0.01,0.01]
    xyz=xyz[bound]
    pointsToVTK('./boundary.0.time.0'+f'{i:03}'+'.vtk', xyz[:,0].copy(), xyz[:,1].copy(), xyz[:,2].copy())


#Writing unnormalized sequence for visualition (for production use vtk) 
X,Y,_=readtet6("morph.tet6")
Xbound=X[bound]
Ybound=Y[bound]
writetet6_3d_points("boundary.tet6",Xbound,Ybound)
