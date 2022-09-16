#!/usr/bin/env python
# coding: utf-8
import numpy as np
import meshio
from scipy.interpolate import lagrange
import sympy
import math
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

def writetet6_2d(filename,X,Y):
    file=open(filename,"w")
    numverts=X.shape[0]
    file.write(str(numverts)+" vertices\n")    
    file.write("0 tets\n")
    for i in range(numverts):
        file.write(str(X[i,0])+" "+str(X[i,1])+" "+str(0)+" "+str(Y[i,0])+" "+str(Y[i,1])+" "+str(0)+" \n")
    file.close()

def area_triangle(a,b,c):
    M=np.zeros([3,3])
    M[0:2,0]=a
    M[0:2,1]=b
    M[0:2,2]=c
    M[2,:]=1
    return abs(np.linalg.det(M))/2

def area(mesh):
    area=0
    T = Delaunay(mesh)
    F=T.simplices
    for i in range(F.shape[0]):
        area=area+area_triangle(mesh[F[i,0]],mesh[F[i,1]],mesh[F[i,2]])
    return area

def get_map_interpolation(filename):
    X,Y,F=readtet6(filename)
    X=X[:,0:2]
    Y=Y[:,0:2]
    t = sympy.symbols("t")
    v=np.array([[0,1/2,1],[area(Y),area(1/2*Y+1/2*X),area(X)]])
    poly = lagrange(v[0], v[1])
    denom=poly[0]+poly[1]*t+poly[2]*(t**2)
    denom=denom**(1/2)
    num=poly[0]**(1/2)*(sympy.Matrix(Y)+(X-Y)*t)
    fun=num/denom
    return fun,t
    
X,Y,F=readtet6("morph.tet6") 

Xtemp=[]
Ytemp=[]
Ftemp=F.copy()

counter=0

for i in range(len(Y)):
    if Y[i].all()!=np.array([0,0,0]).all():
        Xtemp.append(X[i].tolist())
        Ytemp.append(Y[i].tolist())

Xtemp=np.array(Xtemp)
Ytemp=np.array(Ytemp)
Xtemp=Xtemp[:,0:2]
Ytemp=Ytemp[:,0:2]

writetet6_2d("correct_morph.tet6",Xtemp,Ytemp)


def get_boundary_2d_unstructured(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return set(np.unique(np.array(list(edges)).reshape(-1)).tolist())



'''
def get_boundary_2d_unstructured(points):
    T1 = Delaunay(points,False,False,"Qbb Qc Qz Q12 QJ Qz Qt")
    boundary = set()
    for i in range(len(T1.neighbors)):
        for k in range(3):
            if (T1.neighbors[i][k] == -1):
                nk1,nk2 = (k+1)%3, (k+2)%3 
                boundary.add(T1.simplices[i][nk1])
                boundary.add(T1.simplices[i][nk2])
    return boundary
'''

Xbind=get_boundary_2d_unstructured(Xtemp,0.05)
Ybind=get_boundary_2d_unstructured(Ytemp,0.05)
bond=list(Xbind.union(Ybind))
Xbound=Xtemp[bond]
Ybound=Ytemp[bond]
writetet6_2d("boundary.tet6",Xbound,Ybound)
'''
fun,symb=get_map_interpolation("correct_morph.tet6") 
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
    xyz=xyz[bond]
    pointsToVTK('./boundary.0.time.0'+f'{i:03}'+'.vtk', xyz[:,0].copy(), xyz[:,1].copy(), 0*xyz[:,1].copy())
'''
