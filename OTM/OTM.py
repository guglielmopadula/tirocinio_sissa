#!/usr/bin/env python
# coding: utf-8

import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import numpy as np
import meshio
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

def writetet6(filename,X,Y,F):
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


X,Y,F=readtet6("morph.tet6")
meshio.write_points_cells('sampled_after_transport.vtk', X, {'tetra': F})
meshio.write_points_cells('nonsampled_after_transport.vtk', Y, {'tetra': F})
print(volume(Y,F))
X=X/volume(X,F)**(1/3)
Y=Y/volume(Y,F)**(1/3)
print(volume(Y,F))

t=np.linspace(0,1,101)
volumes=np.zeros(101)
for i in range(101):
    volumes[i]=volume(X+t[i]*(Y-X),F)

    
file=open("tetra.0.vtk.series","w")
file.write('{\n')
file.write('  "file-series-version" : "1.0",\n')
file.write('"files" : [\n')
for i in range(101):
    xyz=X+t[i]*(Y-X)/volumes[i]
    meshio.write_points_cells('tetra.0.time.0'+f'{i:03}'+'.vtk', xyz, {'tetra': F})
    writetet6('tetra.0.time.0'+f'{i:03}'+'.tet6',xyz,xyz,F)
    if i!=999:  
        file.write('{ "name" : "tetra.0.time.0,'+f'{i:03}'+'.vtk"'+', "time" : {}'.format(t[i])+'},\n')
    else:
        file.write('{ "name" : "tetra.0.time.0,'+f'{i:03}'+'.vtk"'+', "time" : {}'.format(t[i])+'}\n')
file.write("]\n")
file.write("}\n")
file.close()

