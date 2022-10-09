#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:02:08 2022

@author: cyberguli
"""

import numpy as np




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
    
X1,Y1,F1=readtet6("morph_a.tet6")
X2,Y2,F2=readtet6("morph_b.tet6")

Y=0.5*Y1+0.5*Y2

writetet6_3d_all("morph_int.tet6",X1,Y,F1)
