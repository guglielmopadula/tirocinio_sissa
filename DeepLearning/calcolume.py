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

X,Y,F=readtet6('morph.tet6')
print(volume(X,F))
