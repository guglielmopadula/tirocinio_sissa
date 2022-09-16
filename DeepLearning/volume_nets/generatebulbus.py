import numpy
import meshio
import stl
import numpy as np
your_mesh = meshio.read('../../Data/bulbo.vtk')
from scipy.integrate import solve_ivp,odeint
np.random.seed(0)

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

points=your_mesh.points
F=your_mesh.cells_dict["tetra"]
center=numpy.mean(points,axis=0)
points=points-center
V=volume(points,F)
points=points/(V**(1/3))
temp=numpy.copy(points)
vec1=abs(numpy.random.normal(0,1,100))
vec2=abs(numpy.random.normal(0,1,100))
vec3=abs(numpy.random.normal(0,1,100))
num1=numpy.random.beta(2,2,100)
num2=numpy.random.beta(2,2,100)
num3=numpy.random.beta(2,2,100)


for i in range(0,100):
    temp=numpy.copy(points)
    temp[:,0]=temp[:,0]*(1+vec1[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    temp[:,1]=temp[:,1]*(1+vec2[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    temp[:,2]=temp[:,2]*(1+vec3[i])/(((1+vec1[i])*(1+vec2[i])*(1+vec3[i]))**(1/3))
    meshio.write_points_cells('bulbo_{}.vtk'.format(i), temp.tolist(),[("triangle", your_mesh.cells_dict["triangle"]),("tetra", your_mesh.cells_dict["tetra"])])

def get_sol(a,b,c,points):
    def diff(x,t):
        return (a*x[2],0,-b)

    temp=points.copy()
    new_points=np.zeros(temp.shape)
    for i in range(temp.shape[0]):
        new_points[i,:] = np.array(odeint(diff,temp[i],np.linspace(0,3)))[20].reshape(-1)
    return new_points


for i in range(0,100):
    temp=numpy.copy(points)
    temp=get_sol(num1[i],num2[i],num3[i],temp)
    center=numpy.mean(temp,axis=0)
    temp=temp-center
    V=volume(temp,F)
    #temp=temp/(V**(1/3))
    meshio.write_points_cells('bulbo_{}.vtk'.format(100+i), temp.tolist(),[("triangle", your_mesh.cells_dict["triangle"]),("tetra", your_mesh.cells_dict["tetra"])])

def get_sol(a,b,c,points):    
    def diff(x,t):
        return (c*x[2],a,b)
    
    return np.array(odeint(diff,points,np.linspace(0,3)))

