import numpy
import meshio
import numpy as np
your_mesh = meshio.read('../../Data/bulbo.stl')
from scipy.integrate import odeint
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
        volume=volume+volume_tetra(tetramesh[F[i,0]],tetramesh[F[i,1]],tetramesh[F[i,2]],np.zeros(3))
    return volume

points=your_mesh.points
F=your_mesh.cells_dict["triangle"]
center=numpy.mean(points,axis=0)
points=points-center
V=volume(points,F)
points=points/(V**(1/3))
points=points/(V**(1/3))

size=0.05
num1=size*numpy.random.rand(500)
num2=size*numpy.random.rand(500)
num3=size*numpy.random.rand(500)
num4=size*numpy.random.rand(500)
num5=size*numpy.random.rand(500)
num6=size*numpy.random.rand(500)
num7=size*numpy.random.rand(500)
num8=size*numpy.random.rand(500)
num9=size*numpy.random.rand(500)

size1=1+abs(np.random.rand(500))
size2=1+abs(np.random.rand(500))
size3=1+abs(np.random.rand(500))




def get_sol(a,b,c,d,e,f,g,h,i,points):
    def diff1(x,t):
        return (a*x[1]**3+0.1*np.log(1+x[0]**2),b*x[2]**3+0.1*np.log(1+x[1]**2),c*x[0]**3+0.1*np.log(1+x[2]**2))

    def diff2(x,t):
        return (a*x[2]**3,b*x[0]**3,c*x[1]**3)
    
    def diff3(x,t):
        return (a*x[1]**3,b*x[2]**3,c*x[1]**3)



    temp=points.copy()
    new_points=np.zeros(temp.shape)
    for i in range(temp.shape[0]):
        new_points[i,:] = np.array(odeint(diff1,temp[i],np.linspace(0,1)))[30].reshape(-1)
    temp=new_points
    for i in range(temp.shape[0]):
        new_points[i,:] = np.array(odeint(diff2,temp[i],np.linspace(0,1)))[30].reshape(-1)
    
    temp=new_points
    for i in range(temp.shape[0]):
        new_points[i,:] = np.array(odeint(diff3,temp[i],np.linspace(0,1)))[30].reshape(-1)

        
    return new_points


for i in range(0,500):
    print(i)
    temp=numpy.copy(points)
    temp=get_sol(num1[i],num2[i],num3[i],num4[i],num5[i],num6[i],num7[i],num8[i],num9[i],temp)
    temp[:,0]=temp[:,0]*size1[i]
    temp[:,1]=temp[:,1]*size2[i]
    temp[:,2]=temp[:,2]*size3[i]
    center=numpy.mean(temp,axis=0)
    temp=temp-center
    V=volume(temp,F)
    temp=temp/(V**(1/3))

    meshio.write_points_cells('bulbo_{}.stl'.format(100+i), temp.tolist(),[("triangle", your_mesh.cells_dict["triangle"])])

def get_sol(a,b,c,points):    
    def diff(x,t):
        return (c*x[2],a,b)
    
    return np.array(odeint(diff,points,np.linspace(0,3)))
