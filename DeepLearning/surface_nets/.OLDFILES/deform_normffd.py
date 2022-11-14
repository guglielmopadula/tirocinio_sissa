from smithers.io import STLHandler
from pygem import FFD
import meshio
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
    points_old=mesh.points.astype(np.float32)
    points=points_old[points_old[:,2]>0]
    points_zero=points_old[points_old[:,2]>=0]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[mesh.points[:,2]>0].tolist()
        triangles=mesh.cells_dict['triangle'].astype(np.int64)
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        newmesh_indices_global_zero=np.arange(len(mesh.points))[mesh.points[:,2]>=0].tolist()
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[points_zero[:,2]>0].tolist()
        newtriangles_local_3=[]
        newtriangles_local_2=[]
        newtriangles_local_1=[]
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])


    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3



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



points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3=getinfo("hullpreprocessed.stl",True)

temp=points_old[points_old[:,2]>=0]
temp1=points_old[points_old[:,2]>0]
volume_const=volume(temp[newtriangles_zero])



ffd = FFD([3, 4, 4])
ffd.box_origin = np.array([0.01, 0.01, 0.01])
ffd.box_length = np.array([0.58, 0.12, 0.8])
tempx=ffd.array_mu_x
tempy=ffd.array_mu_y
tempz=ffd.array_mu_z
all_=np.zeros([500,606,3])
for i in range(500):
    param_x = np.random.uniform(-0.3, 0.5, [1,3,3])
    param_y = np.random.uniform(-0.3, 0.5, [1,3,3])
    param_z = np.random.uniform(-0.3, 0.5, [1,3,3])


    ffd.reset_weights()
    for a in range(2,3):
        for b in range(1,4):
            for c in range(1,4):
                ffd.array_mu_x[a,b,c] = tempx[a,b,c]+param_x[0][b-1][c-1]
                ffd.array_mu_y[a,b,c] = tempy[a,b,c]+param_y[0][b-1][c-1]
                ffd.array_mu_z[a,b,c] = tempz[a,b,c]+param_z[0][b-1][c-1]
    
    temp1_new = ffd(temp1)
    temp_new=temp.copy()
    points_new=points_old.copy()
    temp_new[newmesh_indices_local]=temp1_new
    a=(np.abs(np.linalg.det(temp_new[newtriangles_local_3])).sum()/6)
    b=(np.abs(np.linalg.det(temp_new[newtriangles_local_2])).sum()/6)
    c=(np.abs(np.linalg.det(temp_new[newtriangles_local_1])).sum()/6)
    d=-volume_const
    k=multi_cubic(a, b, c, d)
    temp1_new=temp1_new*k
    all_[i]=temp1_new
    points_new[points_new[:,2]>0]=temp1_new
    meshio.write_points_cells("hull_{}.stl".format(i), points_new, [("triangle", triangles)])
    
print(np.sum(np.var(all_,axis=0)))