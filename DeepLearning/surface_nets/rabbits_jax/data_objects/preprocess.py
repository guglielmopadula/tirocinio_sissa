import numpy as np
import meshio
import scipy
import time
import skdim
from tqdm import tqdm
import sklearn
import pickle
def rref(A):
    M=A.copy()
    l=[]
    lead = 0
    rowCount = M.shape[0]
    columnCount = M.shape[1]
    for r in range(rowCount):
        if lead >= columnCount:
            return M
        i = r
        while M[i,lead] == 0:
            i += 1
            if i == rowCount:
                i = r
                lead += 1
                if columnCount == lead:
                    return M
        M[i],M[r] = M[r],M[i]
        M[r] = M[r]/M[r,lead]
        for i in range(rowCount):
            if i != r:
                M[i] = M[i]-M[r]*M[i,lead] 
        l.append(lead)
        lead += 1

    return M,l

def getinfo(stl):
    mesh=meshio.read(stl)
    points=mesh.points
    barycenter=np.mean(points,axis=0)
    return points,barycenter

    
class VolumePreservingEmbedding():
    def __init__(self,A,b):
        self.A=A
        self.b=b
        self.orig_dim=self.A.shape[1]
        self.echelon,self.pivots=rref(np.concatenate((A,-b.reshape(b.shape[0],1)),axis=1))
        self.pivots=np.array(self.pivots)
        self.npivots=np.array(list(set(np.arange(self.orig_dim).tolist()).difference(self.pivots)))
    def transform(self,x):
        return  np.delete(x, self.pivots, axis=0)

    def inverse_transform(self,x_red):
        tmp1=np.delete(self.echelon,self.pivots,axis=1)
        tmp2=np.concatenate((x_red,np.ones((1))))
        x_piv=-np.matmul(tmp1,tmp2)
        tmp3=np.zeros((self.orig_dim))
        tmp3[self.pivots]=x_piv
        tmp3[self.npivots]=x_red
        return tmp3
    
NUM_SAMPLES=200
tmp,barycenter=getinfo("rabbit_0.ply")
data=np.zeros((NUM_SAMPLES,*tmp.reshape(-1).shape))
for i in range(200):
    data[i]=getinfo("rabbit_{}.ply".format(i))[0].reshape(-1)
A=np.tile(np.eye(3),tmp.shape[0])/(tmp.shape[0])
vl=VolumePreservingEmbedding(A,barycenter)
tmparr=np.zeros((NUM_SAMPLES,vl.transform(data[0]).shape[0]))
for i in tqdm(range(NUM_SAMPLES)):
    A=np.tile(np.eye(3),tmp.shape[0])/(tmp.shape[0])
    vl=VolumePreservingEmbedding(A,barycenter)
    tmparr[i]=vl.transform(data[i])

pca=sklearn.decomposition.PCA()
pca.fit(tmparr)
reduced=np.argmin(abs(np.cumsum(pca.explained_variance_ratio_)-(1-1e-10)))
pca=sklearn.decomposition.PCA()
pca.fit(tmparr)
pca=sklearn.decomposition.PCA(n_components=reduced)
pca.fit(tmparr)
redarr=pca.transform(tmparr)
pickle.dump(redarr,open("data.npy", 'wb'))


