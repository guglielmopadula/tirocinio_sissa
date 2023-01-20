import numpy as np
import os

def load(name, folder, cut=None):
    with open(folder+name, 'r') as file:
        with open(folder+"postprocessed_"+name, 'w') as outfile:
            data = file.read()
            data = data.replace("             	", " ")
            data = data.replace("(", "")
            data = data.replace(")", "")
            outfile.write(data)
    loaded = np.array(np.loadtxt(folder+"postprocessed_"+name, skiprows=4))
    if cut is not None:
        print("loaded shape: ", loaded.shape, loaded[:cut].shape)
        return loaded[:cut]
    else:
        print("loaded shape: ", loaded.shape)
        return loaded

#archs = ["data", "AE", "AAE", "VAE", "BEGAN"]
archs = ["BEGAN"]

for arch in archs:
    lisf = []
    lism = []
    for i in range(100):
        print("test case ", i, "architecture: ", arch)
        fold = "./DTCHull_"+arch+"/"+str(i)+"/postProcessing/forces/"
        dirlist = os.listdir(fold)
        
        toStackf = []
        toStackm = []
        
        if len(dirlist)==1:
            lisf.append(load("force.dat", fold+dirlist[0]+"/", -1))
            lism.append(load("moment.dat", fold+dirlist[0]+"/", -1))
        
        elif len(dirlist)==2:
            cut = eval(dirlist[1])
            try:
                toStackf.append(load("force.dat", fold+dirlist[0]+"/", cut))
                toStackm.append(load("moment.dat", fold+dirlist[0]+"/", cut))
            except:
                print(dirlist[0])
                
            try:
                toStackf.append(load("force.dat", fold+dirlist[1]+"/", -1))
                toStackm.append(load("moment.dat", fold+dirlist[1]+"/", -1))
            except:
                print(dirlist[1])
        
            stackedf = np.vstack(toStackf)
            stackedm = np.vstack(toStackm)
            print("Stacked: ", stackedf.shape, stackedm.shape)
            
            lisf.append(stackedf)
            lism.append(stackedm)
        
        elif len(dirlist)==3:
            cut1 = eval(dirlist[1])
            try:
                toStackf.append(load("force.dat", fold+dirlist[0]+"/", cut1))
                toStackm.append(load("moment.dat", fold+dirlist[0]+"/", cut1))
            except:
                print(dirlist[0])
            
            cut2 = eval(dirlist[2])-eval(dirlist[1])
            try:
                toStackf.append(load("force.dat", fold+dirlist[1]+"/", cut2))
                toStackm.append(load("moment.dat", fold+dirlist[1]+"/", cut2))
            except:
                print(dirlist[0])
                
            try:
                toStackf.append(load("force.dat", fold+dirlist[2]+"/", -1))
                toStackm.append(load("moment.dat", fold+dirlist[2]+"/", -1))
            except:
                print(dirlist[1])
        
            stackedf = np.vstack(toStackf)
            stackedm = np.vstack(toStackm)
            print("Stacked: ", stackedf.shape, stackedm.shape)
            
            lisf.append(stackedf)
            lism.append(stackedm)

        print("list length ", len(lisf), len(lism), lisf[-1].shape)
        if lisf[-1].shape[0]!=3999: break
    
    matf = np.array(lisf)
    print(matf.shape)
    np.save("postProcessingForce_"+arch+".npy", matf)
    
    matm = np.array(lism)
    print(matm.shape)
    np.save("postProcessingMomentum_"+arch+".npy", matm)
