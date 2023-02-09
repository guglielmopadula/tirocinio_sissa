#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:36:27 2023

@author: cyberguli
"""
import numpy as np
import pickle
import meshio
from sklearn.linear_model import LinearRegression
import os
filename = 'rotate_model.sav'

if not os.path.isfile(filename):
    m1=meshio.read("../data_objects/hullmoved.stl")
    m2=meshio.read("../data_objects/hullhalved.stl")
    linreg=LinearRegression()
    linreg.fit(m1.points.reshape(-1,3),m2.points.reshape(-1,3))
    print(linreg.score(m1.points.reshape(-1,3),m2.points.reshape(-1,3)))
    pickle.dump(linreg, open(filename, 'wb'))





linreg = pickle.load(open(filename, 'rb'))
for i in range(100):
    #mesh=meshio.read("../inference_objects/AE_{}.stl".format(i))
    #mesh.points=linreg.predict(mesh.points)
    #mesh.write("AE_rotated_{}.stl".format(i))
    #mesh=meshio.read("../inference_objects/VAE_{}.stl".format(i))
    #mesh.points=linreg.predict(mesh.points)
    #mesh.write("VAE_rotated_{}.stl".format(i))
    #mesh=meshio.read("../inference_objects/AAE_{}.stl".format(i))
    #mesh.points=linreg.predict(mesh.points)
    #mesh.write("AAE_rotated_{}.stl".format(i))
    #mesh=meshio.read("../inference_objects/BEGAN_{}.stl".format(i))
    #mesh.points=linreg.predict(mesh.points)
    #mesh.write("BEGAN_rotated_{}.stl".format(i))
    mesh=meshio.read("../data_objects/hull_{}.stl".format(i))
    mesh.points=linreg.predict(mesh.points)
    mesh.write("hull_rotated_{}.stl".format(i))
    print(i)
