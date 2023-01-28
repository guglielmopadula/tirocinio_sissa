#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:36:27 2023

@author: cyberguli
"""

import pickle
import meshio
filename = 'rotate_model.sav'
linreg = pickle.load(open(filename, 'rb'))
#pickle.dump(linreg, open(filename, 'wb'))
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
