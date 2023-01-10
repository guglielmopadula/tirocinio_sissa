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
    mesh=meshio.read("../data_objects/hull_{}.stl".format(i))
    mesh.points=linreg.predict(mesh.points)
    mesh.write("data_rotated_{}.stl".format(i))
    print(i)