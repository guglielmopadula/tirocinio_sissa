#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:34:14 2022

@author: cyberguli
"""

import meshio
import numpy as np

mesh=meshio.read("Stanford_Bunny.stl")
temp=mesh.points
triangles=mesh.cells_dict['triangle']
temp[:,0]=temp[:,0]-np.min(temp[:,0])
temp[:,1]=temp[:,1]-np.min(temp[:,1])
temp[:,2]=temp[:,2]-np.min(temp[:,2])
temp=temp/np.max(temp)
meshio.write_points_cells("bunny.stl", temp, [("triangle", triangles)])
