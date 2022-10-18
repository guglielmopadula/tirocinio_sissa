#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:38:18 2022

@author: cyberguli
"""

import scipy
import numpy as np
from scipy.interpolate import BSpline
np.random.seed(0)



def MultiBSpline(t1,c1,t2,c2,t3,c3,k):
        spline1=BSpline(t1, c1, k)
        spline2=BSpline(t2, c2, k)
        spline3=BSpline(t3, c3, k)
        def function(x):
            return spline1(x[0])*spline2(x[1])*spline3(x[2])
        return function
        


class FFD():
    def __init__(self,box_origin,box_length,n_control,k=2,n=3):
        self.box_origin=np.array(box_origin)
        self.box_length=np.array(box_length)
        self.n_control=np.array(n_control, dtype=np.int64)
        self.n=n
        self.k=k
        self.generate_splines()
        
    def control_point(self,i,j,k):
        return self.box_origin+self.box_length/(self.n_control-1)*np.array([i,j,k])
    
    def apply_to_point(self,x):
        y=0
        for i in range(self.n_control[0]):
            for j in range(self.n_control[1]):
                for k in range(self.n_control[2]):
                    y=y+self.control_point(i,j,k)*self.splines[i,j,k](x)
        return y
        
        
    def generate_splines(self):
        maxlength=np.max(self.box_length)
        minbound=np.min(self.box_origin)
        maxbound=np.max(self.box_origin)+maxlength
        maxcontrol=np.max(self.n_control)
        lst=[]
        c=-0.0001+0.002*np.random.rand(maxcontrol,self.n)
        temp=np.random.rand(maxcontrol,self.n+self.k+1)
        temp.sort(axis=1)
        t=minbound+(maxbound-minbound)*temp
        for i in range(maxcontrol):
            for j in range(maxcontrol):
                for k in range(maxcontrol):
                    lst.append(MultiBSpline(t[i],c[i],t[j],c[j],t[k],c[k],self.k))
        
        self.splines=np.array(lst).reshape((maxcontrol,maxcontrol,maxcontrol))
        
    def check_point_in_box(self,x):
        return (x[0]<= self.box_origin[0]+self.box_length[0]) and (x[0]>=self.box_origin[0]) and (x[1]<= self.box_origin[1]+self.box_length[1]) and (x[1]>=self.box_origin[1]) and  (x[1]<= self.box_origin[2]+self.box_length[2]) and (x[2]>=self.box_origin[2])
        
        
ffd=FFD([0,0,0],[1,1,1],[5,5,5])
print(ffd.apply_to_point(np.array([0.8,0.8,0.8])))