#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 09:02:28 2022

@author: cyberguli
"""

with open("temp.txt", "r") as a_file:
    for line in a_file:
        strline=line.split()
        if "M2<->M1:" in strline:
            print(strline[strline.index("M2<->M1:")+1])
            
