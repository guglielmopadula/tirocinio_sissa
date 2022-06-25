#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 08:25:37 2022

@author: cyberguli
"""
ext1 = open("ext1.tet6", "w")
ext2 = open("ext2.tet6", "w")

 
with open("morph.tet6", "r") as a_file:
    
  for line in a_file:
    stripped_line = line.split()
    if len(stripped_line)==2:
        ext1.write(stripped_line[0]+" "+stripped_line[1]+"\n")
        ext2.write(stripped_line[0]+" "+stripped_line[1]+"\n")
    
    if len(stripped_line)==6:
        ext1.write(stripped_line[0]+" "+stripped_line[1]+" "+stripped_line[2]+" "+ stripped_line[0]+" "+stripped_line[1]+" "+stripped_line[2]+"\n")
        ext2.write(stripped_line[3]+" "+stripped_line[4]+" "+stripped_line[5]+" "+ stripped_line[3]+" "+stripped_line[4]+" "+stripped_line[5]+"\n")
        
    if len(stripped_line)==5:
        ext1.write(stripped_line[0]+" "+stripped_line[1]+" "+stripped_line[2]+" "+stripped_line[3]+" "+stripped_line[4]+"\n")
        ext2.write(stripped_line[0]+" "+stripped_line[1]+" "+stripped_line[2]+" "+stripped_line[3]+" "+stripped_line[4]+"\n")

ext1.close()
ext2.close()

