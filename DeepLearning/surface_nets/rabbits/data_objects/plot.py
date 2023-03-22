#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:44:07 2023

@author: cyberguli
"""

import open3d
import glob
import numpy as np
NUM_SAMPLES=10

def vis_dict(dict):
    pcds=sorted(glob.glob('./data_objects/rabbit_{}*.ply'.format(dict)))
    print(pcds)
    vis=open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    idx=0
    a=vis.get_view_control()
    pcd=open3d.io.read_point_cloud(pcds[idx])
    vis.add_geometry(pcd)
    a.rotate(0,-500)

    #vis.update_renderer()
        
    def right_click(vis):
        nonlocal idx
        print('right_click')
        if idx+1 in dict:
            idx=idx+1;
            vis.clear_geometries()
            pcd=open3d.io.read_point_cloud(pcds[idx])
            vis.add_geometry(pcd)
            a.rotate(0,-500)


    def left_click(vis):
        nonlocal idx
        print('left_click')
        if idx-1 in dict:
            idx=idx-1;
            vis.clear_geometries()
            pcd=open3d.io.read_point_cloud(pcds[idx])
            vis.add_geometry(pcd)
            a.rotate(0,-500)

        
        
    def exit_key(vis):
        vis.destroy_window()
        
    vis.register_key_callback(262,right_click)
    vis.register_key_callback(263,left_click)
    vis.register_key_callback(32,exit_key)
    vis.poll_events()
    vis.run()
    #vis.destroy_window()    
        
if __name__=='__main__':
    vis_dict(np.arange(NUM_SAMPLES).tolist())
    
