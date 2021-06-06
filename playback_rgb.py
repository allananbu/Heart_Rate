# -*- coding: utf-8 -*-
"""
Created on Sat May 15 00:15:55 2021

@author: Allan
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, "Record_resol_4.bag", repeat_playback=False)


images=[]
i = 0
#Haar Classifier
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mean_all=[]
frame_all=[]
# Start streaming
profile=pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

try:
    while True:
         # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number() #Get frame number
        frame_time=frames.get_timestamp() #Get timestamp
        frame_all.append(frame_time)
#        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        
        color_image = np.asanyarray(color_frame.get_data())
        #Convert color to gray for classifier

        print("frame number",frame_no)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        
#        color_all.append(color_frame)
#         Stack both images horizontally

        #print("frame no",frame_no)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
except RuntimeError:
    print("There are no more frames left in the .bag file!")

finally:

    # Stop streaming
    pipeline.stop()
