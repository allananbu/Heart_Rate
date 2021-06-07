# -*- coding: utf-8 -*-
"""
Created on Sat May 15 00:12:39 2021

@author: Allan
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_record_to_file('Record_light_12.bag')
profile=pipeline.start(config)
#Set Laser
device = profile.get_device()
depth_sensor = device.query_sensors()[0]
set_laser = 0
depth_sensor.set_option(rs.option.laser_power, set_laser)
# Start streaming


e1 = cv2.getTickCount()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frame_no=frames.get_frame_number()
        #frame_time=frames.
        frame_time=frames.timestamp
#        depth_frame = frames.get_depth_frame()
#        depth_image = np.asanyarray(depth_frame.get_data())
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
#        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to
#8-bit per pixel first)
#        depth_colormap =cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)

        # Stack both images horizontally
#        images = np.hstack((color_image, ))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        print("frame number",frame_no)
        cv2.waitKey(1)
        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        if t>60: # change it to record what length of video you are interested in
            print("Done!")
            break

finally:

    # Stop streaming
    pipeline.stop()