from tkinter import image_names
import cv2 as cv
import sys
import pyrealsense2 as rs
import pickle
import numpy as np

from loguru import logger

color_info = open('out2/1659970071.996776104-color.pkl', 'rb')
color_msg = pickle.load(color_info)
# f = open('out4/1659969548.509315729-cloud.pkl', 'rb')
# depth_info = open('out2/1659970071.996776104-depth.pkl', 'rb')
# depth_msg = pickle.load(depth_info)

# logger.info(dir(depth_msg[0]))
# colorizer = rs.colorizer()
# colorizer_depth = np.asanyarray(colorizer.colorize(depth_msg[0]).get_data())

# colorizer_depth = cv.resize(colorizer_depth, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
# cv.imshow('colorizer depth', colorizer_depth)

height, width = 480, 640
nrows, ncols = 1, 5

# Concatenate color images
logger.info('Color images')
colors = [np.frombuffer(img.data, dtype=np.uint8).reshape(height, width, -1) for img in color_msg]
# show = cv.vconcat([cv.hconcat([colors[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
# show = cv.vconcat([cv.hconcat([colors[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
# show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
cv.imshow('show color', colors[0])

for i in range(ncols):
    Img_Name = "color_img/" + str(i) + ".jpg"
    cv.imwrite(Img_Name, colors[i])
# print(len(colors))


# logger.info(depths[0].max())

# # Concatenate depth images
# logger.info('Depth images')

# depths = [np.frombuffer(img.data, dtype=np.float16).reshape(height, width, -1) for img in depth_msg]
# depths = np.array(depths)
# np.place(depths, depths>10,10)
# depths[np.isinf(depths)] = 10
# logger.info(depths.max())
# depths = [(((depth - 0) / (10 - 0)) * 255).astype(np.uint8) for depth in depths]
# # for depth in depths:
#     # show = cv.applyColorMap(depth, cv.COLORMAP_HSV)
# show = cv.vconcat([cv.hconcat([depths[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
# show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
# cv.imshow('show depth', show)

# show = cv.applyColorMap(show, cv.COLORMAP_RAINBOW)
# logger.info(show.astype)
# colorizer = rs.colorizer()
# colorizer_depth = colorizer.colorize(depth_msg)

# cv.imshow('show colorized depth', show)

# colorizer_depth = np.asanyarray(colorizer.colorize(depth_msg))
# cv.imshow('show colorizer_depth', colorizer_depth)

key = cv.waitKey()
if key == ord('q'):
    cv.destroyAllWindows()
    sys.exit(0)
# logger.info(depth_msg)

# f.close()
