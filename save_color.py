import os
import sys
import argparse
from pathlib import Path

import pickle
import numpy as np
import cv2 as cv
from loguru import logger
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField


def main(data_dir):
    height, width = 480, 640
    # clouds = sorted(data_dir.glob('*-cloud.pkl'))
    color_frames = sorted(data_dir.glob('*-color.pkl'))
    # depth_frames = sorted(data_dir.glob('*-depth.pkl'))
    logger.info('Found {} color frames'.format(len(color_frames)))
        # logger.info('Found {} clouds, {} color frames, {} depth frames'.format(len(clouds), len(color_frames), len(depth_frames)))
    # assert len(clouds) == len(color_frames)
    # assert len(clouds) == len(depth_frames)
    for idx, color_path in enumerate(color_frames):
            # for idx, (cloud_path, color_path, depth_path) in enumerate(zip(clouds, color_frames, depth_frames)):
        logger.info('#{}:'.format(idx))
        logger.info('#{}:'.format(color_path))
        # sizes = list(map(lambda x: x.stat().st_size/1024/1024, [color_path]))
                # sizes = list(map(lambda x: x.stat().st_size/1024/1024, [cloud_path, color_path, depth_path]))
        # logger.info('cloud {} MB, color {} MB, depth {} MB'.format(*sizes))

        # Load data files
        # cloud_msg = pickle.load(open(cloud_path, 'rb'))
        color_msg = pickle.load(open(color_path, 'rb'))
        # depth_msg = pickle.load(open(depth_path, 'rb'))

        # Extract data from ROS message format
        # points = np.array(list(pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        colors = [np.frombuffer(img.data, dtype=np.uint8).reshape(height, width, -1) for img in color_msg]
        # depths = [np.frombuffer(img.data, dtype=np.float16).reshape(height, width, -1) for img in depth_msg]
	

        # Concatenate color images
        logger.info('Color images')
        nrows, ncols = 1, 4

        """
        show = cv.vconcat([cv.hconcat([colors[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
        show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        cv.imshow('show color', show)

        # Concatenate depth images
        logger.info('Depth images')
        nrows, ncols = 1, 4
        depths = [(((depth - depth.min()) / (depth.max() - depth.min() + 1e-10)) * 255).astype(np.uint8) for depth in depths]
        for depth in depths:
            cv.applyColorMap(depth, cv.COLORMAP_JET)
        show = cv.vconcat([cv.hconcat([depths[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
        show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        cv.imshow('show depth', show)

        # Show point cloud information
        logger.info('Point cloud: {} points'.format(len(points)))
        """

        colors = [np.frombuffer(img.data, dtype=np.uint8).reshape(height, width, -1) for img in color_msg]
        # show = cv.vconcat([cv.hconcat([colors[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
        # show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        # for i in range(ncols):
        i = 3
        Img_Name = "color_img_out8_3/" + str(data_dir) + "-" + str(idx) + "_" + str(i) + ".jpg"
        cv.imwrite(Img_Name, colors[i])   
        key = cv.waitKey(10)
        if key == ord('q'):
            cv.destroyAllWindows()
            sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    main(Path(args.data_dir))
