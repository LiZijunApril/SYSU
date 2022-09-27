#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
from pathlib import Path

import yaml
import cv2 as cv
import numpy as np

# import rospy
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header

# from fusion.msg import Pack

class Pack:
    def __init__(self):
        self.images = None
        self.depths = None


class Takeoff:
    def __init__(self, config):
        self.ready = False
        self.height = 480
        self.width = 640
        # self.sub = rospy.Subscriber('/pack', Pack, self.callback)
        for key, value in config.items():
            if key in ['rvec', 'tvec', 'camera_matrix', 'camera_distort']:
                value = np.array(value)
            setattr(self, key, value)
        self.camera_names = ['back', 'right', 'left', 'bottom', 'top']
        self.statistics = {'clear': 0, 'total': 0}
        self.max_depth = 5
        self.ready = True

    def __del__(self):
        print(self.statistics)

    def callback(self, msg, frame_idx):
        if not self.ready:
            print('This node is not ready')
            return

        images = [np.frombuffer(img.data, dtype=np.uint8).reshape(self.height, self.width, -1)
                  for img in msg.images]
        depths = [np.frombuffer(img.data, dtype=np.float16).reshape(self.height, self.width, -1)
                  for img in msg.depths]

        depths = [depth.copy() for depth in depths]
        for idx, _ in enumerate(depths):
            depths[idx][depths[idx] >= self.max_depth] = 0.0

        rois = [np.zeros(shape=depth.shape, dtype=bool) for depth in depths]
        for idx, roi in enumerate(rois):
            border = 20
            xmin, xmax = 0, (self.height//4)*2
            ymin, ymax = 0+border, self.width-border
            roi[xmin:xmax, ymin:ymax] = True

        for depth, roi in zip(depths, rois):
            depth[~roi] = 0
            # depth[depth < 0.52] = 0

        obstacles = np.array([False for _ in self.camera_names], dtype=bool)
        for idx, (name, depth) in enumerate(zip(self.camera_names, depths)):
            print(name)
            d = np.around(np.sort(np.unique(depth.reshape(-1))), 2)
            print(sorted(list(set(d))))
            if name in ['front', 'back']:
                obstacles[idx] = (rois[idx] & ((0.2 < depth) & (depth <= 2))).any()
            elif name in ['right', 'left']:
                obstacles[idx] = (rois[idx] & ((0.52 < depth) & (depth <= 2))).any()
            elif name in ['top']:
                obstacles[idx] = (rois[idx] & ((0.2 < depth) & (depth <= 2))).any()
            else:
                obstacles[idx] = False
        condition = [t for t in zip(self.camera_names, obstacles)]
        clear = ~(obstacles.any())
        print('#{:06d}: Clear: {} ({})'.format(frame_idx, clear, condition))

        nrows, ncols = 1, 5
        # drawing = [((depth / 10) * 255).astype(np.uint8) for depth in depths]
        drawing = [(((depth - depth.min()) / (depth.max() - depth.min() + 1)) * 255).astype(np.uint8) for depth in depths]
        for idx, _ in enumerate(drawing):
            drawing[idx] = cv.applyColorMap(drawing[idx], cv.COLORMAP_JET)
        show = cv.vconcat([cv.hconcat([drawing[r*ncols+c] for c in range(ncols)]) for r in range(nrows)])
        show = cv.resize(show, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        cv.putText(show, str(frame_idx), (10, 10), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=2)
        cv.imshow('show', show)

        self.statistics['clear'] += clear
        self.statistics['total'] += 1

        key = cv.waitKey(0)
        if key == ord('q'):
            cv.destroyAllWindows()
            sys.exit(0)


def main():
    # rospy.init_node('Takeoff')
    parser = argparse.ArgumentParser('A takeoff node')
    parser.add_argument('--config', default='config/fusion.yaml', help='The path to config file')
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader)
    takeoff = Takeoff(config)

    # rospy.loginfo('Ready...')
    # rospy.spin()

    data_dir = Path(args.data_dir)
    color_frames = sorted(data_dir.glob('*-color.pkl'))
    depth_frames = sorted(data_dir.glob('*-depth.pkl'))
    for idx, (color_path, depth_path) in enumerate(zip(color_frames, depth_frames)):
        if idx % 10 != 0:
            continue
        assert(color_path.stem[:20] == depth_path.stem[:20])
        color_msg = pickle.load(open(color_path, 'rb'))
        depth_msg = pickle.load(open(depth_path, 'rb'))
        msg = Pack()
        msg.images = color_msg
        msg.depths = depth_msg
        takeoff.callback(msg, idx)


if __name__ == '__main__':
    # try:
    #     main()
    # except rospy.ROSInterruptException:
    #     sys.exit(0)
    main()
