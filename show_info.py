import pickle
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

color_info = open('out2/1659969550.718571663-color.pkl', 'rb')
color_msg = pickle.load(color_info)
logger.info('Color images')
# height, width = 480, 640
# colors = [np.frombuffer(img.data, dtype=np.uint8).reshape(height, width, -1) for img in color_msg]
# logger.info(colors)
logger.info(color_msg[0].header)
# cloud_info = open('out3/1659970837.490430832-cloud.pkl', 'rb')
# cloud_msg = pickle.load(open(cloud_path, 'rb'))
# points = np.array(list(pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)))

# def main(data_dir):
#     height, width = 480, 640
#     clouds = sorted(data_dir.glob('*-cloud.pkl'))
#     color_frames = sorted(data_dir.glob('*-color.pkl'))
#     depth_frames = sorted(data_dir.glob('*-depth.pkl'))
#     logger.info('Found {} clouds, {} color frames, {} depth frames'.format(len(clouds), len(color_frames), len(depth_frames)))
#     for idx, (cloud_path, color_path, depth_path) in enumerate(zip(clouds, color_frames, depth_frames)):
#         logger.info('#{}:'.format(idx))
#         logger.info('#{}:'.format(cloud_path))
#         logger.info('#{}:'.format(color_path))
#         logger.info('#{}:'.format(depth_path))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data-dir', required=True)
#     args = parser.parse_args()
#     main(Path(args.data_dir))