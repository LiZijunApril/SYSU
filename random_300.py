import os
import random
import shutil
import argparse
from pathlib import Path

# ori_dir = r'color_img_0'
# tar_dir = r'image0'

def movefile(ori_dir, tar_dir):
    path = os.listdir(ori_dir)
    sample1 = random.sample(path, 100) # random select 300 files
    for name in sample1:
        ori_dir_= os.path.join(ori_dir,name)
        tar_dir_= os.path.join(tar_dir,name)
        shutil.move(ori_dir_, tar_dir_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', required=True)
    parser.add_argument('--tar_dir', required=True)
    args = parser.parse_args()
    # movefile(ori_dir, tar_dir)]
    movefile(Path(args.ori_dir), Path(args.tar_dir))
