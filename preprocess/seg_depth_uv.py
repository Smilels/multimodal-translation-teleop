#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Last Modified:04/01/20
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import multiprocessing as mp
from shutil import copyfile
import os
from seg import seg_hand_depth


##
# This is for recording bbx of segmented human image 96*96
##


def func_human(f):
    # copyfile(f[0], f[1])
    img = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f[0], ' empty!')
        return
    if np.max(img) == np.min(img) == 0:
        print(f[0], ' bad!')
        return
    img = img.astype(np.float32)
    try:
        _, output = seg_hand_depth(img, 500, 1000, 10, 96, 4, 4, 250, True, 300, before_norm=True)
        np.save(f[1][:-4] + '_crop2.npy', output)
    except:
        print(f[1], 'seg failed')
    # print(f[1])


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    # crop human depth images
    file_list = os.listdir('/home/liang/code/shadow_teleop/datasets/crop_human_all')
    fl = []
    for j in np.array(file_list):
        fl.append([
            '/home/liang/code/shadow_teleop/datasets/human96/{}'.format(j),
            '/home/liang/code/shadow_teleop/datasets/seg_bbx/{}'.format(j),
        ])
    pool.map(func_human, fl)


if __name__ == '__main__':
    main()
