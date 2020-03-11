#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name :  crop_shadow_depth.py
# Purpose : This is for crop shadow image into 96*96 (normalized to [0,255))
# Last Modified:01/17/20
# Created By :Shuang Li


import os
import cv2
import multiprocessing as mp
import numpy as np


def crop_shadow(f):
    depth = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH)
    if depth is None:
        print(f[0], ' empty!')
        return
    if np.max(depth) == np.min(depth) == 0:
        print(f[0], 'bad!')
        return
    depth = depth[100:400, 170:470]
    depth[np.where(depth == 700)] = 0
    depth_resize = cv2.resize(depth, (96, 96), interpolation=cv2.INTER_NEAREST)
    depth_resize_norm = ((depth_resize - np.min(depth_resize)).astype(float) / (
                np.max(depth_resize) - np.min(depth_resize)) * 255).astype(np.uint16)
    cv2.imwrite(f[1], depth_resize_norm)


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    base_path = "../data/shadow/depth_shadow/"
    save_path = "../data/shadow/shadow96_norm/"
    depth_files = os.listdir(base_path)
    print('len of shadow images is: ', len(depth_files))
    fl = []
    for frame in depth_files:
        fl.append([base_path + frame, save_path + frame])
    pool.map(crop_shadow, fl)


if __name__ == '__main__':
    main()
