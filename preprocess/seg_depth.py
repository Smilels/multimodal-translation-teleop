#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Last Modified:04/01/20
# Created By :Shuang Li

import cv2
import numpy as np
import multiprocessing as mp
import os
import sys
from seg import seg_hand_depth


##
# This is for cropping human and shadow image into 100*100 (normalized to [0,255))
##


def func_shadow(f):
    img = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f[0], ' empty!')
        return
    if np.max(img) == np.min(img) == 255:
        print(f[0], ' img all 255!')
        return
    if np.max(img) == np.min(img) == 0:
        print(f[0], ' img all 0!')
        return
    crop_to_nobase = True
    if crop_to_nobase:
        img = img[100:400, 170:470]
    img = img.astype(np.float32)
    try:
        output, _ = seg_hand_depth(img)
        cv2.imwrite(f[1], output)
    except:
        print(f[1], ' seg failed')


def func_human(f):
    # copyfile(f[0], f[1])
    img = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH)
    if img is None:
        print(f[0], ' empty!')
        return
    if np.max(img) == np.min(img) == 255:
        print(f[0], ' img all 255')
        return
    if np.max(img) == np.min(img) == 0:
        print(f[0], ' img all 0')
        return
    img = img.astype(np.float32)
    try:
        output, _ = seg_hand_depth(img, 500, 1000, 10, 96, 4, 4, 250, True, 300, norm=True)
        cv2.imwrite(f[1], output)
    except:
        print(f[1], 'seg failed')
    # print(f[1])


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    data_type = sys.argv[1]
    # crop shadow depth images
    if data_type == "shadow":
        file_list = os.listdir('/home/liang/code/shadow_teleop/datasets/18-12-new-data/depth_shadow')
        fl = []
        for j in np.array(file_list):
            fl.append([
                '../datasets/18-12-new-data/depth_shadow/{}'.format(j),
                '../datasets/18-12-new-data/crop/shadow_nobase/{}'.format(j),
                # './data/origin/shadow{}/{}'.format(i, j),
            ])
        pool.map(func_shadow, fl)
    # crop human depth images
    elif data_type == "human":
        file_list = os.listdir('../data/human/human96')
        file_list.sort()
        fl = []
        for j in np.array(file_list[:600000]):
            fl.append([
                '../data/human/human96/{}'.format(j),
                '../data/human/human96_seg_201809/{}'.format(j),
            ])
        pool.map(func_human, fl)
    else:
        print("enter wrong data type")
        exit()


if __name__ == '__main__':
    main()
