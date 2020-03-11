#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : seg_depth.py
# Purpose :
# Last Modified:01/04/20
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import multiprocessing as mp
from shutil import copyfile
import os
import sys
from seg import seg_hand_depth
from IPython import embed


##
# This is for saving preprocessing parameters and uv labels
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
    img = img[100:400, 170:470]
    img = img.astype(np.float32)
    try:
        output, label, _ = seg_hand_depth(img, label=f[1])
        np.save('{}.npy'.format(f[2][:-4]), label)
    except:
        print(f[2], ' seg failed')


def main():
    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    label = np.load('../data/com_201809/joints_18k_uv.npy')
    fl = []
    for j in label:
        uv = j[1:].reshape(16, 3)
        name = j[0]
        fl.append([
            '../data/shadow/201809_shadow_data400K/depth_shadow/{}'.format(name),
            uv,
            '../data/shadow/201809_shadow_data400K/crop_label/{}'.format(name)
        ])
    #    func_shadow(fl[-1])
    pool.map(func_shadow, fl)


if __name__ == '__main__':
    main()
