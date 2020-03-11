#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : human_robot_combine_A_and_B.py
# Purpose : combine paired human-robot hand images
# Last Modified:04/01/20
# Created By :Shuang Li

import os
import numpy as np
import cv2
import argparse
import multiprocessing as mp


##
# This is for crop human and shadow image into 100*100 (normalized to [0,255))
##

def func_combine(f):
    im_A = cv2.imread(f[0], cv2.IMREAD_ANYDEPTH)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(f[1], cv2.IMREAD_ANYDEPTH)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    if im_A is None:
        print(f[0], ' im_A empty!')
        return
    if np.max(im_A) == np.min(im_A) == 0:
        print(f[0], ' im_A is all 0!')
        return
    if np.max(im_A) == np.min(im_A) == 225:
        print(f[0], ' im_A is all 255!')
        return
    if im_B is None:
        print(f[1], ' im_B empty!')
        return
    if np.max(im_B) == np.min(im_B) == 0:
        print(f[1], ' im_B is all 0!')
        return
    if np.max(im_B) == np.min(im_B) == 255:
        print(f[1], ' im_B is all 255!')
        return
    # assert (im_A.shape == im_B.shape)
    im_B = cv2.resize(im_B, (96, 96))
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(f[2], im_AB)


def main():
    parser = argparse.ArgumentParser('create image pairs')
    parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str,
                        default='../data/human/human96_seg_201809')
    parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str,
                        default='../data/shadow/201809_shadow_data400K/preprocess_shadow')
    parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str,
                        default='../data/com_201809/joints_img')
    parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))
    robot_img_list = os.listdir(args.fold_B)
    print('number of images = %d' % (len(robot_img_list)))
    img_fold_A = args.fold_A
    img_fold_B = args.fold_B
    img_fold_AB = args.fold_AB

    cores = mp.cpu_count()
    pool = mp.Pool(processes=cores)
    fl = []
    for n in range(len(robot_img_list)):
        # for n in range(10):
        name = robot_img_list[n]
        path_A = os.path.join(img_fold_A, name)
        path_B = os.path.join(img_fold_B, name)
        path_AB = os.path.join(img_fold_AB, name)
        fl.append([path_A, path_B, path_AB])
    # func_combine(fl[-1])
    pool.map(func_combine, fl)


if __name__ == '__main__':
    main()
