#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File name: result_vis.py
# Description: visualize all realA, realsB and generated images.
# Created by: Shuang Li(sli@informatik.uni-hamburg.de)
# Created on: 02/04/20

import cv2
import numpy as np
import glob
import sys
import os

model_name = sys.argv[1]
load_epoch = sys.argv[2]
if len(sys.argv) > 3:
    com_epoch = sys.argv[3]
    path_com = os.path.join('../pytorch-CycleGAN-and-pix2pix/results', model_name, 'test_' + com_epoch, 'images/', )
path = os.path.join('../pytorch-CycleGAN-and-pix2pix/results', model_name, 'test_' + load_epoch, 'images', )
img_list = glob.glob(path + '/*fake_B.png')

print(path)
for img in img_list:
    print(img[-26:-11])
    fakeb = cv2.imread(img, cv2.IMREAD_ANYCOLOR)
    realb = cv2.imread(img[:-10] + 'real_B.png', cv2.IMREAD_ANYCOLOR)
    reala = cv2.imread(img[:-10] + 'real_A.png', cv2.IMREAD_ANYCOLOR)
    a = np.hstack([reala, realb, fakeb])
    if len(sys.argv) > 3:
        img_com = path_com + img[-26:-10] + 'fake_B.png'
        fakeb = cv2.imread(img_com, cv2.IMREAD_ANYCOLOR)
        realb = cv2.imread(img_com[:-10] + 'real_B.png', cv2.IMREAD_ANYCOLOR)
        reala = cv2.imread(img_com[:-10] + 'real_A.png', cv2.IMREAD_ANYCOLOR)
        b = np.hstack([reala, realb, fakeb])
        a = np.vstack([a, b])
    a = cv2.resize(a, None, fx=4, fy=4)
    cv2.imshow('result', a)
    cv2.waitKey(0)
