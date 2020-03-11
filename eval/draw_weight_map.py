#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File name: utls.py
# Description: dataset check
# Created by: Shuang Li(sli@informatik.uni-hamburg.de)
# Last Modified:11/03/20


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# original depth images
base_path = "../data/uv_test/"
img = os.listdir(base_path)
npy_path = "../data/uv_npy/"
plt.figure()
for frame in img:
    plt.axis('off')
    pic = cv2.imread(base_path + frame, cv2.IMREAD_ANYDEPTH)
    img = cv2.normalize(pic, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    plt.imshow(img)
    # save readable depth images
    plt.savefig('../data/weight_img/depth/' + frame)
    pic = cv2.resize(pic, (96, 96))

    key_points = np.load(npy_path + frame[:-4] + '.npy')[1:]
    print(key_points.shape)
    distance_array = np.array([])
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            distance = np.linalg.norm(np.array([j, i]) * np.ones_like(key_points) - key_points, axis=1)
            distance_array = np.hstack([distance_array, distance.min()])
    distance_array = 1 - distance_array / distance_array.max()
    plt.matshow(distance_array.reshape(96, 96), cmap='OrRd')
    plt.axis('off')
    # plt.show()
    plt.savefig('../data/weight_img/' + frame)
