#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File name: utls.py
# Description: generate weights based on the distances betweent the pixel and the keypoints
# Created by: Shuang Li(sli@informatik.uni-hamburg.de)
# Last Modified:11/03/20


import os
import numpy as np
import multiprocessing as mp


def get_hand_weights(f):
    key_points = np.load(f[0])
    key_points[key_points > 95] = 95
    distance_array = np.array([])
    for i in range(96):
        for j in range(96):
            distance = np.linalg.norm(np.array([j, i]) * np.ones_like(key_points) - key_points, axis=1)
            distance_array = np.hstack([distance_array, distance.min()])
    distance_array = 1 - distance_array / distance_array.max()
    np.save(f[1], distance_array.reshape(96, 96))


cores = mp.cpu_count()
pool = mp.Pool(processes=cores)
uv_files = os.listdir('nowrist_neighbour_uv')
fl = []
for uv in uv_files:
    fl.append([
        '../data/shadow/201809_shadow_data400K/nowrist_neighbour_uv/{}'.format(uv),
        '../data/shadow/201809_shadow_data400K/weight_mask_uv/{}'.format(uv),
    ])
pool.map(get_hand_weights, fl)
