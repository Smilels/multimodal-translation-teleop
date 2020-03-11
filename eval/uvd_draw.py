#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : shadow cartesian2uvd
# Purpose : draw uv label on preprocessing images 96*96
# Creation Date : 21-02-2020
# Created By : Shuang Li

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from IPython import embed


def main():
    base_path = "../data/uv_test/"
    img = os.listdir(base_path)
    npy_path = "../data/uv_npy/"

    for frame in img:
        plt.figure()
        # draw image
        img = cv2.imread(base_path + frame, cv2.IMREAD_ANYDEPTH)
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        plt.imshow(img)

        # draw points
        # embed()
        point = np.load(npy_path + frame[:-4] + '.npy').astype(np.int16)
        for i in range(9):
            points = point[i*16:(i+1)*16, :]
            # palm
            x = [points[0][0]]
            y = [points[0][1]]
            plt.scatter(x, y, s=60, c='green')

            # tf
            x = [points[1][0], points[6][0], points[11][0], points[0][0]]
            y = [points[1][1], points[6][1], points[11][1], points[0][1]]
            plt.scatter(x[0:-1], y[0:-1], s=25, c='cyan')
            plt.plot(x, y, 'c', linewidth=1)

            # ff
            x = [points[2][0], points[7][0], points[12][0], points[0][0]]
            y = [points[2][1], points[7][1], points[12][1], points[0][1]]
            plt.scatter(x[0:-1], y[0:-1], s=25, c='blue')
            plt.plot(x, y, 'b', linewidth=1)

            # mf
            x = [points[3][0], points[8][0], points[13][0], points[0][0]]
            y = [points[3][1], points[8][1], points[13][1], points[0][1]]
            plt.scatter(x[0:-1], y[0:-1], s=25, c='red')
            plt.plot(x, y, 'r', linewidth=1)

            # rf
            x = [points[4][0], points[9][0], points[14][0], points[0][0]]
            y = [points[4][1], points[9][1], points[14][1], points[0][1]]
            plt.scatter(x[0:-1], y[0:-1], s=25, c='yellow')
            plt.plot(x, y, 'y', linewidth=1)

            # lf
            x = [points[5][0], points[10][0], points[15][0], points[0][0]]
            y = [points[5][1], points[10][1], points[15][1], points[0][1]]
            plt.scatter(x[0:-1], y[0:-1], s=25, c='magenta')
            plt.plot(x, y, 'm', linewidth=1.5)

            plt.axis('off')
        # plt.show()
        plt.savefig('../data/img/' + frame)


if __name__ == '__main__':
    main()
