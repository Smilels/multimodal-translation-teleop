#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : img_cartesian2uvd
# Purpose : after get xyz of joint keypoints by feeding joint angles to shadow_fk.launch,
# draw these keypoints on uv images
# Creation Date : 10-02-2020
# Created By : Shuang Li

import numpy as np
import csv
import glob
import cv2
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

''' cv2 color: bgr, matplotliv color: rgb'''


vis_image = False


def surround(i, j, xl, yl, add=1):
    sur = []
    if i-add >= 0:
        sur.append([i-add, j])
    if j-add >= 0:
        sur.append([i, j-add])
    if i+add < xl:
        sur.append([i+add, j])
    if j+add < yl:
        sur.append([i, j+add])
    return sur


def inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add):
        for i, j in zip(x, y):
            sur = surround(i, j, w, l, add)
            for s in sur:
                xx, yy = s
                if gap < abs(img[xx, yy] - img[i, j]):
                    if zero_as_infty or abs(img[xx, yy] - img[i, j]) < thresh:
                        if img[xx, yy] > img[i, j]:
                            if img[i, j] <= fore_thresh:
                                mask[xx, yy] = 0
                                inner_edge.append((i, j))
                        else:
                            if img[xx, yy] <= fore_thresh:
                                mask[i, j] = 0
                                inner_edge.append((xx, yy))
        return inner_edge, mask


def seg_hand_depth(img, gap=100, thresh=500, padding=10, scale=10, add=5, box_z=250, zero_as_infty=False,
                   fore_p_thresh=300):
    img = img.astype(np.float32)
    if zero_as_infty:
        # TODO: for some sensor that maps infty as 0, we should override them
        thresh = np.inf
        his = np.histogram(img[img != 0])
        sum_p = 0
        for i in range(len(his[0])):
            sum_p += his[0][i]
            if his[0][i] == 0 and sum_p > fore_p_thresh:
                fore_thresh = his[1][i]
                break
        else:
            fore_thresh = np.inf
    else:
        fore_thresh = np.inf
    mask = np.ones_like(img)
    w, l = img.shape
    x = np.linspace(0, w-1, w//scale)
    y = np.linspace(0, l-1, l//scale)
    grid = np.meshgrid(x, y)
    x = grid[0].reshape(-1).astype(np.int32)
    y = grid[1].reshape(-1).astype(np.int32)
    if zero_as_infty:
        img[img == 0] = np.iinfo(np.uint16).max

    # morphlogy
    open_mask = np.zeros_like(img)
    open_mask[img != np.iinfo(np.uint16).max] = 1
    tmp = open_mask.copy()
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, np.ones((3, 3)))
    open_mask -= tmp
    img[open_mask.astype(np.bool)] = np.iinfo(np.uint16).max

    inner_edge = [(1,1)]
    inner_edge, mask = inner(inner_edge, img, zero_as_infty, fore_thresh, mask, gap, thresh, x, y, w, l, add)
    inner_edge = inner_edge[1:]

    mask = mask.astype(np.bool)
    edge_x, edge_y = np.where(mask == 0)
    x_min, x_max = np.min(edge_x), np.max(edge_x)
    y_min, y_max = np.min(edge_y), np.max(edge_y)

    x_min = max(0, x_min - padding)
    x_max = min(x_max + padding, w-1)
    y_min = max(0, y_min - padding)
    y_max = min(y_max + padding, l-1)
    if x_max - x_min > y_max - y_min:
        delta = (x_max - x_min) - (y_max - y_min)
        y_min -= delta/2
        y_max += delta/2
    else:
        delta = (y_max - y_min) - (x_max - x_min)
        x_min -= delta/2
        x_max += delta/2

    edge_depth = []
    for (x, y) in inner_edge:
        edge_depth.append(img[x, y])
    avg_depth = np.sum(edge_depth)/float(len(edge_depth))
    depth_min = max(avg_depth - box_z/2, 0)
    depth_max = avg_depth + box_z/2
    seg_area = img.copy()
    seg_area[seg_area < depth_min] = depth_min
    seg_area[seg_area > depth_max] = depth_max
    # normalized
    seg_area = ((seg_area - avg_depth) / (box_z/2)) # [-1, 1]
    seg_area = ((seg_area + 1)/2.) * 255.  # [0, 255]

    return seg_area


def main():
    base_path = "../data/"
    shadow_img = glob.glob("../data/paper_img/robot/*.png")
    DataFile = open(base_path + "joints_18k_xyz.csv", "r")
    lines = DataFile.read().splitlines()
    f_index = {}
    for ind, line in enumerate(shadow_img):
        f_index[line[-19:]] = ind

    # gazebo camera center coordinates and focal length
    mat = np.array([[554.255, 0, 320.5], [0, 554.25, 240.5], [0, 0, 1]])
    uv = np.zeros([16, 2])

    uv_file = open(base_path + "joints_18k_uv.csv", "w")
    writer = csv.writer(uv_file)

    for ln in lines:
        frame = ln.split(',')[0]
        print(frame)
        label_source = ln.split(',')[1:-1]
        keypoints = np.array([float(ll) for ll in label_source]).reshape(16, 3)

        # orientation: w, x,y,z in /gazebo/model_state
        camera_qt = np.array([0.707388269167, 0, 0, 0.706825181105])
        camera_tran = np.array([0, -0.5, 0.35])

        w, x, y, z = camera_qt
        quat = Quaternion(w, x, y, z)
        R = quat.rotation_matrix
        R = R.T
        camera_t = camera_tran
        result = [frame]
        for j in range(0, len(keypoints)):
            if j == 0:
                 keypoints[j][2] = keypoints[j][2] + 0.063
            key = keypoints[j] - camera_t
            cam_world = np.dot(R, key)
            cam_world = np.array([-cam_world[1], -cam_world[2], cam_world[0]])
            uv[j] = ((1 / cam_world[2]) * np.dot(mat, cam_world))[0:2]
            result.append(uv[j][0])
            result.append(uv[j][1])
            result.append(cam_world[2] * 1000)
        writer.writerow(result)

        if vis_image:
            plt.figure()
            # draw image
            img = cv2.imread(shadow_img[f_index[frame]], cv2.IMREAD_ANYDEPTH)
            img = img[100:400, 170:470]
            img = seg_hand_depth(img).astype(np.uint16)
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            plt.imshow(img)

            # draw points
            points = np.array(result[1:]).reshape(16, 3).astype(np.int16)
            # compensate for early crop
            points = points + np.array([-170, -100, 0])

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
            plt.savefig('../data/paper_img/save/' + frame)


if __name__ == '__main__':
    main()
