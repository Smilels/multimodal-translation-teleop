#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval.py
# Purpose :
# Last Modified:02/06/20
# Created By :Shuang Li


import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# avoid type3 font
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def cal_acc(predict, threshold, strategy='avg'):
    if strategy == 'avg':
        return np.sum(np.mean(np.abs(predict * 1000), axis=-1) < threshold).astype(np.float32)/len(predict)
    elif strategy == 'max':
        return np.sum(np.max(np.abs(predict * 1000), axis=-1) < threshold).astype(np.float32)/len(predict)
    else:
        raise NotImplementedError


def main():
    ae = pd.read_csv('../results/xyz/ae_test_xyz.csv').values[:, 1:-1].astype(np.float32)
    stn = pd.read_csv('../results/xyz/stn_test_xyz.csv').values[:, 1:-1].astype(np.float32)
    teach = pd.read_csv('../results/teach_dis_hongzhuo.csv').values.astype(np.float32)
    shadow = pd.read_csv('../results/hongzhuo_shadow_4.csv').values.astype(np.float32)
    gan = pd.read_csv('../results/xyz/gan_test_xyz.csv').values[:, 1:-1].astype(np.float32)
    label_xyz = pd.read_csv('../results/xyz/label_xyz.csv').values[:, 1:-1].astype(np.float32)
    # embed()
    clrs = sns.color_palette("muted", 10)
    with sns.axes_style("dark"):
        fig = plt.figure()
        fig.set_size_inches(5, 5)

        acc1 = []
        acc2 = []
        acc4 = []
        acc5 = []
        acc6 = []

        threshold = []
        for thresh in np.arange(0., 20., 0.25):
            acc1.append(cal_acc(ae-label_xyz, thresh, 'max'))
            acc2.append(cal_acc(teach-label_xyz, thresh, 'max'))
            acc4.append(cal_acc(shadow-label_xyz, thresh, 'max'))
            acc5.append(cal_acc(gan-label_xyz, thresh, 'max'))
            acc6.append(cal_acc(stn-label_xyz, thresh, 'max'))
            threshold.append(thresh)
        plt.xlim(0, 0.4)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0., 20.1, 2.))
        plt.yticks(np.arange(0., 1.1, 0.1))
        plt.plot(threshold, acc4, color=clrs[0], label='Robotonly')
        plt.plot(threshold, acc1, color=clrs[3], label='Transteleop')
        plt.plot(threshold, acc5, color=clrs[2], label='GANteleop')
        plt.plot(threshold, acc2, color=clrs[4], label='TeachNet')
        plt.plot(threshold, acc6, color=clrs[5], label='Transtelop, w/o STN')
        plt.grid(True)
        # plt.title('Max')
        plt.xlabel('Distance Threshold e (mm)', fontsize=16)
        plt.ylabel('Fraction of Frames with Max Error < e', fontsize=16)
        # plt.legend(prop={'size': 16})
        # plt.show()
        fig.savefig('dis_eval.pdf')


if __name__ == '__main__':
    main()
