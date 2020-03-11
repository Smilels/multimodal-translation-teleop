#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File name: utls.py
# Description: dataset check
# Created by: Shuang Li(sli@informatik.uni-hamburg.de)
# Last Modified:01/01/20


import os
from shutil import move, copyfile
import numpy as np
import copy


def move_data_train_to_test():
    ori_folder = "../data/com_humannorm_nobase/train/"
    test_folder = "../data/com_humannorm_nobase/test/"
    # test_folder = "/home/liang/code/shadow_teleop/datasets/sli_test/"
    img_list = os.listdir(ori_folder)
    test_num = int(len(img_list) * 0.2)
    print(test_num)
    for i in range(test_num):
        move(ori_folder + img_list[i], test_folder + img_list[i])
        # copyfile(ori_folder+img_list[i], test_folder+img_list[i])


def find_nocon_data():
    # find which img are not exited in human96_clean folder but have shadow joints file
    # if we want to calculate which human images do not have corresponding shadow joint,
    # when change following two for loops inversely

    shadow_file = np.load("shadow_joints_file_1812.npy")
    human_img_list = os.listdir('human96_clean')

    f_index = {}
    for ind, line in enumerate(human_img_list):
        f_index[line] = ind

    noimg_list = []
    for i in shadow_file[:, 0]:
        try:
            # utf-8 is used here
            # because content in shadow_file[:, 0] is bytes string, not normal string

            line = f_index[i.decode("utf-8")]
        except:
            noimg_list += [i.decode("utf-8")]
            print(i)

    np.save('shadowjoint_nohuman_img.npy', np.array(noimg_list))


def delete_unpaired_data_traintest_numpy():
    train_old = np.load('shadow_joints_file_train.npy')
    test = np.load('shadow_joints_file_test.npy')
    noimg_array = np.load('shadowjoint_nohuman_img.npy')

    delete_index = []
    for i, tmp in enumerate(train_old[:, 0]):
        if tmp in noimg_array:
            delete_index += [i]
    train = np.delete(train_old, delete_index, 0)

    delete_index = []
    for i, tmp in enumerate(test[:, 0]):
        if tmp in noimg_array:
            delete_index += [i]
    test = np.delete(test, delete_index, 0)
    np.save('human_img_shadow_joints_file_train', train)
    np.save('human_img_shadow_joints_file_test', test)


def delete_wrong_images():
    shadow_file = np.load("../stop_images.npy")
    train = os.listdir('train')
    test = os.listdir('test')
    print(len(train))
    print(len(test))

    f_index = {}
    for ind, line in enumerate(shadow_file):
        f_index[line] = ind

    for i, ii, iii in zip(train, train[1:], train[2:]):
        try:
            # delete before and after imges
            line = f_index[ii]
            os.remove('train/' + i)
            os.remove('train/' + ii)
            os.remove('train/' + iii)
        except:
            pass

    for i, ii, iii in zip(test, test[1:], test[2:]):
        try:
            line = f_index[ii]
            os.remove('test/' + i)
            os.remove('test/' + ii)
            os.remove('test/' + iii)
        except:
            pass
    train = os.listdir('train')
    test = os.listdir('test')
    print(len(train))
    print(len(test))


def delete_wrong_label():
    shadow_file = np.load("../../stop_images.npy")
    train = np.load('shadow_joints_file_train_newdata.npy')
    test = np.load('shadow_joints_file_test.npy')
    print(len(train))
    print(len(test))

    f_index = {}
    for ind, line in enumerate(shadow_file):
        f_index[line] = ind

    delete_index = []
    for i, tmp in enumerate(train[:, 0]):
        try:
            # delete before and after imges
            line = f_index[tmp]
            delete_index += [i - 1]
            delete_index += [i]
            delete_index += [i + 1]
        except:
            pass

    tt = np.delete(train, delete_index, 0)
    np.save('shadow_joints_file_train_0204', tt)

    delete_index = []
    for i, tmp in enumerate(test[:, 0]):
        try:
            # delete before and after imges
            line = f_index[tmp]
            delete_index += [i - 1]
            delete_index += [i]
            delete_index += [i + 1]
        except:
            pass

    ttt = np.delete(test, delete_index, 0)
    np.save('shadow_joints_file_test_0204', ttt)
    print(len(tt))
    print(len(ttt))


def save_ganimage_label():
    import glob
    train = np.load('../../human/groundtruth/shadow_joints_file_train_0204.npy')
    test = np.load('../../human/groundtruth/shadow_joints_file_test_0204.npy')
    img_list = glob.glob('test_latest/images/*fake_B.png')

    print(len(train))
    print(len(test))

    f_index = {}
    for ind, line in enumerate(img_list):
        f_index[line[-26:-11] + '.png'] = ind

    gan_label = []
    for i, tmp in enumerate(train):
        try:
            # delete before and after imges
            line = f_index[tmp[0]]
            gan_label += [tmp]
        except:
            pass

    print(len(gan_label))

    for i, tmp in enumerate(test):
        try:
            # delete before and after imges
            line = f_index[tmp[0]]
            gan_label += [tmp]
        except:
            pass

    gan_label_np = np.array(gan_label)
    np.save('shadow_joints_gan_test_label', gan_label_np)
    print(gan_label_np.shape)


# generate 8 neighbour pixels of each keypoint
def neighbour_uv_record():
    uv_files = os.listdir('crop_label')
    for f in uv_files:
        uvd = np.load('crop_label/' + f)
        uv = np.delete(uvd, (2), axis=1)
        uv = np.delete(uv, (0), axis=0)
        a1 = copy.deepcopy(uv)
        a1[:, 0] -= 1
        a2 = copy.deepcopy(uv)
        a2[:, 0] += 1
        a3 = copy.deepcopy(uv)
        a3[:, 1] -= 1
        a4 = copy.deepcopy(uv)
        a4[:, 1] += 1
        a5 = copy.deepcopy(uv)
        a5[:, 0] -= 1
        a5[:, 1] += 1
        a6 = copy.deepcopy(uv)
        a6[:, 0] -= 1
        a6[:, 1] -= 1
        a7 = copy.deepcopy(uv)
        a7[:, 0] += 1
        a7[:, 1] += 1
        a8 = copy.deepcopy(uv)
        a8[:, 0] += 1
        a8[:, 1] -= 1
        t = np.vstack([uv, a1, a2, a3, a4, a5, a6, a7, a8])
        np.save('nowrist_neighbour_uv/' + f, t)
