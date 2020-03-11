#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval_error_bar.py
# Creation Date : 10-01-2020
# Created By : Shuang Li


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# single Shadow
shadow_output = pd.read_csv('../results/joint_acc.csv').values.astype(np.float32)
shadow_output = np.delete(shadow_output, (3, 8, 12, 16, 21), axis=1)
shadow_ave_loss = abs(shadow_output).mean()
shadow_joint_loss = abs(shadow_output).mean(axis=0)
shadow_loss = np.hstack([shadow_joint_loss, shadow_ave_loss]).tolist()

# stn
teach_output = pd.read_csv('../results/joint_acc.csv').values
teach_output = np.delete(teach_output, (3, 8, 12, 16, 21), axis=1)
teach_ave_loss = abs(teach_output).mean()
teach_joint_loss = abs(teach_output).mean(axis=0)
teach_loss = np.hstack([teach_joint_loss, teach_ave_loss]).tolist()

# new ae branch
ae_output = pd.read_csv('../results/joint_acc.csv').values.astype(np.float32)
ae_output = np.delete(ae_output, (3, 8, 12, 16, 21), axis=1)
ae_ave_loss = abs(ae_output).mean()
ae_joint_loss = abs(ae_output).mean(axis=0)
ae_loss=np.hstack([ae_joint_loss, ae_ave_loss]).tolist()

# gan
gan_output = pd.read_csv('../results/joint_acc.csv').values.astype(np.float32)
gan_output = np.delete(gan_output, (3, 8, 12, 16, 21), axis=1)
gan_output_ave_loss = abs(gan_output).mean()
gan_output_joint_loss = abs(gan_output).mean(axis=0)
gan_output_loss=np.hstack([gan_output_joint_loss, gan_output_ave_loss]).tolist()

# # stn
stn_output = pd.read_csv('../results//joint_acc.csv').values.astype(np.float32)
stn_output = np.delete(stn_output, (3, 8, 12, 16, 21), axis=1)
stn_output_ave_loss = abs(stn_output).mean()
stn_output_joint_loss = abs(stn_output).mean(axis=0)
stn_output_loss=np.hstack([stn_output_joint_loss, stn_output_ave_loss]).tolist()


name_list = ['F4', 'F3', 'F2',
             'L5', 'L4', 'L3', 'L2',
             'M4', 'M3', 'M2',
             'R4', 'R3', 'R2',
             'T5', 'T4', 'T3', 'T2', 'Ave']

x = list(range(len(name_list)))
total_width, n = 1.1, 6
width = total_width / n

clrs = sns.color_palette("muted", 10)
with sns.axes_style('darkgrid'):
    fig = plt.figure()
    fig.set_size_inches(10, 5)

    plt.bar(x, shadow_loss, width=width,  label='Robotonly', fc=clrs[0])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, ae_loss, width=width, label='Transteleop', fc=clrs[3])
    for i in range(len(x)):
         x[i] = x[i] + width
    plt.bar(x, gan_output_loss, width=width,  label='GANteleop', fc=clrs[2], tick_label=name_list)
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, teach_loss, width=width,  label='TeachNet', fc=clrs[4])
    for i in range(len(x)):
         x[i] = x[i] + width
    plt.bar(x, stn_output_loss, width=width,  label='Transtelop, w/o STN', fc=clrs[5])

    ax = plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    plt.yticks(np.arange(0, 0.081, 0.01))
    plt.margins(x=0)
    plt.ylabel('Absolute Mean Angle Error (rad)', fontsize=16)
    # plt.legend(prop={'size': 16})
    fig.savefig('error_bar.pdf')
