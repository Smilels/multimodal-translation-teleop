#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : bighand_preprocess.py
# Purpose : Provides importer classes for importing data
# Last Modified:01/07/20
# Created By :Shuang Li

'''
this method cannot directly apply on bighand dataset because there are many images which the hand is not the closest object to the camera.
but is can be used on images from human96 which have roughly cropped by the groudtruth
'''
import scipy.io
import numpy as np
from PIL import Image
import os
import progressbar as pb
import struct
from basetypes import DepthFrame, NamedImgSequence
from handdetector import HandDetector
from transformations import transformPoints2D
from IPython import embed

basepath = '/homeL/demo/ros_workspace/pr2_shadow_ws/src/TeachNet_Teleoperation/ros/src/shadow_teleop/data/Human_label'
# the annotation txt file name
txtname = 'Training_Annotation'
# txtname = 'text_annotation'
# the depth image folder
# image_path = '/test_data'
image_path = '/human96_test'


class DepthImporter(object):
    def __init__(self):
        """
        Initialize object
        x: focal length in x direction
        fy: focal length in y direction
        ux: principal point in x direction
        uy: principal point in y direction
        """
        #
        # self.fx = 475.065948
        # self.fy = 475.065857
        # self.ux = 315.944855
        # self.uy = 245.287079
        self.fx = 554.255
        self.fy = 554.25
        self.ux = 320.5
        self.uy = 240.5
        self.depth_map_size = (640, 480)
        self.refineNet = None
        self.crop_joint_idx = 0
        self.numJoints = 21

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x4 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[4:7, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[4:7, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[7:10, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[7:10, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[10:13, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[10:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:16, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:16, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 160.) / 241.42 * depth
        col = (pts[:, 1] - 120.) / 241.42 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    def load_file(self):
        trainlabels = '{}/{}.txt'.format(basepath, txtname)
        inputfile = open(trainlabels)
        txt = 'Loading {}'.format(txtname)
        pbar = pb.ProgressBar(maxval=len(inputfile.readlines()), widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        inputfile.seek(0)

        data = []
        i = 0
        # adjust a proper cube size
        config = {'cube': (200, 200, 200)}
        for line in inputfile:
            frame = line.split(' ')[0].replace("\t", "")
            dptFileName = '{}/{}'.format(basepath + image_path, str(frame))

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1

            # loadDepthMap(dptFileName)
            img = Image.open(dptFileName)  # open image
            # img.show()
            assert len(img.getbands()) == 1  # ensure depth image
            dpt = np.asarray(img, np.float32)

            label_source = line.split('\t')[1:]
            label = []
            label.append([float(l.replace(" ", "")) for l in label_source[0:63]])

            # joints in image coordinates
            # this gt3Dorig is the keypoints of 640*480
            gt3Dorig = np.array(label).reshape(21, 3).astype(np.float32)

            # normalized joints in 3D coordinates
            gtorig = self.joints3DToImg(gt3Dorig)

            # print gt3D
            # self.showAnnotatedDepth(DepthFrame(dpt, gtorig, gtorig, 0, gt3Dorig, gt3Dcrop, 0, dptFileName, frame, ''))

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=None, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i += 1
                continue
            try:
                # we can use one value of uv of 96*96 as com
                dpt, M, com = hd.cropArea3D(com=None, size=config['cube'], dsize=(96, 96), docom=True)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                i += 1
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = transformPoints2D(gtorig, M)

            # print("{}".format(gt3Dorig))
            self.showAnnotatedDepth(DepthFrame(dpt, gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName, frame,
                                               'right', ''))

            data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName,
                                   frame, 'right', {}))

            # for train
            data0 = np.asarray(data[-1].dpt, 'float32')
            label0 = np.asarray(data[-1].gtorig, 'float32')
            h, w = data0.shape
            j, d = label0.shape
            imgStack = np.zeros((1, h, w), dtype='float32')  # stack_size,rows,cols
            labelStack = np.zeros(( j, d), dtype='float32')  # joints,dim
            if True: #norm to [-1,1]
                    imgD = np.asarray(data[-1].dpt.copy(), 'float32')
                    imgD[imgD == 0] = data[-1].com[2] + (config['cube'][2] / 2.)
                    imgD -= (data[-1].com[2] - (config['cube'][2] / 2.))
                    imgD /= config['cube'][2]
            else:
                    imgD = np.asarray(data[-1].dpt.copy(), 'float32')
                    imgD[imgD == 0] = data[-1].com[2] + (config['cube'][2] / 2.)
                    imgD -= data[-1].com[2]
                    imgD /= (config['cube'][2] / 2.)

            imgStack = imgD
            labelStack = np.asarray(data[-1].gt3Dcrop, dtype='float32') / (config['cube'][2] / 2.)

            pbar.update(i)
            i += 1

        inputfile.close()
        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

if __name__ == '__main__':
    importer = DepthImporter()
    importer.load_file()
