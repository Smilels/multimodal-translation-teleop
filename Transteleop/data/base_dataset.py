"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from __future__ import (absolute_import, division,print_function,unicode_literals)
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABCMeta, abstractmethod


class BaseDataset(data.Dataset):
    __metaclass__ = ABCMeta
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def cv2_transform(opt, img, is_a=False):
    img = img.astype(np.float32)
    if 'resize' in opt.preprocess:
        img = cv2.resize(img, (opt.load_size, opt.load_size))
    if is_a:
        if 'rotate' in opt.preprocess:
            angle = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D(((opt.load_size - 1) / 2.0, (opt.load_size - 1) / 2.0), angle, 1)
            img = cv2.warpAffine(img, M, (opt.load_size, opt.load_size), borderValue=255)

        if 'jitter' in opt.preprocess:
            min_img = np.min(img[img != 255.])
            max_img = np.max(img[img != 255.])
            delta = np.random.rand() * (255. - max_img + min_img) - min_img
            img[img != 255.] += delta
            img = img.clip(max=255., min=0.)

        if 'flip' in opt.preprocess:
            img = cv2.flip(img, -1)  # -1 is Flipped Horizontally & Vertically
            # 1 is Flipped Horizontally, 0 is Flipped Vertically

    # Normalized
    img = img / 255. * 2. - 1  # [-1, 1]
    img = img[np.newaxis, ...]
    img = torch.from_numpy(img)
    return img


def get_normalization(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
