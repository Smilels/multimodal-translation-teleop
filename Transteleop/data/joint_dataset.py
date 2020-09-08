import os.path
from data.base_dataset import BaseDataset, cv2_transform
from data.image_folder import make_dataset
import cv2
import numpy as np
from IPython import embed


joint_upper_range = np.array([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                              1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                              1.571, 1.047, 1.222, 0.209, 0.524, 1.571]).astype(np.float32)
joint_lower_range = np.array([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                              -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0]).astype(np.float32)


class JointDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, 'joints_img')  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.uv_path = os.path.join(opt.dataroot, '../data/')
        if opt.phase == "train":
           self.label = np.load(os.path.join(opt.dataroot, 'train.npy'))
        elif opt.phase == "test":
            self.label = np.load(os.path.join(opt.dataroot, 'test.npy'))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            joints (numpy array) -- joint angle groundtruth of the robot hand
            uv (numpy array) -- uv of the 15 keypoints of the robot hand
        """
        # read a image given a random integer index
        tag = self.label[index]
        fname = tag[0]
        target = tag[3:].astype(np.float32)
        uv = np.load(os.path.join(self.uv_path, fname[:-4]+'.npy')).astype(np.float32)
        AB_path = os.path.join(self.dir_AB, fname)
        AB = cv2.imread(AB_path, cv2.IMREAD_ANYDEPTH)

        # split AB image into A and B
        h, w = AB.shape
        A = AB[0:h, 0:w/2]
        B = AB[0:h, w/2:w]

        A = cv2_transform(self.opt, A, is_a=True)
        B = cv2_transform(self.opt, B)
        uv = uv[np.newaxis, ...]

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'joints': target, 'uv': uv}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label)
