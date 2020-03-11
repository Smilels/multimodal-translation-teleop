import cv2
import numpy as np
import os

base_path = '/homeL/demo/ros_workspace/pr2_shadow_ws/src/TeachNet_Teleoperation/ros/src/shadow_teleop/data/Human_label/'
depth_path = 'human96_test/'
save_path = 'normal_map/'
img_list = os.listdir(base_path + depth_path)

for img in img_list:
    d_im = cv2.imread(base_path + depth_path + img, cv2.IMREAD_ANYDEPTH)

    # Also keep in mind that importing an 8-bit image as a depth map will
    #  no gives the same results than using directly the depth map matrix.
    d_im = d_im.astype(np.float32)
    zy, zx = np.gradient(d_im)
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    # ::-1 means flip these columns.
    cv2.imwrite(base_path + save_path + img[:-4] + '.jpg', normal[:, :, ::-1])
