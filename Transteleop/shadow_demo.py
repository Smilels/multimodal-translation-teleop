from __future__ import print_function
from options.test_options import TestOptions
from models import create_model
import torch
import torch.utils.data
import numpy as np
import cv2
from util import seg_hand_depth

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from shadow_teleop.srv import *
from sr_robot_commander.sr_hand_commander import SrHandCommander


opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)  # regular setup: load and print networks; create schedulers


def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)
    img = cv2.resize(img, (opt.load_size, opt.load_size))
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.Tensor(img)
    if len(opt.gpu_ids) > 0:
        img = img.cuda()

    # human part
    model.set_input(img)
    model.test()
    joint_human = model.get_current_joints()
    # feature = model.get_current_feature().squeeze()[0]
    # cv2.imshow("feature", feature.cpu().data.numpy())
    # cv2.waitKey(1)
    return joint_human.cpu().data.numpy()[0]


class Teleoperation():
    def __init__(self):
        self.bridge = CvBridge()
        self.hand_commander = SrHandCommander(name="right_hand")
        rospy.sleep(1)

    def online_once(self):
        while True:
            img_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
            rospy.loginfo("Got an image ^_^")
            try:
                img = self.bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
            except CvBridgeError as e:
                rospy.logerr(e)

            try:
                # preproces
                img = seg_hand_depth(img, 500, 1000, 10, 100, 4, 4, 250, True, 300)
                img = img.astype(np.float32)
                img = img / 255. * 2. - 1

                n = cv2.resize(img, (0, 0), fx=2, fy=2)
                n = cv2.normalize(n, n, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("segmented human hand", n)
                cv2.waitKey(1)

                # get the clipped joints
                goal = tuple(self.joint_cal(img, isbio=True))
                hand_pos = self.hand_commander.get_joints_position()

                # first finger
                hand_pos.update({"rh_FFJ3": goal[3]})
                hand_pos.update({"rh_FFJ2": goal[4]})
                hand_pos.update({"rh_FFJ4": goal[2]})

                # middle finger
                hand_pos.update({"rh_MFJ3": goal[12]})
                hand_pos.update({"rh_MFJ2": goal[13]})

                # ring finger
                hand_pos.update({"rh_RFJ3": goal[16]})
                hand_pos.update({"rh_RFJ2": goal[17]})

                # little finger
                hand_pos.update({"rh_LFJ3": goal[8]})
                hand_pos.update({"rh_LFJ2": goal[9]})

                # thumb
                hand_pos.update({"rh_THJ5": goal[19]})
                hand_pos.update({"rh_THJ4": goal[20]})
                hand_pos.update({"rh_THJ3": goal[21]})
                hand_pos.update({"rh_THJ2": goal[22]})

                self.hand_commander.move_to_joint_value_target_unsafe(hand_pos, 0.3, False, angle_degrees=False)
                #rospy.sleep(0.5)

                rospy.loginfo("Next one please ---->")
            except:
                rospy.loginfo("no images")

    def joint_cal(self, img, isbio=False):
        # start = rospy.Time.now().to_sec()

        # run the model
        feature = test(model, img)
        # network_time = rospy.Time.now().to_sec() - start
        # print("network_time is ", network_time)

        joint = [0.0, 0.0]
        joint += feature.tolist()
        if isbio:
            joint[5] = 0.3498509706185152
            joint[10] = 0.3498509706185152
            joint[14] = 0.3498509706185152
            joint[18] = 0.3498509706185152
            joint[23] = 0.3498509706185152

        # joints crop
        joint[2] = self.clip(joint[2], 0.349, -0.349)
        joint[3] = self.clip(joint[3], 1.57, 0)
        joint[4] = self.clip(joint[4], 1.57, 0)
        joint[5] = self.clip(joint[5], 1.57, 0)

        joint[6] = self.clip(joint[6], 0.785, 0)

        joint[7] = self.clip(joint[7], 0.349, -0.349)
        joint[8] = self.clip(joint[8], 1.57, 0)
        joint[9] = self.clip(joint[9], 1.57, 0)
        joint[10] = self.clip(joint[10], 1.57, 0)

        joint[11] = self.clip(joint[11], 0.349, -0.349)
        joint[12] = self.clip(joint[12], 1.57, 0)
        joint[13] = self.clip(joint[13], 1.57, 0)
        joint[14] = self.clip(joint[14], 1.57, 0)

        joint[15] = self.clip(joint[15], 0.349, -0.349)
        joint[16] = self.clip(joint[16], 1.57, 0)
        joint[17] = self.clip(joint[17], 1.57, 0)
        joint[18] = self.clip(joint[18], 1.57, 0)

        joint[19] = self.clip(joint[19], 1.047, -1.047)
        joint[20] = self.clip(joint[20], 1.222, 0)
        joint[21] = self.clip(joint[21], 0.209, -0.209)
        joint[22] = self.clip(joint[22], 0.524, -0.524)
        joint[23] = self.clip(joint[23], 1.57, 0)

        return joint

    def clip(self, x, maxv=None, minv=None):
        if maxv is not None and x > maxv:
            x = maxv
        if minv is not None and x < minv:
            x = minv
        return x


def main():
    rospy.init_node('human_teleop_shadow')
    tele = Teleoperation()
    while not rospy.is_shutdown():
        tele.online_once()


if __name__ == "__main__":
    main()
