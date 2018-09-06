"""Ask robot to chase block."""

import copy
import sys

from geometry_msgs.msg import Pose
import moveit_commander
import intera_interface
import rospy

from garage.contrib.ros.worlds import BlockWorld

BLOCK_NAME = 'block_0'

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.140923828125,
    'right_j1': -1.2789248046875,
    'right_j2': -3.043166015625,
    'right_j3': -2.139623046875,
    'right_j4': -0.047607421875,
    'right_j5': -0.7052822265625,
    'right_j6': -1.4102060546875,
}


class ChaseBlock(object):
    def __init__(self, limb='right', hover_distance=0.0, tip_name="right_gripper_tip"):
        self._limb_name = limb
        self._limb = intera_interface.Limb(limb)
        self._limb.set_joint_position_speed(0.3)
        self._tip_name = tip_name
        self._gripper = intera_interface.Gripper()
        self._hover_distance = hover_distance

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)

    def approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self._limb.ik_request(approach, self._tip_name)
        self._guarded_move_to_joint_position(joint_angles)

    def _guarded_move_to_joint_position(self, joint_angles, timeout=60.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")


def run():
    rospy.init_node('set_joint_torques', anonymous=True)

    limb = intera_interface.Limb('right')

    limb.move_to_joint_positions({
        'right_j0': -0.140923828125,
        'right_j1': -1.2789248046875,
        'right_j2': -3.043166015625,
        'right_j3': -2.139623046875,
        'right_j4': -0.047607421875,
        'right_j5': -0.7052822265625,
        'right_j6': -1.4102060546875,
    })
    rospy.sleep(6)
    while True:
        limb.set_joint_torques({'right_j0': 2,
                                'right_j1': 2,
                                'right_j2': 3,
                                'right_j3': 4,
                                'right_j4': 5,
                                'right_j5': 6,
                                'right_j6': 7})


if __name__ == '__main__':
    run()
