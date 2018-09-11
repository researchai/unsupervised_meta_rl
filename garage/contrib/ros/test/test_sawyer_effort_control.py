"""Ask robot to chase block."""

import intera_interface
import rospy
import numpy as np


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
    limb.set_command_timeout(0.03)
    rate = rospy.Rate(33)
    while True:
        limb.set_joint_torques({'right_j0': np.random.uniform(-3, 3),
                                'right_j1': np.random.uniform(-10, 10),
                                'right_j2': np.random.uniform(-5, 5),
                                'right_j3': np.random.uniform(-5, 5),
                                'right_j4': np.random.uniform(-2.25, 2.25),
                                'right_j5': np.random.uniform(-2.25, 2.25),
                                'right_j6': np.random.uniform(-2.25, 2.25)})
        rate.sleep()


if __name__ == '__main__':
    run()
