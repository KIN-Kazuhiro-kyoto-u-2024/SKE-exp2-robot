#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


if __name__ == "__main__":
    rospy.init_node("keyboard_controller")

    pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pub_auto = rospy.Publisher("/auto_mode", Bool, queue_size=1)

    auto_mode_flag = False

    # 起動時に確実に出力するためのログ
    rospy.loginfo("Keyboard Controller Initialized.")
    print("\n-------------------------------------------")
    print("w/a/s/d: 手動操作 | r: 自動追従ON/OFF | q: 終了")
    print("-------------------------------------------\n")

    while not rospy.is_shutdown():
        key = get_key()
        twist = Twist()
        publish_vel = False

        if key == "w":
            twist.linear.x = 0.2
            publish_vel = True
        elif key == "s":
            twist.linear.x = -0.2
            publish_vel = True
        elif key == "a":
            twist.angular.z = 1.0
            publish_vel = True
        elif key == "d":
            twist.angular.z = -1.0
            publish_vel = True
        elif key == "r":
            auto_mode_flag = not auto_mode_flag
            pub_auto.publish(auto_mode_flag)

            # printだとroslaunchの出力に埋もれることがあるので、強制表示
            rospy.loginfo(f"AUTO MODE SWITCHED: {auto_mode_flag}")

            # トピックが届く時間を確保するため一瞬だけ待つ
            rospy.sleep(0.1)
        elif key == "q":
            break

        # w,a,s,dの時だけcmd_velを投げる（自動モードの邪魔をしないため）
        if publish_vel:
            pub_vel.publish(twist)
