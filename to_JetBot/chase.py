#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import rospy
from config import GameConfig
from geometry_msgs.msg import Twist
from mpc_for_real import MPCController
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MainController:
    def __init__(self):

        # 位置情報の格納 (番号は ArUco marker の ID)
        self.poses = {1: None, 2: None}
        self.is_stopped = False
        self.auto_mode = False  # デフォルトは手動モード（勝手に動かない）

        # 自分の ArUco marker の ID
        self.myID = 1

        # # ONNX モデルの読み込み
        # # 1. このプログラムファイル（launch_onnx.py）があるフォルダの絶対パスを取得
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # # 2. そのフォルダのパスと、ONNXファイルの名前を正しく結合する
        # onnx_path = os.path.join(current_dir, "residual_ppo_jetbot_dim=11.onnx")  # model.onnxの部分は実際のファイル名に変えてください
        # self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        # self.input_name = self.session.get_inputs()[0].name

        # パブリッシャの設定
        self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pub_debug_img = rospy.Publisher("/aruco/debug_image2", Image, queue_size=1)

        # サブスクライバの設定
        self.sub_marker = rospy.Subscriber(
            "/aruco/marker_json", String, self.marker_callback
        )
        self.sub_auto = rospy.Subscriber("/auto_mode", Bool, self.auto_mode_callback)

        # ゲームコンフィグの読み込み
        self.cfg = GameConfig()

        # MPC コントローラ
        self.MPCctr = MPCController(self.cfg)

        rospy.loginfo("=======================================================")
        rospy.loginfo("  Main Control Node Started [SIMPLE ON/OFF CONTROL MODE]")
        rospy.loginfo("=======================================================")

    def auto_mode_callback(self, msg):
        self.auto_mode = msg.data
        rospy.loginfo(f"[MODE] auto_mode changed to: {self.auto_mode}")

    def marker_callback(self, msg):
        try:
            raw_data = msg.data

            if raw_data.startswith('"') and raw_data.endswith('"'):
                raw_data = raw_data[1:-1]

            cleaned_data = raw_data.replace('\\"', '"').replace("\\\\", "\\")
            data = json.loads(cleaned_data)

            if isinstance(data, str):
                data = json.loads(data)

            self.is_stopped = data.get("stop", False)
            markers = data.get("markers", [])

            for marker in markers:
                m_id = marker.get("id")
                try:
                    actual_id = int(m_id)
                except (ValueError, TypeError):
                    actual_id = m_id

                if actual_id in [1, 2]:
                    self.poses[actual_id] = {
                        "x": float(marker.get("x")),
                        "y": float(marker.get("y")),
                        "theta": float(marker.get("theta")),
                    }

        except json.JSONDecodeError as je:
            rospy.logerr(f"[ERR] JSON Decode Failed: {je}")
        except Exception as e:
            rospy.logerr(f"[ERR] Unexpected error in callback: {e}")

    def publish_debug_image(self, distance=0.0, angle_diff=0.0, current_theta=0.0):
        height, width = 600, 600
        img = np.zeros((height, width, 3), np.uint8)
        scale = 0.4
        offset_x, offset_y = 300, 300

        def to_pixel(x, y):
            return (int(x * scale + offset_x), int(y * scale + offset_y))

        cv2.drawMarker(img, (offset_x, offset_y), (50, 50, 50), cv2.MARKER_CROSS, 20, 1)

        if self.poses[1] and self.poses[2]:
            p1 = to_pixel(self.poses[1]["x"], self.poses[1]["y"])
            p2 = to_pixel(self.poses[2]["x"], self.poses[2]["y"])

            cv2.circle(img, p1, 15, (255, 0, 0), -1)
            cv2.circle(img, p2, 15, (0, 0, 255), -1)

            arrow_len = 40
            head_x = int(p1[0] + arrow_len * math.cos(current_theta))
            head_y = int(p1[1] + arrow_len * math.sin(current_theta))
            cv2.arrowedLine(img, p1, (head_x, head_y), (0, 255, 0), 3)

            cv2.line(img, p1, p2, (150, 150, 150), 1)

            cv2.putText(
                img,
                f"Dist: {distance:.1f} px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"AngDiff: {math.degrees(angle_diff):.1f} deg",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        mode_text = "AUTO TRACKING" if self.auto_mode else "MANUAL (KEYBOARD)"
        mode_color = (0, 255, 255) if self.auto_mode else (0, 255, 0)
        cv2.putText(
            img, mode_text, (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2
        )

        if self.is_stopped:
            cv2.putText(
                img,
                "EMERGENCY STOP (TOO CLOSE)",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.height, img_msg.width = height, width
        img_msg.encoding = "bgr8"
        img_msg.step = 3 * width
        img_msg.data = img.tobytes()
        self.pub_debug_img.publish(img_msg)

    def decide_action(self):

        twist = Twist()

        if self.is_stopped:
            rospy.logwarn("[DEBUG] EMERGENCY STOP ACTIVE!!")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publish_debug_image(distance=0.0, angle_diff=0.0, current_theta=0.0)
            return twist

        if self.poses[1] is not None and self.poses[2] is not None:

            me = self.myID
            enemy = 3 - me

            x0, y0 = float(self.poses[me]["x"]), float(self.poses[me]["y"])
            theta0 = float(self.poses[me]["theta"])
            x1, y1 = float(self.poses[enemy]["x"]), float(self.poses[enemy]["y"])
            theta1 = float(self.poses[enemy]["theta"])
            poses0 = np.array([x0, y0, theta0], dtype=np.float32)
            poses1 = np.array([x1, y1, theta1], dtype=np.float32)

            dx = x1 - x0
            dy = y1 - y0

            # 補助入力パラメータの計算
            c = math.cos(-theta0)
            s = math.sin(-theta0)
            rel_body_x = c * dx - s * dy
            rel_body_y = s * dx + c * dy
            dist = math.sqrt(dx**2 + dy**2)
            bearing = wrap_angle(math.atan2(dy, dx) - theta0)
            target_heading_rel = wrap_angle(theta1 - theta0)

            obs = np.array(
                [
                    x0 / self.cfg.field_w,
                    y0 / self.cfg.field_h,
                    theta0,
                    x1 / self.cfg.field_w,
                    y1 / self.cfg.field_h,
                    theta1,
                    rel_body_x / self.cfg.field_w,
                    rel_body_y / self.cfg.field_h,
                    dist / max(self.cfg.field_w, self.cfg.field_h),
                    bearing,
                    target_heading_rel,
                ],
                dtype=np.float32,
            )
            obs = obs[np.newaxis, :]

            # 学習済みの ONNX モデルで推論
            # outputs = self.session.run(None, {self.input_name: obs})
            # rl_action = outputs[0][0]
            # residual = np.array([
            #     self.cfg.rl_residual_v_scale*rl_action[0],
            #     self.cfg.rl_residual_omega_scale*rl_action[1],
            # ], dtype=np.float64)

            # MPC の結果と足し合わせる
            mpc_actions = self.MPCctr._compute_mpc_actions(poses0, poses1)
            # actions = mpc_actions + residual
            actions = mpc_actions
            actions[0] = np.clip(actions[0], -self.cfg.max_v, self.cfg.max_v)
            actions[1] = np.clip(actions[1], -self.cfg.max_omega, self.cfg.max_omega)

            # モータ指令
            twist.linear.x = actions[0]
            twist.angular.z = actions[1]

            # 衝突回避のため，相手に近づいたら静止する（非常停止機能１）
            self.is_stopped = True if dist < 0.3 else False
            rospy.loginfo("[EMERGENCY STOP] Almost crashed")

            # 場外に出たら静止する（非常停止機能２）
            if x0 < 0.0 or x0 > self.cfg.field_w:
                self.is_stopped = True
            if y0 < 0.0 or y0 > self.cfg.field_h:
                self.is_stopped = True
            rospy.loginfo("[EMERGENCY STOP] Out of field")

            self.publish_debug_image(
                distance=dist, angle_diff=bearing, current_theta=theta0
            )

        else:

            id1_status = "OK" if self.poses[1] is not None else "MISSING"
            id2_status = "OK" if self.poses[2] is not None else "MISSING"
            rospy.logwarn_throttle(
                1.0,
                f"[MISSING] 自己位置(ID1): {id1_status} | ターゲット(ID2): {id2_status}",
            )
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        return twist

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.auto_mode:
                # For Debug
                rospy.loginfo("[1]is_stopped: %s", self.is_stopped)
                # 自動追従モード（rキー入力後）のみ、決定したアクションをパブリッシュする
                cmd_vel = self.decide_action()
                self.pub_vel.publish(cmd_vel)
            else:
                # 手動モード（初期状態）のときは絶対に勝手に動かないようにパブリッシュを叩かない
                # if self.poses[1] is not None:
                #     theta1 = self.poses[1]['theta']
                #     self.publish_debug_image(current_theta=theta1)
                # else:
                #     self.publish_debug_image()
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.pub_vel.publish(twist)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("main_control_node")
    controller = MainController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
