#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

print("--- Loading Aruco Detector Script with Midpoint Origin & ID0-ID3 X-Axis ---")


class ArucoDetector:
    def __init__(self):
        rospy.init_node("aruco_detector_node")

        self.bridge = CvBridge()
        self.image_topic = rospy.get_param("~image_topic", "usb_cam/image_raw")

        self.marker_pub = rospy.Publisher("/aruco/marker_json", String, queue_size=10)
        self.image_pub = rospy.Publisher("/aruco/debug_image", Image, queue_size=1)

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.debug_fps_limit = 5.0
        self.last_debug_time = rospy.get_time()

        self.sub = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24
        )

        # 実世界におけるフィールドの広さ（メートル単位）
        self.field_w = 1.8
        self.field_h = 0.9

        rospy.loginfo("Aruco detector started. (Midpoint Origin Mode)")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        marker_list = []
        stop_flag = False
        info_id1 = None
        info_id2 = None

        # 基準座標系の情報 (原点x, 原点y, 全体の傾き)
        base_grid_info = None
        side_x = 1.0

        if ids is not None:
            ids_flat = ids.flatten().tolist()

            # --- 【超重要】まずループの前に、基準となる ID0 と ID3 から座標系を確定させる ---
            if 0 in ids_flat and 3 in ids_flat:
                # ID 0 の中心
                idx0 = ids_flat.index(0)
                c0 = corners[idx0][0]
                x0 = sum(p[0] for p in c0) / 4.0
                y0 = sum(p[1] for p in c0) / 4.0

                # 衝突判定用にID0の一辺の長さをキープ (左上[0]から右上[1]の距離)
                # 実世界ではこれが 0.1m．これをもとにピクセル -> メートルのスケール変換のための unit を計算．
                side_x = math.sqrt(
                    (c0[1][0] - c0[0][0]) ** 2 + (c0[1][1] - c0[0][1]) ** 2
                )
                unit = 0.1 / side_x

                # ID 3 の中心
                idx3 = ids_flat.index(3)
                c3 = corners[idx3][0]
                x3 = sum(p[0] for p in c3) / 4.0
                y3 = sum(p[1] for p in c3) / 4.0

                # 原点を ID0 と ID3 の中点にする
                ox = (x0 + x3) / 2.0
                oy = (y0 + y3) / 2.0

                # ID0 から ID3 へ向かうベクトルを X 軸とする
                theta_base = math.atan2(y3 - y0, x3 - x0)

                # 座標系の基盤情報を確定
                base_grid_info = (ox, oy, theta_base)

                # --- 確定した座標系を使って、すべてのマーカーの位置を一斉に計算する ---
                temp_coords = {}

                for i, corner in enumerate(corners):
                    target_id = ids_flat[i]

                    c = corner[0]
                    xi = sum(p[0] for p in c) / 4.0
                    yi = sum(p[1] for p in c) / 4.0

                    # 【修正ポイント】マーカーの「上方向」を正面とするベクトル
                    # c[3]は左下、c[0]は左上。よって「左下から左上」へ向かうベクトルで計算すると、正面が0度になります。
                    thetai = math.atan2(c[0][1] - c[3][1], c[0][0] - c[3][0])

                    # 計算した中点（原点）からの相対ピクセル距離
                    dx, dy = xi - ox, yi - oy
                    cos_tb, sin_tb = math.cos(-theta_base), math.sin(-theta_base)

                    # X軸（ID0->ID3ライン）に合わせた回転変換
                    rx = dx * cos_tb - dy * sin_tb
                    ry = dx * sin_tb + dy * cos_tb
                    rtheta = math.atan2(
                        math.sin(thetai - theta_base), math.cos(thetai - theta_base)
                    )

                    ###################################################
                    # ここからシミュレーションと合わせるため座標系を変換 ###
                    ###################################################

                    # OpenCV 座標系を通常の二次元直交座標系に変換する
                    mry = -ry
                    mrtheta = -rtheta

                    real_x = rx * unit + 0.5 * self.field_w
                    real_y = mry * unit + 0.5 * self.field_h

                    marker_data = {
                        "id": int(target_id),
                        "x": round(real_x, 2),
                        "y": round(real_y, 2),
                        "theta": round(mrtheta, 3),
                        "x_pixel": round(rx, 2),
                        "y_pixel": round(ry, 2),
                        "theta_cv": round(rtheta, 3),
                        "center_pixel": (int(xi), int(yi)),
                        "theta_image": thetai,
                    }
                    marker_list.append(marker_data)

                    if target_id == 1:
                        info_id1 = marker_data
                    if target_id == 2:
                        info_id2 = marker_data
                    temp_coords[target_id] = (xi, yi)

                # --- 衝突判定 ---
                # ID1 と ID2 との距離が0.2m（side_x の2倍）以下になったら非常停止
                if 1 in temp_coords and 2 in temp_coords:
                    p1 = temp_coords[1]
                    p2 = temp_coords[2]
                    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if dist <= 2.0 * side_x:
                        stop_flag = True

        # JSONパブリッシュ
        output_data = {"markers": marker_list, "stop": stop_flag}
        self.marker_pub.publish(json.dumps(output_data))

        ###############################
        # ここからデバッグ用の画像処理 ###
        ###############################

        now = rospy.get_time()
        if (now - self.last_debug_time) > (1.0 / self.debug_fps_limit):
            self.last_debug_time = now

            if base_grid_info:
                grid_ox, grid_oy, ot = base_grid_info

                def rel_to_pix(rx, ry):
                    u = grid_ox + (rx * math.cos(ot) - ry * math.sin(ot))
                    v = grid_oy + (rx * math.sin(ot) + ry * math.cos(ot))
                    return (int(u), int(v))

                grid_color = (70, 70, 70)
                for g in range(-2000, 2001, 100):
                    cv2.line(
                        frame, rel_to_pix(g, -2000), rel_to_pix(g, 2000), grid_color, 1
                    )
                    cv2.line(
                        frame, rel_to_pix(-2000, g), rel_to_pix(2000, g), grid_color, 1
                    )

                cv2.line(
                    frame, rel_to_pix(-2000, 0), rel_to_pix(2000, 0), (0, 0, 255), 2
                )
                cv2.line(
                    frame, rel_to_pix(0, -2000), rel_to_pix(0, 2000), (0, 255, 0), 2
                )
                cv2.circle(frame, (int(grid_ox), int(grid_oy)), 5, (255, 0, 0), -1)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # 修正された正面ベクトル（矢印）の描画
            if ids is not None:
                for m in marker_list:
                    cx, cy = m["center_pixel"]
                    theta_img = m["theta_image"]

                    arrow_length = int(side_x * 1.5)

                    arrow_x = int(cx + arrow_length * math.cos(theta_img))
                    arrow_y = int(cy + arrow_length * math.sin(theta_img))

                    cv2.arrowedLine(
                        frame,
                        (cx, cy),
                        (arrow_x, arrow_y),
                        (0, 0, 255),
                        3,
                        tipLength=0.3,
                    )

            overlay_y = 30

            def draw_text(text, y_pos, color=(255, 255, 255)):
                cv2.putText(
                    frame,
                    text,
                    (frame.shape[1] - 350, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if info_id1:
                theta1_deg = math.degrees(info_id1["theta_cv"])
                draw_text(
                    "ID1: X:{:.0f} Y:{:.0f} T:{:.1f}deg".format(
                        info_id1["x_pixel"], info_id1["y_pixel"], theta1_deg
                    ),
                    overlay_y,
                )
                overlay_y += 25
                theta1_deg = math.degrees(info_id1["theta"])
                draw_text(
                    "(real): X:{:.2f} Y:{:.2f} T:{:.1f}deg".format(
                        info_id1["x"], info_id1["y"], theta1_deg
                    ),
                    overlay_y,
                )
                overlay_y += 25
            if info_id2:
                theta2_deg = math.degrees(info_id2["theta_cv"])
                draw_text(
                    "ID2: X:{:.0f} Y:{:.0f} T:{:.1f}deg".format(
                        info_id2["x_pixel"], info_id2["y_pixel"], theta2_deg
                    ),
                    overlay_y,
                )
                overlay_y += 25
                theta2_deg = math.degrees(info_id2["theta"])
                draw_text(
                    "(real): X:{:.2f} Y:{:.2f} T:{:.1f}deg".format(
                        info_id2["x"], info_id2["y"], theta2_deg
                    ),
                    overlay_y,
                )
                overlay_y += 25

            stop_color = (0, 0, 255) if stop_flag else (0, 255, 0)
            draw_text("STOP: {}".format(stop_flag), overlay_y, stop_color)

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.image_pub.publish(debug_msg)
            except Exception as e:
                rospy.logerr("debug image publish error %s", e)


if __name__ == "__main__":
    detector = ArucoDetector()
    rospy.spin()
