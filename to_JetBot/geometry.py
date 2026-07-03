import math
from typing import Tuple

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_to(src: np.ndarray, dst: np.ndarray) -> float:
    d = dst - src
    return math.atan2(float(d[1]), float(d[0]))


def unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)


def pose_step(
    pose: np.ndarray, action: np.ndarray, dt: float, max_v: float, max_omega: float
) -> np.ndarray:

    # pose[0]: 自分の x 座標，pose[1]: 自分の y 座標，pose[2]: 自分の向き(th)[rad]
    # action[0]: 自分の速さ(v)，action[1]: 自分の角速度(w)[rad/s]

    v = float(np.clip(action[0], -max_v, max_v))
    w = float(np.clip(action[1], -max_omega, max_omega))
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])

    # Midpoint integration gives stable enough behavior for short MPC rollouts.
    mid_th = th + 0.5 * w * dt

    # dt 後の位置と向きを計算
    x += v * math.cos(mid_th) * dt
    y += v * math.sin(mid_th) * dt
    th = wrap_angle(th + w * dt)
    return np.array([x, y, th], dtype=np.float64)


def cell_of_xy(
    x: float, y: float, cell_size: float, grid_w: int, grid_h: int
) -> Tuple[int, int]:
    ix = int(np.clip(math.floor(x / cell_size), 0, grid_w - 1))
    iy = int(np.clip(math.floor(y / cell_size), 0, grid_h - 1))
    return ix, iy


def xy_of_cell(cell: Tuple[int, int], cell_size: float) -> np.ndarray:
    ix, iy = cell
    return np.array([(ix + 0.5) * cell_size, (iy + 0.5) * cell_size], dtype=np.float64)


# attacker が target の後方の扇形の領域に入っているか
def point_in_rear_sector(
    attacker_pose: np.ndarray,
    target_pose: np.ndarray,
    view_range: float,
    half_angle: float,
) -> bool:

    # attacker と target の相対位置の計算
    attacker_xy = attacker_pose[:2]
    target_xy = target_pose[:2]
    rel = attacker_xy - target_xy

    # 距離の判定
    dist = float(np.linalg.norm(rel))
    if dist > view_range or dist < 1e-9:
        return False

    # 角度の判定
    rear_dir = wrap_angle(float(target_pose[2]) + math.pi)
    bearing = math.atan2(float(rel[1]), float(rel[0]))
    return abs(wrap_angle(bearing - rear_dir)) <= half_angle


# attacker が target の後方の扇形の領域に入っているか～改～
# target の背後をより忠実に表現
def camera_sees_target_rear(
    attacker_pose: np.ndarray,
    target_pose: np.ndarray,
    camera_range: float,
    half_angle: float,
) -> bool:
    attacker_xy = attacker_pose[:2]
    # QR is assumed to be on the back edge center.
    qr_xy = target_pose[:2] - unit(float(target_pose[2])) * 0.075
    dist = float(np.linalg.norm(qr_xy - attacker_xy))
    if dist > camera_range or dist < 1e-9:
        return False
    bearing = angle_to(attacker_xy, qr_xy)
    return abs(wrap_angle(bearing - float(attacker_pose[2]))) <= half_angle


# 勝利判定（attacker が target の背後をとれたか）
def has_won(attacker_pose: np.ndarray, target_pose: np.ndarray, cfg) -> bool:
    return point_in_rear_sector(
        attacker_pose, target_pose, cfg.rear_view_range, cfg.rear_sector_half_angle
    ) and camera_sees_target_rear(
        attacker_pose, target_pose, cfg.camera_range, cfg.camera_fov_half_angle
    )
