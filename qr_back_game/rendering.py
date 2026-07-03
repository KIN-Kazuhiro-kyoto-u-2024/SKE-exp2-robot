import math

import numpy as np

from .geometry import unit


def render_env(env, mode="human"):
    import pygame

    cfg = env.cfg
    scale = 260
    w_px, h_px = int(cfg.field_w * scale), int(cfg.field_h * scale)
    if env.viewer is None:
        pygame.init()
        if mode == "human":
            env.viewer = pygame.display.set_mode((w_px, h_px))
        else:
            env.viewer = pygame.Surface((w_px, h_px))
    surf = env.viewer
    surf.fill((245, 245, 245))

    def to_px(xy):
        return int(xy[0] * scale), int((cfg.field_h - xy[1]) * scale)

    # Grid.
    for ix in range(cfg.grid_w + 1):
        x = int(ix * cfg.cell_size * scale)
        pygame.draw.line(surf, (215, 215, 215), (x, 0), (x, h_px), 1)
    for iy in range(cfg.grid_h + 1):
        y = int(iy * cfg.cell_size * scale)
        pygame.draw.line(surf, (215, 215, 215), (0, y), (w_px, y), 1)

    # Paths.
    for idx, path in enumerate(env.last_paths):
        if len(path) >= 2:
            pts = [to_px(p) for p in path]
            pygame.draw.lines(
                surf, (80, 150, 230) if idx == 0 else (230, 140, 80), False, pts, 2
            )

    # Rear sectors and robot rectangles.
    colors = [(40, 90, 220), (220, 70, 40)]
    for i, pose in enumerate(env.poses):
        xy = pose[:2]
        th = float(pose[2])
        # rear sector fan
        rear = th + math.pi
        fan = [to_px(xy)]
        for a in np.linspace(
            rear - cfg.rear_sector_half_angle, rear + cfg.rear_sector_half_angle, 18
        ):
            fan.append(
                to_px(xy + cfg.rear_view_range * np.array([math.cos(a), math.sin(a)]))
            )
        if len(fan) > 2:
            pygame.draw.polygon(surf, (225, 225, 235), fan, 0)

    for i, pose in enumerate(env.poses):
        xy = pose[:2]
        th = float(pose[2])
        f = unit(th)
        r = np.array([-f[1], f[0]])
        corners = [
            xy + f * cfg.robot_l / 2 + r * cfg.robot_w / 2,
            xy + f * cfg.robot_l / 2 - r * cfg.robot_w / 2,
            xy - f * cfg.robot_l / 2 - r * cfg.robot_w / 2,
            xy - f * cfg.robot_l / 2 + r * cfg.robot_w / 2,
        ]
        pygame.draw.polygon(surf, colors[i], [to_px(c) for c in corners])
        pygame.draw.circle(
            surf, (70, 70, 70), to_px(xy), int(cfg.collision_radius * scale), 1
        )
        front = xy + f * cfg.robot_l * 0.75
        pygame.draw.line(surf, (20, 20, 20), to_px(xy), to_px(front), 3)
        qr = xy - f * cfg.robot_l / 2
        pygame.draw.circle(surf, (0, 0, 0), to_px(qr), 3)

    if mode == "human":
        import pygame

        pygame.display.flip()
        pygame.event.pump()
        return None
    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2))
