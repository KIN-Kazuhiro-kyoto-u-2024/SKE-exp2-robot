from typing import Dict, List, Optional
import math

try:
    import gym
    from gym import spaces
except ImportError:  # keeps headless tests usable before installing requirements.txt
    class _Env(object):
        pass
    class _Box(object):
        def __init__(self, low, high, dtype=None, shape=None):
            self.low = low
            self.high = high
            self.dtype = dtype
            self.shape = shape
    class _Spaces(object):
        Box = _Box
    class _Gym(object):
        Env = _Env
    gym = _Gym()
    spaces = _Spaces()
import numpy as np

from .config import GameConfig
from .geometry import has_won, pose_step, wrap_angle
from .mpc import MPCController
from .path_planning import plan_path_to_rear


class BackQrGameEnv(gym.Env):
    """Two-robot back-taking game.

    Observation is a flat vector [x0, y0, th0, x1, y1, th1].
    External actions are accepted for RL experiments; when action=None,
    built-in A*+MPC controls both robots.

    No collision-prediction safety filter is applied. Actions from the
    policy/MPC are integrated directly.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, cfg: Optional[GameConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or GameConfig()
        high = np.array([self.cfg.field_w, self.cfg.field_h, math.pi] * 2, dtype=np.float32)
        low = np.array([0.0, 0.0, -math.pi] * 2, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.cfg.max_v, -self.cfg.max_omega] * 2, dtype=np.float32),
            high=np.array([self.cfg.max_v, self.cfg.max_omega] * 2, dtype=np.float32),
            dtype=np.float32,
        )
        self.render_mode = render_mode
        self.controllers = [MPCController(self.cfg, self.cfg.mpc_seed), MPCController(self.cfg, self.cfg.mpc_seed + 1)]
        self.viewer = None
        self.seed(None)
        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _sample_random_start_poses(self) -> np.ndarray:
        margin = float(getattr(self.cfg, "start_margin", 0.20))
        min_dist = float(getattr(self.cfg, "start_min_dist", 0.45))
        for _ in range(1000):
            x0 = self.np_random.uniform(margin, self.cfg.field_w - margin)
            y0 = self.np_random.uniform(margin, self.cfg.field_h - margin)
            x1 = self.np_random.uniform(margin, self.cfg.field_w - margin)
            y1 = self.np_random.uniform(margin, self.cfg.field_h - margin)
            dx = x1 - x0
            dy = y1 - y0
            dist = float(np.hypot(dx, dy))
            if dist < min_dist:
                continue
            theta0 = float(np.arctan2(dy, dx))
            theta1 = float(np.arctan2(-dy, -dx))
            return np.array([[x0, y0, theta0], [x1, y1, theta1]], dtype=np.float64)
        return self._fixed_start_poses()

    def _fixed_start_poses(self) -> np.ndarray:
        return np.array([
            [0.75, self.cfg.field_h / 2.0, 0.0],
            [2.25, self.cfg.field_h / 2.0, math.pi],
        ], dtype=np.float64)

    def reset(self):
        if bool(getattr(self.cfg, "random_start", False)):
            self.poses = self._sample_random_start_poses()
        else:
            self.poses = self._fixed_start_poses()
        self.step_count = 0
        self.last_paths: List[List[np.ndarray]] = [[], []]
        self.last_actions = np.zeros((2, 2), dtype=np.float64)
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.array(list(self.poses[0]) + list(self.poses[1]), dtype=np.float32)

    # 自分から相手の後方をめがけて経路探索
    # preffered_y は A* アルゴリズムのパラメータ
    def _compute_grid_path(self, attacker_idx: int) -> List[np.ndarray]:
        target_idx = 1 - attacker_idx
        preferred_y = 0.35 if attacker_idx == 0 else self.cfg.field_h - 0.35
        return plan_path_to_rear(
            self.poses[attacker_idx],
            self.poses[target_idx],
            self.cfg,
            avoid_xy=self.poses[target_idx, :2],
            preferred_y=preferred_y,
        )

    def _compute_mpc_actions(self) -> np.ndarray:
        paths = [self._compute_grid_path(0), self._compute_grid_path(1)]
        self.last_paths = paths
        a0, _ = self.controllers[0].act(self.poses[0], self.poses[1], paths[0])
        a1, _ = self.controllers[1].act(self.poses[1], self.poses[0], paths[1])
        return np.vstack([a0, a1])

    def step(self, action=None):

        # actions を最終調整
        # None の場合，MPC で actions を決定（_compute_mpc_actions()）
        self.step_count += 1
        if action is None:
            actions = self._compute_mpc_actions()
        else:
            arr = np.asarray(action, dtype=np.float64).reshape(2, 2)
            actions = arr.copy()
            actions[:, 0] = np.clip(actions[:, 0], -self.cfg.max_v, self.cfg.max_v)
            actions[:, 1] = np.clip(actions[:, 1], -self.cfg.max_omega, self.cfg.max_omega)

        self.last_actions = actions.copy()

        # i=0: 自分，i=1: 相手
        # それぞれの位置と向きを1ステップ分だけ更新し，位置をフィールド内に制限
        for i in range(2):
            self.poses[i] = pose_step(self.poses[i], actions[i], self.cfg.dt, self.cfg.max_v, self.cfg.max_omega)
            self.poses[i, 0] = float(np.clip(self.poses[i, 0], 0.0, self.cfg.field_w))
            self.poses[i, 1] = float(np.clip(self.poses[i, 1], 0.0, self.cfg.field_h))
            self.poses[i, 2] = wrap_angle(float(self.poses[i, 2]))

        # 勝敗判定
        win0 = has_won(self.poses[0], self.poses[1], self.cfg)
        win1 = has_won(self.poses[1], self.poses[0], self.cfg)
        done = bool(win0 or win1 or self.step_count >= self.cfg.max_steps)
        winner = 0 if win0 else (1 if win1 else None)

        # 報酬の設定
        reward = 0.0
        if win0 and not win1:
            reward = 1.0
        elif win1 and not win0:
            reward = -1.0
        elif win0 and win1:
            reward = 0.0
        info: Dict = {
            "winner": winner,
            "win0": win0,
            "win1": win1,
            "actions": self.last_actions.copy(),
            "paths": self.last_paths,
            "step_count": self.step_count,
        }
        return self._obs(), reward, done, info

    def render(self, mode="human"):
        from .rendering import render_env
        return render_env(self, mode=mode)

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.quit()
            self.viewer = None
