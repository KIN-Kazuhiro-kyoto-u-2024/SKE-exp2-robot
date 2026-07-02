"""Single-agent residual RL wrapper.

robot0 is controlled by MPC + PPO residual.
robot1 remains the original A* + MPC policy.

The PPO action is normalized to [-1, 1]^2 and converted to
[dv, d_omega]. The final robot0 command is

    u0 = u0_mpc + [dv, d_omega]

clipped to the same limits as the base environment.
"""
import math
from typing import Optional

try:
    import gym
    from gym import spaces
except ImportError:
    class _Env(object):
        pass
    class _Box(object):
        def __init__(self, low, high, shape=None, dtype=None):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
    class _Spaces(object):
        Box = _Box
    class _Gym(object):
        Env = _Env
    gym = _Gym()
    spaces = _Spaces()

import numpy as np

from .config import GameConfig
from .env import BackQrGameEnv
from .geometry import angle_to, point_in_rear_sector, wrap_angle


class ResidualPPOEnv(gym.Env):
    """Gym env for training only robot0 residual over the MPC controller."""

    def __init__(self, cfg: Optional[GameConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or GameConfig()
        self.base_env = BackQrGameEnv(cfg=self.cfg, render_mode=render_mode)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation is robot0-centric and includes the MPC command.
        obs_dim = 11
        #obs_dim = 18
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.last_mpc_actions = np.zeros((2, 2), dtype=np.float64)
        self.prev_rear_score = 0.0
        self.prev_dist = 0.0

    def seed(self, seed=None):
        return self.base_env.seed(seed)

    def reset(self):
        self.base_env.reset()
        self.last_mpc_actions = self.base_env._compute_mpc_actions()
        self.prev_rear_score = self._rear_score()
        self.prev_dist = self._distance_to_opponent()
        return self._get_obs()

    def step(self, rl_action):
        rl_action = np.asarray(rl_action, dtype=np.float64).reshape(2)
        rl_action = np.clip(rl_action, -1.0, 1.0)

        before_rear_score = self._rear_score()
        before_dist = self._distance_to_opponent()

        #自分と相手の行動を MPC により決定
        mpc_actions = self.base_env._compute_mpc_actions()
        self.last_mpc_actions = mpc_actions.copy()

        residual = np.array([
            self.cfg.rl_residual_v_scale * rl_action[0],
            self.cfg.rl_residual_omega_scale * rl_action[1],
        ], dtype=np.float64)

        # 自分（0）のみ強化学習の結果を residual として加算
        actions = mpc_actions.copy()
        actions[0] = actions[0] + residual
        actions[0, 0] = np.clip(actions[0, 0], -self.cfg.max_v, self.cfg.max_v)
        actions[0, 1] = np.clip(actions[0, 1], -self.cfg.max_omega, self.cfg.max_omega)
        # robot1 is intentionally left as MPC-only.

        _, base_reward, done, info = self.base_env.step(actions)

        reward = self._compute_reward(
            rl_action=rl_action,
            before_rear_score=before_rear_score,
            after_rear_score=self._rear_score(),
            before_dist=before_dist,
            after_dist=self._distance_to_opponent(),
            done=done,
            info=info,
        )
        info["mpc_actions"] = mpc_actions.copy()
        info["residual_action_norm"] = rl_action.copy()
        info["robot0_action_after_residual"] = actions[0].copy()
        info["base_reward"] = base_reward
        return self._get_obs(), float(reward), done, info

    def _get_obs(self) -> np.ndarray:

        # p0: 自分の位置・向き（x0, y0, theta0）
        # p1: 相手の位置・向き（x1, y1, theta1）
        # rel_world: 相手の相対的な位置（x1-x0, y1-y0）
        # rel_body: 相手の相対的な位置，ただし自分が今向いている方向を x 軸正の向きとする．
        # dist: 相手までの距離（\sqrt{(x1-x0)^2 + (y1-y0)^2}）
        # bearing: 相手がいる方向が，今自分が向いている方向からどれだけずれているか
        # target_heading_rel: 相手と自分の向きの違い

        p0 = self.base_env.poses[0]
        p1 = self.base_env.poses[1]
        rel_world = p1[:2] - p0[:2]
        c = math.cos(-float(p0[2]))
        s = math.sin(-float(p0[2]))
        rel_body = np.array([
            c * rel_world[0] - s * rel_world[1],
            s * rel_world[0] + c * rel_world[1],
        ], dtype=np.float64)
        dist = float(np.linalg.norm(rel_world))
        bearing = wrap_angle(angle_to(p0[:2], p1[:2]) - float(p0[2]))
        target_heading_rel = wrap_angle(float(p1[2]) - float(p0[2]))
    
        obs = np.array([
            p0[0] / self.cfg.field_w,
            p0[1] / self.cfg.field_h,
            float(p0[2]),
            p1[0] / self.cfg.field_w,
            p1[1] / self.cfg.field_h,
            float(p1[2]),
            rel_body[0] / self.cfg.field_w,
            rel_body[1] / self.cfg.field_h,
            dist / max(self.cfg.field_w, self.cfg.field_h),
            bearing,
            target_heading_rel,
        ], dtype=np.float32)
        return obs

    def _distance_to_opponent(self) -> float:
        return float(np.linalg.norm(self.base_env.poses[0, :2] - self.base_env.poses[1, :2]))

    # 自分（0）が相手（1）の背後をどれだけうまく取れているかを表すスコアを計算（rear_score とよぶ）
    def _rear_score(self) -> float:
        """Smooth progress score for robot0 getting behind robot1."""
        p0 = self.base_env.poses[0]
        p1 = self.base_env.poses[1]
        rel = p0[:2] - p1[:2]
        dist = float(np.linalg.norm(rel))
        if dist < 1e-9:
            return 0.0
        rear_dir = wrap_angle(float(p1[2]) + math.pi)
        bearing_from_target = math.atan2(float(rel[1]), float(rel[0]))
        rear_align = math.cos(wrap_angle(bearing_from_target - rear_dir))
        # Favor being in the rear direction and not too far away.
        range_score = max(0.0, 1.0 - dist / max(self.cfg.rear_view_range, 1e-6))
        return 0.5 * (rear_align + 1.0) * range_score

    # rear_score や自分と相手の距離（before_dist, after_dist）をもとに報酬を決定する
    def _compute_reward(self, rl_action, before_rear_score, after_rear_score, before_dist, after_dist, done, info) -> float:
        reward = 0.0
        if info.get("winner") == 0:
            reward += 10.0
        elif info.get("winner") == 1:
            reward -= 10.0
        elif done:
            reward -= 1.0

        reward += 2.0 * (after_rear_score - before_rear_score)
        reward += 0.03 * after_rear_score
        reward -= 0.002  # small time penalty

        # Do not reintroduce a safety filter; this is only a learning penalty.
        close_r = float(getattr(self.cfg, "close_distance_penalty_radius", 0.24))
        if after_dist < close_r:
            reward -= 0.25 * (close_r - after_dist) / max(close_r, 1e-6)

        reward -= float(self.cfg.rl_residual_penalty) * float(np.sum(np.square(rl_action)))
        return reward

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()
