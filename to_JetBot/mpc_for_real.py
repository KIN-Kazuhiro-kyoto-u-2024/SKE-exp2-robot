import math
from typing import List, Tuple

import numpy as np
from geometry import angle_to, has_won, pose_step, wrap_angle
from path_planning import plan_path_to_rear


class MPCController:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.base_actions = self._make_action_library()

    # 移動ロボットの行動は，有限個の速さ v と角速度 w の組み合わせ
    # 2次元配列として保持
    def _make_action_library(self) -> np.ndarray:
        c = self.cfg
        # Include reverse and reverse-turn candidates so the MPC can choose them
        # as part of ordinary path tracking. No post-action safety filter is used.
        vs = np.array([-0.65, -0.35, 0.0, 0.35, 0.70, 1.0]) * c.max_v
        ws = np.array([-1.0, -0.55, 0.0, 0.55, 1.0]) * c.max_omega
        acts = []
        for v in vs:
            for w in ws:
                acts.append([float(v), float(w)])
        return np.array(acts, dtype=np.float64)

    def _sample_sequences(self) -> np.ndarray:

        # num_sequences X horizon のランダムな行動配列（2次元）
        c = self.cfg
        n, h = c.mpc_num_sequences, c.mpc_horizon
        idx = self.rng.randint(0, len(self.base_actions), size=(n, h))
        seq = self.base_actions[idx].copy()

        # Include useful deterministic primitives.
        primitives = []
        for v in [c.max_v, 0.65 * c.max_v, 0.0, -0.45 * c.max_v, -0.65 * c.max_v]:
            for w in [0.0, -0.65 * c.max_omega, 0.65 * c.max_omega]:
                primitives.append(np.tile(np.array([v, w], dtype=np.float64), (h, 1)))

        # Brake/retreat first, then resume forward: often resolves head-on conflicts.
        for sign in [-1.0, 1.0]:
            seq0 = np.zeros((h, 2), dtype=np.float64)
            split = max(1, h // 2)
            seq0[:split, 0] = -0.55 * c.max_v
            seq0[:split, 1] = sign * 0.55 * c.max_omega
            seq0[split:, 0] = 0.60 * c.max_v
            primitives.append(seq0)
        k = min(len(primitives), n)
        seq[:k] = np.array(primitives[:k])
        return seq

    def act(
        self, self_pose: np.ndarray, other_pose: np.ndarray, path: List[np.ndarray]
    ) -> Tuple[np.ndarray, dict]:

        if not path:
            return np.zeros(2, dtype=np.float64), {"cost": 0.0}

        # 自分・相手の位置と向き（self_pose, other_pose）と最適経路（path）の情報から，MPC で最適な行動（速度と角速度の組）を求める
        # _sample_sequences() で得た初期 sequence から始めて，最適な行動を探索している？
        seqs = self._sample_sequences()
        costs = np.zeros(seqs.shape[0], dtype=np.float64)
        best_i = 0
        best_cost = float("inf")
        for i, seq in enumerate(seqs):
            cost = self._rollout_cost(self_pose, other_pose, seq, path)
            costs[i] = cost
            if cost < best_cost:
                best_cost = cost
                best_i = i
        return seqs[best_i, 0].copy(), {
            "cost": float(best_cost),
            "mean_cost": float(np.mean(costs)),
        }

    def _path_target(self, xy: np.ndarray, path: List[np.ndarray]) -> np.ndarray:
        d = [float(np.linalg.norm(p - xy)) for p in path]
        nearest = int(np.argmin(d))
        idx = min(nearest + self.cfg.waypoint_lookahead, len(path) - 1)
        return path[idx]

    def _rollout_cost(
        self,
        pose: np.ndarray,
        other_pose: np.ndarray,
        seq: np.ndarray,
        path: List[np.ndarray],
    ) -> float:
        c = self.cfg
        p = pose.copy()
        cost = 0.0
        last_dist_to_end = float(np.linalg.norm(path[-1] - p[:2]))
        prev_action = np.zeros(2, dtype=np.float64)
        for t, u in enumerate(seq):
            p = pose_step(p, u, c.dt, c.max_v, c.max_omega)
            target = self._path_target(p[:2], path)
            dist_path = float(np.linalg.norm(target - p[:2]))
            dist_end = float(np.linalg.norm(path[-1] - p[:2]))
            if dist_end < 0.28:
                heading_ref = angle_to(p[:2], other_pose[:2])
            else:
                heading_ref = angle_to(p[:2], target)
            heading_err = abs(wrap_angle(heading_ref - float(p[2])))

            # Other robot is assumed nearly constant over this short horizon.
            # This is only an MPC cost term; the environment does not override
            # the selected action with a safety filter.
            sep = float(np.linalg.norm(p[:2] - other_pose[:2]))
            danger_margin = 2.35 * c.collision_radius
            hard_margin = 2.0 * c.collision_radius
            risk = max(0.0, danger_margin - sep)
            hard_collision = max(0.0, hard_margin - sep)

            progress_reward = last_dist_to_end - dist_end
            last_dist_to_end = dist_end

            cost += 18.0 * dist_path**2
            cost += 7.0 * dist_end**2
            cost += 0.9 * heading_err**2
            cost += 420.0 * risk**2
            cost += 2500.0 * hard_collision**2
            cost += 0.04 * (u[1] / c.max_omega) ** 2
            cost += 0.03 * ((u[0] - prev_action[0]) / max(c.max_v, 1e-6)) ** 2
            cost -= 14.0 * progress_reward
            cost -= (
                0.18 * max(float(u[0]), 0.0) / c.max_v
            )  # prefer quick forward tracking
            if sep < danger_margin and float(u[0]) < 0.0:
                cost -= (
                    0.75 * abs(float(u[0])) / max(c.max_v, 1e-6)
                )  # allow deliberate retreat
            prev_action = u
            if has_won(p, other_pose, c):
                cost -= 140.0 / (t + 1)
                break
        return float(cost)

    def _compute_mpc_actions(self, poses0, poses1) -> np.ndarray:
        path = self._compute_grid_path(poses0, poses1)
        a0, _ = self.act(poses0, poses1, path)
        return a0

        # paths = [self._compute_grid_path(0), self._compute_grid_path(1)]
        # self.last_paths = paths
        # a0, _ = self.act(self.poses[0], self.poses[1], paths[0])
        # a1, _ = self.act(self.poses[1], self.poses[0], paths[1])
        # return np.vstack([a0, a1])

    # 自分から相手の後方をめがけて経路探索
    # preffered_y は A* アルゴリズムのパラメータ
    def _compute_grid_path(self, poses0, poses1) -> List[np.ndarray]:
        return plan_path_to_rear(
            poses0,
            poses1,
            self.cfg,
            avoid_xy=poses1[:2],
            preferred_y=0.35,
        )
