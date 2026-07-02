from dataclasses import dataclass
import math


@dataclass
class GameConfig:
    # Field: 3 m x 2 m, discretized into 10 cm cells.
    # field_w: float = 3.0
    # field_h: float = 2.0
    # cell_size: float = 0.10

    # Field2: 1.8m x 0.9m discretized into 10cm cells.
    field_w: float = 1.8
    field_h: float = 0.9
    cell_size: float = 0.10

    # Robot body size: width 10 cm, length 15 cm.
    robot_w: float = 0.10
    robot_l: float = 0.15

    # Differential/unicycle-like action limits.
    dt: float = 0.10
    max_v: float = 0.55       # m/s
    max_omega: float = 3.20   # rad/s

    # Collision prediction is done with center circles.
    collision_radius: float = 0.11
    collision_pause_steps: int = 4

    # Winning condition: attacker is inside target's rear cone and sees target rear.
    rear_view_range: float = 0.85
    rear_sector_half_angle: float = math.radians(50.0)
    camera_fov_half_angle: float = math.radians(45.0)
    camera_range: float = 1.10

    # Episode.
    max_steps: int = 500

    # MPC.
    mpc_horizon: int = 6     # MPC が最適か計算で考慮する未来の応答の時間長さ（タイムステップ）
    mpc_num_sequences: int = 90
    mpc_seed: int = 7
    waypoint_lookahead: int = 3


    # Initial state. random_start=True is useful for evaluation/RL.
    random_start: bool = True
    start_margin: float = 0.20
    start_min_dist: float = 0.45

    # Residual RL scales. The PPO policy outputs values in [-1, 1].
    rl_residual_v_scale: float = 0.08
    rl_residual_omega_scale: float = 1.20
    rl_residual_penalty: float = 0.015
    close_distance_penalty_radius: float = 0.24

    @property
    def grid_w(self) -> int:
        return int(round(self.field_w / self.cell_size))

    @property
    def grid_h(self) -> int:
        return int(round(self.field_h / self.cell_size))
