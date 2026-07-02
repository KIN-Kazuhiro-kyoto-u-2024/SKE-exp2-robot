# QR Back-Taking Two-Robot Game

1.8m x 0.9m (or 3 m x 2 m) field, 10 cm grid, two 10 cm x 15 cm robots. QR decoding is intentionally omitted; victory is judged geometrically by whether one robot enters the rear sector of the other robot and sees its rear marker.

## Run

```bash
python3.8 -m pip install -r requirements.txt
cd qr_back_game
python run_sim.py
```

Headless evaluation:

```bash
python evaluate.py --episodes 20 --max-steps 500
```

## Main design

- `BackQrGameEnv`: Gym 0.19 compatible environment.
- `path_planning.py`: A* over a 18 x 9 (or 30 x 20) grid. The planner is 4-connected only, so waypoints are adjacent 10 cm cell centers and the path never cuts diagonally between cells. Goal cells are sampled in the opponent rear fan, then snapped to grid cells.
- `mpc.py`: Sampling MPC. Loss includes grid-path tracking distance, remaining distance, heading error, collision risk, control smoothness, forward progress, and an explicit retreat bonus when the robot is inside a collision danger margin.
- `env.py`: Applies the selected MPC/RL actions directly. The previous collision-prediction safety filter has been removed, so there is no automatic pause, stop, reverse, or evasive action override.
- `geometry.py`: rear sector, camera field of view, unicycle dynamics, and center-circle collision prediction.
- `rendering.py`: pygame renderer. Body protrusion beyond the field is allowed conceptually; the simulator only clips the robot center inside the field.

## Notes for real robot transfer

The output action is `[forward_velocity_mps, yaw_rate_radps]` for each robot. For an actual differential-drive robot, convert it with:

```text
left_wheel_speed  = v - yaw_rate * wheel_base / 2
right_wheel_speed = v + yaw_rate * wheel_base / 2
```

Tune `GameConfig.max_v`, `max_omega`, `collision_radius`, `rear_view_range`, and the MPC weights before deploying to hardware. For real robots, add a separate low-level velocity limiter and emergency stop if collision prevention is needed; this simulator version no longer performs automatic collision-prediction stopping.

## Residual PPO experiment

This version includes a single-agent residual RL setup:

- robot0: A* + MPC + PPO residual
- robot1: A* + MPC only

Train robot0 residual policy:

```bash
python train_residual_ppo.py --timesteps 100000 --seed 0 --save models/residual_ppo_robot0
```

Evaluate 100 randomized episodes against the MPC-only baseline:

```bash
python evaluate_residual.py --model models/residual_ppo_robot0.zip --episodes 100 --max-steps 500 --seed 1000
```

Render one trained episode:

```bash
python run_residual_sim.py --model models/residual_ppo_robot0.zip --seed 0
```

The PPO action is a normalized residual `[dv, domega]` in `[-1, 1]`, scaled by
`GameConfig.rl_residual_v_scale` and `GameConfig.rl_residual_omega_scale`.
The base environment still has no collision-prediction safety filter; close-range
penalties are only used as RL rewards, not as action overrides.
