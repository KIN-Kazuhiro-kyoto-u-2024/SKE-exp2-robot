import argparse

import numpy as np
from stable_baselines3 import PPO

from qr_back_game.config import GameConfig
from qr_back_game.env import BackQrGameEnv
from qr_back_game.residual_env import ResidualPPOEnv


def eval_mpc_only(episodes, max_steps, seed):
    cfg = GameConfig(random_start=True, max_steps=max_steps)
    env = BackQrGameEnv(cfg=cfg, render_mode=None)
    counts = {0: 0, 1: 0, "draw": 0}
    steps_list = []
    for ep in range(episodes):
        env.seed(seed + ep)
        env.reset()
        done = False
        info = {"winner": None}
        step = 0
        while not done and step < max_steps:
            _, _, done, info = env.step(None)
            step += 1
        winner = info.get("winner")
        if winner in (0, 1):
            counts[winner] += 1
        else:
            counts["draw"] += 1
        steps_list.append(step)
    env.close()
    return counts, steps_list


def eval_residual(model_path, episodes, max_steps, seed, render=False):
    cfg = GameConfig(random_start=True, max_steps=max_steps)
    env = ResidualPPOEnv(cfg=cfg, render_mode="human" if render else None)
    model = PPO.load(model_path)
    counts = {0: 0, 1: 0, "draw": 0}
    steps_list = []
    for ep in range(episodes):
        env.seed(seed + ep)
        obs = env.reset()
        done = False
        info = {"winner": None}
        step = 0
        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            if render:
                env.render()
            step += 1
        winner = info.get("winner")
        if winner in (0, 1):
            counts[winner] += 1
        else:
            counts["draw"] += 1
        steps_list.append(step)
        print(f"Episode {ep + 1:03d}: winner={winner}, steps={step}")
    env.close()
    return counts, steps_list


def print_result(name, counts, steps):
    episodes = sum(counts.values())
    print(f"\n=== {name} ===")
    print(f"Episodes : {episodes}")
    print(f"Robot 0  : {counts[0]} wins")
    print(f"Robot 1  : {counts[1]} wins")
    print(f"Draw     : {counts['draw']}")
    print(
        f"Robot0 win rate excl. draws : {counts[0] / max(counts[0] + counts[1], 1):.3f}"
    )
    print(f"Robot0 win rate incl. draws : {counts[0] / max(episodes, 1):.3f}")
    print(f"Average steps : {float(np.mean(steps)):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/residual_ppo_robot0.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if not args.skip_baseline:
        b_counts, b_steps = eval_mpc_only(args.episodes, args.max_steps, args.seed)
        print_result("MPC only baseline", b_counts, b_steps)

    r_counts, r_steps = eval_residual(
        args.model, args.episodes, args.max_steps, args.seed, render=args.render
    )
    print_result("robot0 = MPC + residual PPO, robot1 = MPC only", r_counts, r_steps)


if __name__ == "__main__":
    main()
