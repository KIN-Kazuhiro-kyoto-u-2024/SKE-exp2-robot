import argparse
import numpy as np
from qr_back_game.env import BackQrGameEnv
from qr_back_game.config import GameConfig


def run_episode(env):
    obs = env.reset()
    done = False
    info = {}
    reward = 0.0
    while not done:
        obs, reward, done, info = env.step(None)
    return info.get("winner"), info.get("step_count"), reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    env = BackQrGameEnv(GameConfig(max_steps=args.max_steps))
    winners, steps, rewards = [], [], []
    for _ in range(args.episodes):
        w, s, r = run_episode(env)
        winners.append(-1 if w is None else int(w))
        steps.append(int(s))
        rewards.append(float(r))
    env.close()
    print("episodes", args.episodes)
    print("winner_counts", {str(k): int(np.sum(np.array(winners) == k)) for k in [-1, 0, 1]})
    print("mean_steps", float(np.mean(steps)))
    print("mean_reward", float(np.mean(rewards)))


if __name__ == "__main__":
    main()
