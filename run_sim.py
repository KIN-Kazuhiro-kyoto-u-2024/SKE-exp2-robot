import argparse
import time

from qr_back_game.config import GameConfig
from qr_back_game.env import BackQrGameEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = GameConfig(max_steps=args.max_steps)
    env = BackQrGameEnv(cfg=cfg, render_mode=None if args.no_render else "human")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            obs, reward, done, info = env.step(None)  # built-in A* + MPC
            if not args.no_render:
                env.render("human")
                time.sleep(cfg.dt)
        print(
            {
                "episode": ep,
                "winner": info.get("winner"),
                "steps": info.get("step_count"),
                "reward": reward,
            }
        )
    env.close()


if __name__ == "__main__":
    main()
