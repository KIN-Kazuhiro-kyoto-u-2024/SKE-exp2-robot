import argparse
import time

from stable_baselines3 import PPO

# ここは residual_env.py の実際のクラス名に合わせて変更してください
# 例: class ResidualPPOEnv(...) なら下の行でOK
from qr_back_game.residual_env import ResidualPPOEnv as ResidualEnv


def run_one_episode(model, seed, args):
    env = ResidualEnv(render_mode="human")

    try:
        env.seed(seed)
    except AttributeError:
        # print('env has no attribute "seed"')
        pass

    obs = env.reset()
    done = False
    info = {}

    dt_wall = 1.0 / max(args.fps, 1e-6)

    print()
    print("=" * 60)
    print(f"Seed {seed}")
    print("=" * 60)

    for step in range(args.max_steps):

        action, _ = model.predict(obs, deterministic=args.deterministic)

        obs, reward, done, info = env.step(action)

        try:
            env.render()
        except AttributeError:
            # print('env has no attribute "render"')
            pass

        print(
            f"seed={seed}, "
            f"step={step + 1:03d}, "
            f"reward={reward:.3f}, "
            f"done={done}, "
            f"winner={info.get('winner', None)}"
        )

        time.sleep(dt_wall)

        if done:
            print()
            print("=== Episode finished ===")
            print(f"seed  : {seed}")
            print(f"winner: {info.get('winner', None)}")
            print(f"steps : {step + 1}")
            print(f"reward: {reward:.3f}")

            if args.hold_seconds > 0:
                print(f"holding window for {args.hold_seconds} seconds...")
                time.sleep(args.hold_seconds)

            break

    if not done:
        print()
        print("=== Episode reached max_steps ===")
        print(f"seed  : {seed}")
        print(f"winner: {info.get('winner', None)}")
        print(f"steps : {args.max_steps}")

        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)

    try:
        env.close()
    except AttributeError:
        # print('env has no attribute "close"')
        pass

    return {
        "seed": seed,
        "winner": info.get("winner", None),
        "steps": step + 1,
        "done": done,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model)

    results = []

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        result = run_one_episode(model, seed, args)
        results.append(result)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    robot0_wins = sum(1 for r in results if r["winner"] == 0)
    robot1_wins = sum(1 for r in results if r["winner"] == 1)
    draws = len(results) - robot0_wins - robot1_wins

    for r in results:
        print(
            f"seed={r['seed']:03d}, "
            f"winner={r['winner']}, "
            f"steps={r['steps']}, "
            f"done={r['done']}"
        )

    print()
    print(f"Robot0 wins: {robot0_wins}")
    print(f"Robot1 wins: {robot1_wins}")
    print(f"Draws      : {draws}")


if __name__ == "__main__":
    main()
