import argparse

from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from qr_back_game.config import GameConfig
from qr_back_game.residual_env import ResidualPPOEnv


def make_env(seed=None):
    def _init():
        cfg = GameConfig(random_start=True)

        # python3.10 に移行するため使用ライブラリを変更（富田追加） #
        # GymV21CompatibilityV0 で gym 環境から gymnasium 環境に変換できる
        raw_env = ResidualPPOEnv(cfg=cfg, render_mode=None)
        env = GymV21CompatibilityV0(env=raw_env)
        #########################################################

        if seed is not None:
            env.seed(seed)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="models/residual_ppo_robot0")
    args = parser.parse_args()

    env = DummyVecEnv([make_env(args.seed)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
