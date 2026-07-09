import argparse

from shimmy import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from qr_back_game.config import GameConfig
from qr_back_game.residual_enemy_env import ResidualPPOWithEnemyEnv


class EnvForTrain(ResidualPPOWithEnemyEnv):
    def __init__(
        self,
        cfg: GameConfig | None = None,
        render_mode: str | None = None,
        enemy_model=None,
    ):
        super().__init__(cfg, render_mode)
        self.enemy_model = enemy_model

    def step(self, rl_action, enemy_action=None):
        if self.enemy_model is None:
            return super().step(rl_action)

        enemy_obs = self._get_obs(robot=1)
        enemy_action, _ = self.enemy_model.predict(enemy_obs, deterministic=True)
        (obs, enemy_obs), (reward, enemy_reward), done, info = super().step(
            rl_action, enemy_action
        )
        return obs, reward, done, info


def make_env(seed=None, enemy_model=None):
    def _init():
        cfg = GameConfig(random_start=True)
        # cfg = GameConfig(random_start=True, enemy_moves=False)

        # python3.10 に移行するため使用ライブラリを変更（富田追加） #
        # GymV21CompatibilityV0 で gym 環境から gymnasium 環境に変換できる
        raw_env = EnvForTrain(cfg=cfg, render_mode=None, enemy_model=enemy_model)
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
    parser.add_argument("--model", type=str, default="models/residual_ppo_robot0.zip")
    args = parser.parse_args()

    enemy_model = PPO.load(args.model)

    env = DummyVecEnv([make_env(args.seed, enemy_model)])
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
