import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from Fr5_env import FR5_Env
import time
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
# === Step 1: 构建环境 ===
def make_env(rank):
    def _init():
        env = FR5_Env(gui=(rank == 0))
        env = RolloutInfoWrapper(env)  # 关键：添加这个 wrapper！
        return env
    set_random_seed(rank)
    return _init

if __name__ == '__main__':
    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])


    # === Step 2: 加载专家模型 ===
    expert = PPO.load("C:/Users/lyfly/Downloads/FR5_Reinforcement-learning-master/models/PPO/best_model.zip")

    # === Step 3: 收集专家数据 ===
    rng = np.random.default_rng()

    # 使用 imitation 的 rollout 工具
    trajectories = rollout.rollout(
        policy=expert,
        venv=env,
        sample_until=rollout.make_sample_until(min_timesteps=50000, min_episodes=50),
        rng=rng,
        deterministic_policy=True,
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=None, 
        hid_sizes=[256, 256] 
    )
    # === Step 4: 初始化 GAIL ===
    learner = GAIL(
        demonstrations=trajectories,
        venv=env,
        gen_algo=PPO("MlpPolicy", env, verbose=1,learning_rate=3e-4, n_steps=2048, batch_size=256),
        reward_net=reward_net,
        n_disc_updates_per_round=2,
        demo_batch_size=2048,
        allow_variable_horizon=True
    )

    # === Step 5: 训练 GAIL 生成器策略 ===
    learner.train(20000000)

    # === Step 6: 保存模型 ===
    learner.gen_algo.save("gail_FR5_model.zip")


    gail_model = learner.gen_algo
    mean_reward, std_reward = evaluate_policy(gail_model, env, n_eval_episodes=20)
    print(f"GAIL Policy mean reward: {mean_reward} ± {std_reward}")