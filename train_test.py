import os
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
from imitation.algorithms.bc import BC
from Fr5_env import FR5_Env  # 导入你的自定义环境

# 加载专家数据
with open("/home/hn/FR5_Reinforcement-learning-master/FR_Gym/expert_demos.pkl", "rb") as f:
    expert_trajectories = pickle.load(f)

# 转换为 imitation 库需要的 Trajectory 格式
trajectories = []
for episode in expert_trajectories:
    obs = np.array(episode["observations"])
    acts = np.array(episode["actions"])
    obs = obs.squeeze()
    if len(obs.shape) != 2:
        raise ValueError(f"obs shape 错误: {obs.shape}, 应该是二维的 (T, obs_dim)")
    if len(obs) == len(acts):
        obs = np.vstack([obs, obs[-1]])  # 补上最后一个 observation（复制上一个）

    traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
    trajectories.append(traj)

# 初始化环境（注意 imitation 需要向量化环境）
def make_env():
    return FR5_Env(gui=True)

venv = DummyVecEnv([make_env])

# 创建一个用于初始化的模型（必须使用相同的策略架构）
rng = np.random.default_rng(0)
bc_trainer = BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=trajectories,
    rng=rng,
    batch_size=32,
    ent_weight=1e-3,
)

# 训练行为克隆模型
bc_trainer.train(n_epochs=20000)

# 保存模型
bc_trainer.policy.save("bc_policy.zip")
print("✅ 行为克隆模型已保存。")

# 使用保存的模型进行评估
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO("MlpPolicy", venv, verbose=0)
model.policy = bc_trainer.policy  # 替换策略为BC学到的策略

mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=10)
print(f"�� 在专家数据上的平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
