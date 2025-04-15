'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import sys
sys.path.append(r"FR5_Reinforcement-learning\utils")
sys.path.append("FR5_Reinforcement-learning\FR_Gym")

from stable_baselines3 import A2C,PPO,DDPG,TD3
from Fr5_env import FR5_Env
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data import rollout
import gymnasium as gym
import time
import random
import numpy as np
import torch
from arguments import get_args


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    args, kwargs = get_args()
    env = FR5_Env(gui=args.gui)
    env.render()
    model = PPO.load(args.model_path)
    test_num = args.test_num  # 测试次数
    success_num = 0  # 成功次数
    print("测试次数：",test_num)
    states=[]
    actions=[]
    for i in range(test_num):
        state,_ = env.reset()
        done = False 
        score = 0
        # time.sleep(3)
        
        while not done:
            # action = env.action_space.sample()     # 随机采样动作
            action, _ = model.predict(observation=state,deterministic=True)
            print("state:",state)
            print("action:",action)
            states.append(state)
            actions.append(action)
            state, reward, done, _,info = env.step(action=action)
            score += reward
            # env.render()
            time.sleep(0.01)

        if info['is_success']:
            success_num += 1
        print("奖励：",score)
    success_rate = success_num/test_num
    print("成功率：",success_rate)
    env.close()
    n_samples = 30  # 采样30个数据
    states = np.array(states)
    actions = np.array(actions)
    random_index = random.sample(range(states.shape[0]), n_samples)
    expert_s = states[random_index]
    expert_a = actions[random_index]

    ## 测试内容
    expert = rollout.rollout()

