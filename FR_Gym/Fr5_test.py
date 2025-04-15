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
import time
# from arguments import get_args
from imitation.algorithms import bc
from stable_baselines3.common.policies import ActorCriticPolicy

if __name__ == '__main__':
    # args, kwargs = get_args()
    env = FR5_Env(gui=True)
    env.render()
    # model = PPO.load('C:/Users/lyfly/Downloads/FR5_Reinforcement-learning-master/dagger_model.zip',env=env)
    policy = ActorCriticPolicy.load("dagger_model.zip")
    # model = bc.reconstruct_policy("dagger_models")
    test_num = 100  # 测试次数
    success_num = 0  # 成功次数
    print("测试次数：",test_num)
    for i in range(test_num):
        state,_ = env.reset()
        done = False 
        score = 0
        # time.sleep(3)
        step = 0
        while not done:
            # action = env.action_space.sample()     # 随机采样动作
            step += 1
            action, _ = policy.predict(observation=state,deterministic=True)
            # print("state:",state)
            # print("action:",action)
            state, reward, done, _,info = env.step(action=action)
            score += reward
            # env.render()
            # time.sleep(0.05)

        if info['is_success']:
            success_num += 1
        print("奖励：",score)
    success_rate = success_num/test_num
    print("成功率：",success_rate)
    env.close()

