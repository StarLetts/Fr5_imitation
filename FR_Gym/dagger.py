'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''


import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(r"FR5_Reinforcement-learning\utils")

from stable_baselines3 import A2C,PPO,DDPG,TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from Fr5_env import FR5_Env
from pathlib import Path
import time
import numpy as np
import tempfile
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies.serialize import load_policy, save_stable_model
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from imitation.util.util import save_policy
from stable_baselines3.common.policies import ActorCriticPolicy
# from arguments import get_args

now = time.strftime('%m%d-%H%M%S', time.localtime())
# args, kwargs = get_args()

# HACK
# models_dir = args.models_dir
# logs_dir = args.logs_dir
# checkpoints = args.checkpoints
# test = args.test

def make_env(i):
    def _init():
        if i == 0:
            env = FR5_Env(gui=True)
        else:
            env = FR5_Env(gui=False)
        # env = Monitor(env, logs_dir)
        env.render()
        env.reset()
        return env
    set_random_seed(0)
    return _init

if __name__ == '__main__':
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)
    # if not os.path.exists(logs_dir):    
    #     os.makedirs(logs_dir)
    # if not os.path.exists(checkpoints):
    #     os.makedirs(checkpoints)

    # Instantiate the env
    num_train = 10
    # env = make_vec_env("Robot", rng=np.random.default_rng(), post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],)  # needed for computing rollouts later
    env = SubprocVecEnv([make_env(i) for i in range(num_train)])
    # env=RolloutInfoWrapper(env)
    # env = DummyVecEnv([make_env() for i in range(num_train)])

    
    expert = load_policy("ppo",path="C:/Users/lyfly/Downloads/FR5_Reinforcement-learning-master/models/PPO/best_model.zip",venv=env,)
    # 评估专家数据
    # reward, _ = evaluate_policy(expert, env, 10)
    # print(reward)
    reward = 0
    while(reward < 1000):
        reward, _ = evaluate_policy(expert, env, 10)
        print(reward)

    rng=np.random.default_rng()
    # rollouts=rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=None, min_episodes=50), rng=rng, unwrap=False)
    # transitions=rollout.flatten_trajectories(rollouts)

    # # 检查rollout生成情况
    # print(f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    #         After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    #         The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    #         """
    #         )

    # 开始DAgger训练
    bc_trainer = bc.BC(observation_space=env.observation_space,action_space=env.action_space,rng=rng,)
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=np.random.default_rng(),
        )
    print(isinstance(dagger_trainer.policy, ActorCriticPolicy))
    dagger_trainer.train(20000)

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 20)
    print(reward)
    dagger_trainer.policy.save("dagger_model.zip")
    # save_policy(dagger_trainer.policy, "dagger_models")
