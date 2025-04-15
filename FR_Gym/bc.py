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
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy, save_stable_model
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout
from imitation.algorithms import bc
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback,CallbackList,BaseCallback,CheckpointCallback
from Callback import TensorboardCallback
from loguru import logger
from imitation.util.util import save_policy
# from utils.arguments import get_args

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
    env = SubprocVecEnv([make_env(i) for i in range(num_train)])
    # env=RolloutInfoWrapper(env)
    # env = DummyVecEnv([make_env() for i in range(num_train)])
    
    expert = load_policy("ppo",path="C:/Users/lyfly/Downloads/FR5_Reinforcement-learning-master/models/PPO/best_model.zip",venv=env,)
    # 评估专家数据
    reward = 0
    while(reward < 1000):
        reward, _ = evaluate_policy(expert, env, 10)
        print(reward)

    rng=np.random.default_rng()
    rollouts=rollout.rollout(expert, env, rollout.make_sample_until(min_timesteps=None, min_episodes=100), rng=rng, unwrap=False)
    transitions=rollout.flatten_trajectories(rollouts)

    print(f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
            After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
            The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
            """
            )

    bc_trainer = bc.BC(observation_space=env.observation_space,action_space=env.action_space,demonstrations=transitions,rng=rng,)
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")

    bc_trainer.train(n_epochs=1000)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")
    save_policy(bc_trainer.policy, "bc_models")