import gymnasium as gym
import numpy as np
import os
import pickle
from tqdm import trange
import pybullet as p
from scipy.spatial.transform import Rotation as R
from Fr5_env import FR5_Env # 根据你的实际模块路径导入


if __name__ == '__main__':
    # 初始化环境
    env = FR5_Env(gui=True)
    obs, _ = env.reset(seed=0)

    trajectories = []
    num_episodes = 10
    Gripper_posx = p.getLinkState(env.fr5, 6)[0][0]
    Gripper_posy = p.getLinkState(env.fr5, 6)[0][1]
    Gripper_posz = p.getLinkState(env.fr5, 6)[0][2]
    relative_position = np.array([0, 0, 0.15])
    gripper_orientation = p.getLinkState(env.fr5, 7)[1]
    gripper_orientation = R.from_quat(gripper_orientation)
    gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
    # 固定夹爪相对于机械臂末端的相对位置转换
    rotation = R.from_quat(p.getLinkState(env.fr5, 7)[1])
    rotated_relative_position = rotation.apply(relative_position)
    # print([Gripper_posx, Gripper_posy,Gripper_posz])
    gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position

    print("gripper center pose:", gripper_centre_pos)
    for ep in range(num_episodes):
        episode = {"observations": [], "actions": []}
        obs, _ = env.reset()
        
        for _ in range(200):
            ee_pos = p.getLinkState(env.fr5, 6)[0]
            target_pos = env.target_position

            # 使用逆运动学计算目标位置对应的关节角
            target_joint_angles = p.calculateInverseKinematics(env.fr5, 6, target_pos)
            target_joint_angles = target_joint_angles[:6]  # 取前6个关节

            # 获取当前关节角
            current_joint_angles = [p.getJointState(env.fr5, i)[0] for i in [1,2,3,4,5,6]]

            # 计算动作：角度增量（单位是度）
            delta_joint = (np.array(target_joint_angles) - np.array(current_joint_angles)) * 180 / np.pi
            action = np.clip(delta_joint, -5, 5)  # 限制动作幅度（单位：度）

            # 补齐为6维（已有），无需额外拼接
            episode["observations"].append(obs)
            episode["actions"].append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            
            # 可选的抓取条件（根据你的环境设置）
            if terminated or env._check_grasp_success():
                break

        trajectories.append(episode)
        print(f"Episode {ep + 1} collected. Steps: {len(episode['actions'])}")

    # 保存数据
    save_path = "./expert_demos.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Expert data saved to {save_path}")
