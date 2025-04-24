'''
 @Author: Prince Wang 
 @Date: 2024-02-22 
 @Last Modified by:   Prince Wang 
 @Last Modified time: 2023-10-24 23:04:04 
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random
from reward import grasp_reward

from math import radians, sin, cos



def set_axes_equal(ax):
# 这一段是copy别人的。用处不是很大。
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def dh_matrix(alpha, a, d, theta):
    # 传入四个DH参数，根据公式3-6，输出一个T矩阵。
    alpha = alpha / 180 * np.pi
    theta = theta / 180 * np.pi
    matrix = np.identity(4)
    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta)*cos(alpha)
    matrix[0, 2] = sin(theta)*sin(alpha)
    matrix[0, 3] = a*cos(theta)
    matrix[1, 0] = sin(theta)
    matrix[1, 1] = cos(theta)*cos(alpha)
    matrix[1, 2] = -cos(theta)*sin(alpha)
    matrix[1, 3] = a*sin(theta)
    matrix[2, 0] = 0
    matrix[2, 1] = sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = d
    matrix[3, 0] = 0
    matrix[3, 1] = 0
    matrix[3, 2] = 0
    matrix[3, 3] = 1
    return matrix
    
joint_num = 6

# --- Robotic Arm construction ---
# DH参数表，分别用一个列表来表示每个关节的东西。
joints_alpha = [90, 0, 0, 90, -90, 0]
joints_a = [0, -425, -395, 0, 0, 0]
joints_d = [152, 0.0, 0.4, 102, 102, 244]
joints_theta = [0, 0, 0, 0, 0, 0]


class FR5_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,gui = False):
        super(FR5_Env).__init__()
        self.step_num = 0
        self.Con_cube = None
        # self.last_success = False

        # 设置最小的关节变化量
        low_action = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
        high_action = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1,12),dtype=np.float32)
        high = np.ones((1,12),dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 初始化pybullet环境
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else :
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        # self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 初始化环境
        self.init_env()

    def init_env(self):
        '''
            仿真环境初始化
        '''
        # boxId = self.p.loadURDF("plane.urdf")
        # 创建机械臂
        self.fr5 = self.p.loadURDF("C:/Users/lyfly/Downloads/FR5_Reinforcement-learning-master/fr5_description/urdf/fr5v6.urdf",useFixedBase=True, basePosition=[0, 0, 0],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),flags = p.URDF_USE_SELF_COLLISION)

        # 创建桌子
        self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # 创建目标
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                          radius=0.02,height = 0.05)
        self.target = self.p.createMultiBody(baseMass=0,  # 质量
                           baseCollisionShapeIndex=collisionTargetId,
                           basePosition=[0.5, 0.5, 2]) 
        
        # 创建目标杯子的台子
        collisionTargetId = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                            radius=0.03,height = 0.3)
        self.targettable = self.p.createMultiBody(baseMass=0,  # 质量
                            baseCollisionShapeIndex=collisionTargetId,
                            basePosition=[0.5, 0.5, 2])                                                          

    def step(self, action):
        '''step'''
        info = {}
        # Execute one time step within the environment
        # 初始化关节角度列表
        joint_angles = []

        # 获取每个关节的状态
        for i in [1,2,3,4,5,6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angle = joint_info[0]  # 第一个元素是当前关节角度
            joint_angles.append(joint_angle)

        # 执行action
        Fr5_joint_angles = np.array(joint_angles)+(np.array(action[0:6])/180*np.pi)
        gripper = np.array([0,0])
        anglenow = np.hstack([Fr5_joint_angles,gripper])
        p.setJointMotorControlArray(self.fr5,[1,2,3,4,5,6,8,9],p.POSITION_CONTROL,targetPositions=anglenow)
        
        for _ in range(20):
            self.p.stepSimulation()
            # time.sleep(1./240.)

        self.reward,info = grasp_reward(self)
        
        # observation计算
        self.get_observation()

        self.step_num += 1

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        '''重置环境参数'''
        self.step_num = 0
        self.reward = 0
        self.terminated = False
        self.success = False
        # 重新设置机械臂的位置
        neutral_angle =[ -49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118,-49.45849125928217,0,0,0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        p.setJointMotorControlArray(self.fr5,[1,2,3,4,5,6,8,9],p.POSITION_CONTROL,targetPositions=neutral_angle)

        # # 重新设置目标位置
        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.6, 0.8, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.targettable_position = [self.goalx[0], self.goaly[0], self.goalz[0]-0.175]
        self.p.resetBasePositionAndOrientation(self.targettable,self.targettable_position, [0, 0, 0, 1])
        self.p.resetBasePositionAndOrientation(self.target,self.target_position, [0, 0, 0, 1])
        
        
        for i in range(100):
            self.p.stepSimulation()
            # time.sleep(10./240.)

        self.get_observation()
        
        
        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        return self.observation,infos

    def get_observation(self,add_noise = False):
        """计算observation"""
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])
        
        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        # print([Gripper_posx, Gripper_posy,Gripper_posz])
        gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position

        joint_angles = [0,0,0,0,0,0]
        for i in [1,2,3,4,5,6]:
            joint_info = p.getJointState(self.fr5, i)
            joint_angles[i-1]  = joint_info[0]*180/np.pi  # 第一个元素是当前关节角度
            if add_noise == True:
                joint_angles[i-1] = self.add_noise(joint_angles[i-1],range=0,gaussian=False)
        # print("joint_angles",str(joint_angles))
        # print("gripper_centre_pos",str(gripper_centre_pos))

        # 计算夹爪的朝向
        gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)

        # 计算obs
        obs_joint_angles = ((np.array(joint_angles,dtype=np.float32)/180)+1)/2
        
        joint_hm = []
        for i in [1, 2, 3, 4, 5, 6]:        
            joint_hm.append(dh_matrix(alpha=joints_alpha[i-1], a=joints_a[i-1], d=joints_d[i-1], theta=joints_theta[i-1]+joint_angles[i-1]))

        # -----------连乘计算----------------------
        for i in range(joint_num-1):
            joint_hm[i+1] = np.dot(joint_hm[i], joint_hm[i+1])    
        # Prepare the coordinates for plotting
        # for i in range(joint_num):
        #     print(np.round(joint_hm[i][:3, 3], 5))
        # 获取坐标值
        X = [hm[0, 3] for hm in joint_hm]
        Y = [hm[1, 3] for hm in joint_hm]
        Z = [hm[2, 3] for hm in joint_hm]
        x = -X[5]/1000
        y = -Y[5]/1000
        z = Z[5]/1000

        # print(gripper_centre_pos)
        # print("X:", x)
        # print("Y:", y)
        # print("Z:", z)
        gripper_centre_pos = np.array([x, y, z])

        # gripper_centre_pos[0] = self.add_noise(gripper_centre_pos[0],range=0.005,gaussian=True)
        # gripper_centre_pos[1] = self.add_noise(gripper_centre_pos[1],range=0.005,gaussian=True)
        # gripper_centre_pos[2] = self.add_noise(gripper_centre_pos[2],range=0.005,gaussian=True)
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0]+0.922)/1.844,
                                           (gripper_centre_pos[1]+0.922)/1.844,
                                           (gripper_centre_pos[2]+0.5)/1],dtype=np.float32)
        
        obs_gripper_orientation = (np.array([gripper_orientation[0],gripper_orientation[1],gripper_orientation[2]],dtype=np.float32)+180)/360
        
        self.target_position = np.array(p.getBasePositionAndOrientation(self.target)[0])

        obs_target_position = np.array([(self.target_position[0]+0.2)/0.4,
                                        (self.target_position[1]-0.6)/0.2,
                                        (self.target_position[2]-0.1)/0.2],dtype=np.float32)

        self.observation = np.hstack((obs_gripper_centre_pos,obs_joint_angles,obs_target_position),dtype=np.float32).flatten()

        self.observation = self.observation.flatten()
        self.observation = self.observation.reshape(1,12)
        # self.observation = np.hstack((np.array(joint_angles,dtype=np.float32),target_delta_position[0]),dtype=np.float32)


    def render(self):
        '''设置观察角度'''
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0, cameraYaw=90, cameraPitch=-7.6, cameraTargetPosition=[0.39, 0.45, 0.42])
    
    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        '''添加噪声'''
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-10, 10)
        return angle
    
    def _check_grasp_success(self):
        Gripper_posx = p.getLinkState(self.fr5, 6)[0][0]
        Gripper_posy = p.getLinkState(self.fr5, 6)[0][1]
        Gripper_posz = p.getLinkState(self.fr5, 6)[0][2]
        relative_position = np.array([0, 0, 0.15])
        
        # 固定夹爪相对于机械臂末端的相对位置转换
        rotation = R.from_quat(p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        # print([Gripper_posx, Gripper_posy,Gripper_posz])
        gripper_centre_pos = [Gripper_posx, Gripper_posy,Gripper_posz] + rotated_relative_position
        object_pos = self.target_position
        ee_pos = gripper_centre_pos
        gripper_orientation = p.getLinkState(self.fr5, 7)[1]
        gripper_orientation = R.from_quat(gripper_orientation)
        gripper_orientation = gripper_orientation.as_euler('xyz', degrees=True)
        dist = np.linalg.norm(np.array(object_pos) - np.array(gripper_centre_pos))
        
        # 距离接近 + 物体抬高
        close_enough = dist < 0.05
        # lifted = object_pos[2] > 0.12
        return close_enough

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env 
    Env = FR5_Env(gui=True)
    Env.reset()
    check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
    time.sleep(100)
