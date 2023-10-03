from functools import cached_property

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv, recover_action
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.laptop_env import LaptopEnv
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info

class LaptopRLEnv(LaptopEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="allegro_hand_free", rotation_reward_weight=0, 
                object_pose_noise=0.01, constant_object_state=False, randomness_scale=1, friction=1, zero_joint_pos=0,
                 **renderer_kwargs):
        super().__init__(use_gui, frame_skip, **renderer_kwargs)
        if robot_name != "mano":
            self.setup(robot_name)
            # NOTE: For using allegro without retargeting!
            self.rl_step = self.simple_sim_step
        else:
            self.mano_setup(frame_skip, zero_joint_pos)
        self.rotation_reward_weight = rotation_reward_weight
        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise

        # Parse link name
        if self.is_robot_free:
            if robot_name == "mano":
                info = generate_free_robot_hand_info()["mano_hand_free"]
            else:
                info = generate_free_robot_hand_info()[robot_name]
        else:
            info = generate_arm_robot_hand_info()[robot_name]
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        self.laptop_link = [link for link in self.laptop.get_links() if link.get_name() == "link_0"][0]
        self.is_contact = False
        self.is_unlock = False   

    def mano_setup(self, frame_skip, zero_joint_pos):
        self.robot_name = "mano"
        self.mano_robot = MANORobotHand(self.scene, self.renderer, init_joint_pos=zero_joint_pos,
                            control_interval=frame_skip * self.scene.get_timestep(), full_dof=True,
                            scale=1)
        self.robot = self.mano_robot.robot
        self.is_robot_free = True
        if self.is_robot_free:
            info = generate_free_robot_hand_info()["mano_hand_free"]
            # velocity_limit = np.array([1.0] * 3 + [1.57] * 3 + [3.14] * (self.robot.dof - 6))
            # self.velocity_limit = np.stack([-velocity_limit, velocity_limit], axis=1)
            init_pose = sapien.Pose(np.array([-0.3, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
            self.robot.set_pose(init_pose)
            self.arm_dof = 0

        self.robot_info = info
        self.robot_collision_links = [link for link in self.robot.get_links() if len(link.get_collision_shapes()) > 0]
        self.control_time_step = self.scene.get_timestep() * self.frame_skip

        # Choose different step function
        if self.is_robot_free:
            self.rl_step = self.mano_sim_step
            # self.rl_step = self.free_sim_step
        # else:
        #     self.rl_step = self.arm_sim_step

        # Scene light and obs
        if self.use_visual_obs:
            self.get_observation = self.get_visual_observation
            if not self.no_rgb:
                add_default_scene_light(self.scene, self.renderer)
        else:
            self.get_observation = self.get_oracle_state                

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        laptop_qpos_vec = self.laptop.get_qpos()
        laptop_qvel_vec = self.laptop.get_qvel()
        lid_pose = self.laptop_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        self.is_contact = self.check_contact(self.robot.get_links(), [self.laptop_link])
        # return np.concatenate(
        #     [robot_qpos_vec, palm_pose.p, palm_pose.q, palm_v, palm_w, laptop_qpos_vec, lid_pose.p])
        # return np.concatenate([robot_qpos_vec, palm_pose.p, palm_pose.q, laptop_qpos_vec, lid_pose.p, [int(self.is_contact)]])
        return np.concatenate(
            [robot_qpos_vec, palm_pose.p, palm_pose.q, palm_v, palm_w, laptop_qpos_vec,
            lid_pose.p, [int(self.is_contact)]])


    def get_proprioception(self): # TODO: helin: do not understand.
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate(
            [robot_qpos_vec, palm_pose.p])

    # TODO: Change this part for DAPG
    def get_reward(self, action):
        laptop_qpos = self.laptop.get_qpos()[0]
        lid_pose = self.laptop_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.is_contact

        reward = (0.2 - laptop_qpos)
        reward += -0.1 * min(np.linalg.norm(palm_pose.p - lid_pose.p), 0.5)
        if is_contact:
            reward += 0.1

        # TODO: DONT FORGET THIS
        reward = 0
        return reward

    def reset(self):
        self.reset_internal()
        self.is_contact = False
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        return self.robot.dof + 2 + 6 + 3 # holly shit, this should be 33 instead of 34

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 250


def main_env():
    # from hand_teleop.dapg.dapg_wrapper import DAPGWrapper
    from time import time
    base_env = LaptopRLEnv(use_gui=True, robot_name="allegro_hand_free")
    robot_dof = base_env.robot.dof
    # env = DAPGWrapper(base_env)
    env = base_env
    env.seed(0)
    env.reset()

    tic = time()
    env.reset()
    tac = time()
    print(f"Reset time: {(tac - tic) * 1000} ms")

    tic = time()
    for i in range(1000):
        action = np.random.rand(robot_dof) * 2 - 1
        action[2] = 0.1
        obs, reward, done, _ = env.step(action)
    tac = time()
    print(f"Step time: {(tac - tic)} ms")

    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    add_default_scene_light(base_env.scene, base_env.renderer)
    base_env.viewer = viewer

    env.reset()
    for i in range(1000):
        action = np.ones(robot_dof) * 0
        action[2] = 0.1
        obs, reward, done, _ = env.step(action)
        base_env.render()

    viewer.toggle_pause(True)
    while not viewer.closed:
        base_env.simple_step()
        base_env.render()


if __name__ == '__main__':
    main_env()        