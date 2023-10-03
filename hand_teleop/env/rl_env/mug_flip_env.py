from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
import transforms3d

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.mug_flip_env import MugFlipEnv
from hand_teleop.real_world import lab
from hand_teleop.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand


class MugFlipRLEnv(MugFlipEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", constant_object_state=False,
                 object_scale=1.0, randomness_scale=1, friction=1, object_pose_noise=0.01, zero_joint_pos=None, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, object_scale, randomness_scale, friction, **renderer_kwargs)

        if robot_name != "mano":
            self.setup(robot_name)
        else:
            self.mano_setup(frame_skip, zero_joint_pos)

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
        if "allegro" in self.robot_name:
            self.thumb_tip_link = [link for link in self.robot.get_links() if link.get_name() == "link_15.0_tip"][0]
            self.thumb_link = [link for link in self.robot.get_links() if link.get_name() == "link_15.0"][0]
            self.finger_links = [self.thumb_link, self.thumb_tip_link]
        else:
            self.finger_links = self.robot.get_links()

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()

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

        # Scene light and obs
        if self.use_visual_obs:
            self.get_observation = self.get_visual_observation
            if not self.no_rgb:
                add_default_scene_light(self.scene, self.renderer)
        else:
            self.get_observation = self.get_oracle_state

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.object_episode_init_pose if self.constant_object_state else self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        palm_pose = self.palm_link.get_pose()
        object_in_palm = object_pose.p - palm_pose.p
        v = self.manipulated_object.get_velocity()
        w = self.manipulated_object.get_angular_velocity()
        return np.concatenate([robot_qpos_vec, object_pose_vec, v, w, object_in_palm, np.array([theta_cos])])

    def get_reward(self, action):
        object_pose = self.manipulated_object.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.check_contact(self.finger_links, [self.manipulated_object])
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        lift_cos = max(theta_cos, 0)
        obj_target_distance = min(np.linalg.norm(self.original_object_pos[:2] - object_pose.p[:2]), 0.1)

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        if is_contact:
            reward += 0.1
            reward += lift_cos
            if lift_cos > 0.6:
                reward += 2
                reward -= obj_target_distance * 20

        if lift_cos > 0.9 and obj_target_distance < 0.08:
            reward += 5

        table_hand_collision = self.check_contact(self.robot_collision_links, [self.table])
        reward += table_hand_collision * -1
        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # super().reset(seed=seed) helin
        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        else:
            init_pose = sapien.Pose(np.array([-0.3, 0, 0.2]), transforms3d.euler.euler2quat(0, np.pi / 2, 0))
        self.robot.set_pose(init_pose)
        self.reset_internal()
        # NOTE: No randomness for now.
        # self.object_episode_init_pose = self.manipulated_object.get_pose()
        # random_quat = transforms3d.euler.euler2quat(*(self.np_random.randn(3) * self.object_pose_noise * 10))
        # random_pos = self.np_random.randn(3) * self.object_pose_noise
        # self.object_episode_init_pose = self.object_episode_init_pose * sapien.Pose(random_pos, random_quat)
        return self.get_observation()

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        # return np.concatenate( [robot_qpos_vec, palm_pose.p])
        return np.concatenate([robot_qpos_vec, palm_pose.p, palm_pose.q])

    @cached_property
    def obs_dim(self):
        return self.robot.dof + 7 + 6 + 3 + 1

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 200


def main_env():
    # from hand_teleop.dapg.dapg_wrapper import DAPGWrapper
    from time import time
    # base_env = MugFlipRLEnv(use_gui=True, robot_name="allegro_hand_xarm6_wrist_mounted_rotate")
    env = MugFlipRLEnv(use_gui=True, robot_name="allegro_hand_xarm6_wrist_mounted_face_front")
    base_env = env
    robot_dof = base_env.robot.dof
    # env = DAPGWrapper(base_env)
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
        # action[2] = 0.1
        obs, reward, done, _ = env.step(action)
        base_env.render()

    viewer.toggle_pause(True)
    while not viewer.closed:
        base_env.simple_step()
        base_env.render()


if __name__ == '__main__':
    main_env()
