from functools import cached_property

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.drawer_env import DrawerEnv
from hand_teleop.utils.common_robot_utils import generate_free_robot_hand_info, generate_arm_robot_hand_info


class DrawerRLEnv(DrawerEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", rotation_reward_weight=0, 
                object_pose_noise=0.01, constant_object_state=False, randomness_scale=1, friction=1, 
                object_name="drawer_1", **renderer_kwargs):
        super().__init__(use_gui, frame_skip, friction=friction, object_name=object_name, **renderer_kwargs)
        self.setup(robot_name)
        self.rotation_reward_weight = rotation_reward_weight
        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        else:
            info = generate_arm_robot_hand_info()[robot_name]
        self.palm_link_name = info.palm_name
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == self.palm_link_name][0]
        self.drawer_link = [link for link in self.drawer.get_links() if link.get_name() == "link1"][0]
        self.is_contact = False
        self.is_unlock = False

    def get_state_observation(self):
        robot_qpos_vec = self.robot.get_qpos()
        door_qpos_vec = self.drawer.get_qpos()
        handle_pose = self.drawer_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        handle_in_palm = handle_pose.p - palm_pose.p
        palm_v = self.palm_link.get_velocity()
        palm_w = self.palm_link.get_angular_velocity()
        self.is_contact = self.check_contact(self.robot.get_links(), [self.drawer_link])
        return np.concatenate(
            [robot_qpos_vec, door_qpos_vec, palm_v, palm_w, handle_in_palm,
             [int(self.is_contact)]])

    def get_proprioception(self): # TODO: helin: do not understand.
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate(
            [robot_qpos_vec, palm_pose.p])

    def reward(self):
        drawer_qpos = self.drawer.get_qpos()[0]
        handle_pose = self.drawer_link.get_pose()
        palm_pose = self.palm_link.get_pose()
        is_contact = self.is_contact

        reward = (0.2 - drawer_qpos)
        reward += -0.1 * min(np.linalg.norm(palm_pose.p - handle_pose.p), 0.5)
        if is_contact:
            reward += 0.1

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
    from hand_teleop.dapg.dapg_wrapper import DAPGWrapper
    from time import time
    base_env = DrawerRLEnv(use_gui=True, robot_name="allegro_hand_free")
    robot_dof = base_env.robot.dof
    env = DAPGWrapper(base_env)
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
