from functools import cached_property
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.sim_env.constructor import (
    add_default_scene_light,
    random_scene_light,
    random_environment_map,
)
from hand_teleop.env.sim_env.pour_env import PourBoxEnv
from hand_teleop.real_world import lab
from hand_teleop.utils.common_robot_utils import (
    generate_free_robot_hand_info,
    generate_arm_robot_hand_info,
)


class PourBoxRLEnv(PourBoxEnv, BaseRLEnv):
    def __init__(
        self,
        use_gui=False,
        frame_skip=5,
        robot_name="adroit_hand_free",
        constant_object_state=False,
        object_name="chip_can",
        object_seed=0,
        randomness_scale=1,
        friction=1,
        zero_joint_pos=None,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            object_name=object_name,
            randomness_scale=randomness_scale,
            friction=friction,
            object_seed=object_seed,
            **renderer_kwargs,
        )

        self.setup(robot_name)
        self.randomness_scale = randomness_scale

        # Parse link name
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        else:
            info = generate_arm_robot_hand_info()[robot_name]
        self.palm_link_name = info.palm_name
        self.palm_link = [
            link
            for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name
        ][0]

        # Object init pose
        self.object_episode_init_pose = sapien.Pose()
        self.is_object_lifted = False
        add_default_scene_light(self.scene, self.renderer)

    def random_map(self, randomness_scale=1):
        random_environment_map(self.scene, randomness_scale)

    def random_light(self, randomness_scale=1):
        print(
            f"###############################Random Scene Light####################################"
        )
        random_scene_light(self.scene, self.renderer, randomness_scale)

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        object_pose = self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([object_pose.p, object_pose.q])
        palm_pose = self.palm_link.get_pose()
        return np.concatenate(
            [object_pose_vec, robot_qpos_vec, palm_pose.p, palm_pose.q]
        )

    def get_robot_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        ee_pose = self.ee_link.get_pose()
        jacobian = self.kinematic_model.compute_end_link_spatial_jacobian(
            robot_qpos_vec[: self.arm_dof]
        ).flatten()
        return np.concatenate([jacobian, robot_qpos_vec, ee_pose.p, ee_pose.q])

    def get_reward(self, action):
        reward = 0
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed) helin
        if not self.is_robot_free:
            qpos = np.zeros(self.robot.dof)
            xarm_qpos = self.robot_info.arm_init_qpos
            qpos[: self.arm_dof] = xarm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array(lab.ROBOT2BASE.p) + self.robot_info.root_offset
            init_pose = sapien.Pose(
                init_pos, transforms3d.euler.euler2quat(0, 0, 0))
        else:
            init_pose = sapien.Pose(
                np.array([-0.3, 0, 0.2]
                         ), transforms3d.euler.euler2quat(0, np.pi / 2, 0)
            )
        self.robot.set_pose(init_pose)
        self.reset_internal()
        self.is_object_lifted = False
        self.boxes_not_in_bowl = [i for i in range(len(self.boxes))]
        return self.get_observation()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return self.robot.dof + 7 + 6 + 9 + 4 + 1
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return False

    @cached_property
    def horizon(self):
        return 250

    def _num_box_in_bowl(self):
        # contact_buffer = self.check_actor_pair_contacts(
        #     self.boxes, self.target_object)
        # return np.sum(contact_buffer)
        # check the contact between the box and the target
        tar_pos_x = self.target_object.pose.p[0]
        tar_pos_y = self.target_object.pose.p[1]
        for i in self.boxes_not_in_bowl:
            box = self.boxes[i]
            # correct the box pose
            box_world_pose = box.pose * \
                sapien.Pose([0, 0, 0.025 * i + 0.1], box.pose.q)
            is_bottle_contact = self.check_contact(
                [box], [self.manipulated_object])
            is_box_still = np.linalg.norm(box.velocity) <= 1e-6
            if (
                np.abs(box_world_pose.p[0] - tar_pos_x) < 0.0795
                and np.abs(box_world_pose.p[1] - tar_pos_y) < 0.0795
                and np.abs(box_world_pose.p[2]) > 0.0125
                and not is_bottle_contact
                and is_box_still
            ):
                self.boxes_not_in_bowl.remove(i)

        return len(self.boxes) - len(self.boxes_not_in_bowl)

    def _is_object_lifted(self):
        # check the x-y position of the object against the target
        object_z = self.manipulated_object.pose.p[-1]
        if object_z >= 0.02:
            self.is_object_lifted = True

        return self.is_object_lifted

    def _is_close_to_target(self):
        # check the x-y position of the object against the target
        object_xy = self.manipulated_object.pose.p[:-1]
        target_xy = self.target_object.pose.p[:-1]
        dist_xy = np.linalg.norm(object_xy - target_xy)
        close_to_target = dist_xy <= 0.2

        return close_to_target

    def _is_bottle_still(self):
        # check if the object is close to still
        velocity_norm = np.linalg.norm(self.manipulated_object.velocity)
        object_is_still = velocity_norm <= 1e-6

        return object_is_still

    def _is_success(self):
        return self._num_box_in_bowl() >= 2 and self._is_object_lifted()

    def get_info(self):
        return {
            "num_box_in_bowl": self._num_box_in_bowl(),
            "is_object_lifted": self._is_object_lifted(),
            "is_bottle_still": self._is_bottle_still(),
            "success": self._is_success(),
        }


def main_env():
    from time import time

    env = PourBoxRLEnv(
        use_gui=True,
        robot_name="allegro_hand_free",
        object_name="tomato_soup_can",
        object_category="YCB",
        frame_skip=5,
        use_visual_obs=False,
    )
    base_env = env
    robot_dof = env.robot.dof
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
    base_env.viewer = viewer

    env.reset()
    pose = env.palm_link.get_pose()
    for i in range(5000):
        action = np.zeros(robot_dof)
        action[0] = 0.1
        obs, reward, done, _ = env.step(action)
        env.render()
        if i == 200:
            pose_error = pose.inv() * env.palm_link.get_pose()
            print(pose_error)

    while not viewer.closed:
        env.render()


if __name__ == "__main__":
    main_env()
