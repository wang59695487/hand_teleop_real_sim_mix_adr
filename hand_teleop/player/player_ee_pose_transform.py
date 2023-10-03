import shutil
from typing import Dict, Any, Optional, List

import numpy as np
import sapien.core as sapien
import transforms3d
import pickle
from pathlib import Path
import os
import imageio

from hand_teleop.env.rl_env.base import BaseRLEnv, compute_inverse_kinematics
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.utils.common_robot_utils import LPFilter
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.real_world import lab


class DataPlayer:
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: BaseRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        self.meta_data = meta_data
        self.data = data
        self.scene = env.scene
        self.env = env

        # Human robot hand
        if zero_joint_pos is not None:
            if 'hand_mode' in meta_data.keys():
                hand_mode = meta_data['hand_mode']
            else:
                hand_mode = "right_hand"
            self.human_robot_hand = MANORobotHand(env.scene, env.renderer, init_joint_pos=zero_joint_pos,
                                                  hand_mode=hand_mode, control_interval=env.frame_skip * env.scene.get_timestep())
        else:
            self.human_robot_hand = None

        # Generate actor id mapping
        scene_actor2id = {actor.get_name(): actor.get_id() for actor in self.scene.get_all_actors()}
        meta_actor2id = self.meta_data["actor"]
        meta2scene_actor = {}
        for key, value in meta_actor2id.items():
            if key not in scene_actor2id:
                print(f"Demonstration actor {key} not exists in the scene. Will skip it.")
            else:
                meta2scene_actor[value] = scene_actor2id[key]

        # Generate articulation id mapping
        all_articulation_root = [robot.get_links()[0] for robot in self.scene.get_all_articulations()]
        scene_articulation2id = {actor.get_name(): actor.get_id() for actor in all_articulation_root}
        scene_articulation2dof = {r.get_links()[0].get_name(): r.dof for r in self.scene.get_all_articulations()}
        meta_articulation2id = self.meta_data["articulation"]
        meta_articulation2dof = self.meta_data["articulation_dof"]
        meta2scene_articulation = {}
        for key, value in meta_articulation2id.items():
            if key not in scene_articulation2id:
                print(f"Recorded articulation {key} not exists in the scene. Will skip it.")
            else:
                if meta_articulation2dof[key] == scene_articulation2dof[key]:
                    meta2scene_articulation[value] = scene_articulation2id[key]
                else:
                    print(
                        f"Recorded articulation {key} has {meta_articulation2dof[key]} dof while "
                        f"scene articulation has {scene_articulation2dof[key]}. Will skip it.")

        self.meta2scene_actor = meta2scene_actor
        self.meta2scene_articulation = meta2scene_articulation

        self.action_filter = LPFilter(50, 5)

    def get_sim_data(self, item) -> Dict[str, Any]:
        sim_data = self.data[item]["simulation"]
        actor_data = sim_data["actor"]
        drive_data = sim_data["articulation_drive"]
        articulation_data = sim_data["articulation"]
        scene_actor_data = {self.meta2scene_actor[key]: value for key, value in actor_data.items() if
                            key in self.meta2scene_actor}
        scene_drive_data = {self.meta2scene_articulation[key]: value for key, value in drive_data.items() if
                            key in self.meta2scene_articulation}
        scene_articulation_data = {self.meta2scene_articulation[key]: value for key, value in articulation_data.items()
                                   if key in self.meta2scene_articulation}
        return dict(actor=scene_actor_data, articulation_drive=scene_drive_data, articulation=scene_articulation_data)

    @staticmethod
    def collect_env_state(actors: List[sapien.Actor], articulations: List[sapien.Articulation] = []):
        data = dict(actor={}, articulation={})
        for actor in actors:
            v = actor.get_velocity()
            w = actor.get_angular_velocity()
            pose = actor.get_pose()
            actor_data = dict(velocity=v, angular_velocity=w, pose=np.concatenate([pose.p, pose.q]))
            data["actor"][actor.get_id()] = actor_data

        for articulation in articulations:
            links = articulation.get_links()
            pose = links[0].get_pose()
            qpos = articulation.get_qpos()
            qvel = articulation.get_qvel()
            articulation_data = dict(qvel=qvel, qpos=qpos, pose=np.concatenate([pose.p, pose.q]))
            data["articulation"][links[0].get_id()] = articulation_data
        return data

    def get_finger_tip_retargeting_result(self, human_robot_hand: MANORobotHand, retargeting: PositionRetargeting,
                                          indices,
                                          use_root_local_pose=True):
        assert human_robot_hand.free_root
        fix_global = len(retargeting.optimizer.fixed_joint_indices) == 6
        assert fix_global or len(retargeting.optimizer.fixed_joint_indices) == 0

        links = human_robot_hand.finger_tips
        if indices is not None:
            links = [links[i] for i in indices]
        if not fix_global:
            base = human_robot_hand.palm
            links = [base] + links
            fixed_qpos = np.array([])
        else:
            fixed_qpos = human_robot_hand.robot.get_qpos()[:6]

        if use_root_local_pose:
            base_pose_inv = human_robot_hand.robot.get_links()[0].get_pose().inv()
            human_hand_joints = np.stack([(base_pose_inv * link.get_pose()).p for link in links])
        else:
            base_pose_inv = self.env.robot.get_links()[0].get_pose().inv()
            human_hand_joints = np.stack([(base_pose_inv * link.get_pose()).p for link in links])
        robot_qpos = retargeting.retarget(human_hand_joints, fixed_qpos=fixed_qpos)
        return robot_qpos

    def get_finger_tip_middle_retargeting_result(self, human_robot_hand: MANORobotHand,
                                                 retargeting: PositionRetargeting,
                                                 indices, use_root_local_pose=True):
        assert human_robot_hand.free_root
        fix_global = len(retargeting.optimizer.fixed_joint_indices) == 6
        assert fix_global or len(retargeting.optimizer.fixed_joint_indices) == 0

        links = human_robot_hand.finger_tips + human_robot_hand.finger_middles
        if indices is not None:
            links = [links[i] for i in indices]
        if not fix_global:
            base = human_robot_hand.palm
            links = [base] + links
            fixed_qpos = np.array([])
        else:
            fixed_qpos = human_robot_hand.robot.get_qpos()[:6]

        if use_root_local_pose:
            base_pose_inv = human_robot_hand.robot.get_links()[0].get_pose().inv()
            human_hand_joints = np.stack([(base_pose_inv * link.get_pose()).p for link in links])
        else:
            base_pose_inv = self.env.robot.get_links()[0].get_pose().inv()
            human_hand_joints = np.stack([(base_pose_inv * link.get_pose()).p for link in links])

        if np.allclose(retargeting.last_qpos, np.zeros(retargeting.optimizer.dof)):
            retargeting.last_qpos[:6] = human_robot_hand.robot.get_qpos()[:6]
        robot_qpos = retargeting.retarget(human_hand_joints, fixed_qpos=fixed_qpos)
        return robot_qpos

    def compute_action_from_states(self, robot_qpos_prev, robot_qpos, is_contact: bool):
        v_limit = self.env.velocity_limit[:6, :]
        alpha = 1
        duration = self.env.scene.get_timestep() * self.env.frame_skip
        if not self.env.is_robot_free:
            arm_dof = self.env.arm_dof
            end_link = self.env.kinematic_model.partial_robot.get_links()[-1]
            self.env.kinematic_model.partial_robot.set_qpos(robot_qpos_prev[:arm_dof])
            prev_link_pose = end_link.get_pose()
            self.env.kinematic_model.partial_robot.set_qpos(robot_qpos[:arm_dof])
            current_link_pose = end_link.get_pose()
            delta_pose_spatial = current_link_pose * prev_link_pose.inv()
            axis, angle = transforms3d.quaternions.quat2axangle(delta_pose_spatial.q)
            target_velocity = np.concatenate([delta_pose_spatial.p, axis * angle]) / duration
        else:
            delta_qpos = robot_qpos[:6] - robot_qpos_prev[:6]
            target_velocity = delta_qpos / duration
        target_velocity = np.clip((target_velocity - v_limit[:, 0]) / (v_limit[:, 1] - v_limit[:, 0]) * 2 - 1, -1, 1)
        if not self.action_filter.is_init:
            self.action_filter.init(target_velocity)
        filtered_velocity = self.action_filter.next(target_velocity)
        filtered_velocity[:6] = filtered_velocity[:6] * alpha
        final_velocity = np.clip(filtered_velocity, -1, 1)

        # pos_gain = 2.2 if is_contact else 2
        pos_gain = 2
        limit = self.env.robot.get_qlimits()[6:]
        target_position = robot_qpos[6:]
        target_position = np.clip((target_position - limit[:, 0]) / (limit[:, 1] - limit[:, 0]) * pos_gain - 1, -1, 1)
        action = np.concatenate([final_velocity, target_position])
        return action


class RelocateObjectEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: RelocateRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[])
        manipulated_object = self.env.manipulated_object
        use_local_pose = False

        # Set target as pose
        self.scene.unpack(self.get_sim_data(self.meta_data["data_len"] - 1))
        target_pose = manipulated_object.get_pose()
        self.env.target_object.set_pose(target_pose)
        self.env.target_pose = target_pose
        baked_data["target_pose"] = np.concatenate([target_pose.p, target_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([manipulated_object])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                qpos = self.env.robot.get_qpos()
            baked_data["robot_qpos"].append(qpos)
            self.env.robot.set_qpos(qpos)
            if i >= 1:
                baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                            np.sum(contact_finger_index) > 0))
            if i >= 2:
                duration = self.env.frame_skip * self.scene.get_timestep()
                finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                root_qvel = baked_data["action"][-1][:6]
                self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())

            # Environment state
            baked_data["state"].append(self.collect_env_state([manipulated_object]))

        baked_data["action"].append(baked_data["action"][-1])
        return baked_data


class FlipMugEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: MugFlipRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        print(use_human_hand)
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        manipulated_object = self.env.manipulated_object
        use_local_pose = False

        # Set initial pose
        self.scene.unpack(self.get_sim_data(0))
        manipulated_object_pose = manipulated_object.get_pose()
        baked_data["init_mug_pose"] = np.concatenate([manipulated_object_pose.p, manipulated_object_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))
            contact_finger_index = self.human_robot_hand.check_contact_finger([manipulated_object_pose])

            # Robot qpos
            if use_human_hand:
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())

            # Environment state
            baked_data["state"].append(self.collect_env_state([manipulated_object]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


class TableDoorEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: TableDoorRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        table_door = self.env.table_door
        use_local_pose = False

        # Set initial pose
        self.scene.unpack(self.get_sim_data(0))
        table_door_pose = table_door.get_pose()
        baked_data["init_door_pose"] = np.concatenate([table_door_pose.p, table_door_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([table_door])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())

            # Environment state
            baked_data["state"].append(self.collect_env_state(actors=[], articulations=[table_door]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


class LaptopEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: LaptopRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        laptop = self.env.laptop
        use_local_pose = False

        # Set initial pose
        self.scene.unpack(self.get_sim_data(0))
        baked_data["init_laptop_qpos"] = laptop.get_qpos()
        baked_data["init_laptop_qvel"] = laptop.get_qvel()
        self.scene.step()
        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([laptop])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())
            # Environment state
            baked_data["state"].append(self.collect_env_state(actors=[], articulations=[laptop]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


class PickPlaceEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: PickPlaceRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        manipulated_object = self.env.manipulated_object
        use_local_pose = False

        # Set target as pose
        self.scene.unpack(self.get_sim_data(0))
        init_pose = manipulated_object.get_pose()
        baked_data["init_pose"] = np.concatenate([init_pose.p, init_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([manipulated_object])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())
            # Environment state
            baked_data["state"].append(self.collect_env_state([manipulated_object]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


class PenDrawEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: PenDrawRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        pen = self.env.pen
        use_local_pose = False

        # Set initial pose
        self.scene.unpack(self.get_sim_data(0))
        pen_pose = pen.get_pose()
        baked_data["init_pen_pose"] = np.concatenate([pen_pose.p, pen_pose.q])

        for i in range(self.meta_data["data_len"]):
            self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([pen])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                    # qvel = self.data[i]["qvel"]
                    # qpos = self.data[i]["robot_qpos"]
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
            baked_data["robot_qpos"].append(qpos)
            baked_data["robot_qvel"].append(qvel)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # action i is the transition from state i to state i
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.data[i]["action"])
                    # baked_data["action"].append(self.env.robot.get_drive_target())
                    # self.env.robot.set_qvel(qvel)
            # Environment observation
            baked_data["obs"].append(self.env.get_observation())
            # Environment state
            baked_data["state"].append(self.collect_env_state(actors=[], articulations=[pen]))

        if self.env.robot_name != "mano":
            baked_data["action"].append(baked_data["action"][-1])
            
        return baked_data


class HammerEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: HammerRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        hammer = self.env.hammer
        use_local_pose = False

        # Set target as pose
        self.scene.unpack(self.get_sim_data(0))
        init_pose = hammer.get_pose()
        baked_data["init_pose"] = np.concatenate([init_pose.p, init_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([hammer])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())
            # Environment state
            baked_data["state"].append(self.collect_env_state([hammer]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


class InsertObjectEnvPlayer(DataPlayer):
    def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: InsertObjectRLEnv,
                 zero_joint_pos: Optional[np.ndarray] = None):
        super().__init__(meta_data, data, env, zero_joint_pos)

    def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
        use_human_hand = self.human_robot_hand is not None and retargeting is not None
        baked_data = dict(obs=[], robot_qpos=[], state=[], action=[], robot_qvel=[], ee_pose=[])
        manipulated_object = self.env.manipulated_object
        use_local_pose = False

        # Set target as pose
        self.scene.unpack(self.get_sim_data(0))
        init_pose = manipulated_object.get_pose()
        baked_data["init_pose"] = np.concatenate([init_pose.p, init_pose.q])
        self.scene.step()

        for i in range(self.meta_data["data_len"]):
            for _ in range(self.env.frame_skip):
                self.scene.step()
            self.scene.unpack(self.get_sim_data(i))

            # Robot qpos
            if use_human_hand:
                contact_finger_index = self.human_robot_hand.check_contact_finger([manipulated_object])
                if method == "tip_middle":
                    qpos = self.get_finger_tip_middle_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                         use_root_local_pose=use_local_pose)
                elif method == "tip":
                    qpos = self.get_finger_tip_retargeting_result(self.human_robot_hand, retargeting, indices,
                                                                  use_root_local_pose=use_local_pose)
                else:
                    raise ValueError(f"Retargeting method {method} is not supported")
            else:
                if self.env.robot_name == "mano":
                    qpos = self.human_robot_hand.robot.get_qpos()
                    qvel = self.human_robot_hand.robot.get_qvel()
                else:
                    qpos = self.env.robot.get_qpos()
                    qvel = self.env.robot.get_qvel()
                baked_data["robot_qvel"].append(qvel)
            baked_data["robot_qpos"].append(qpos)
            ee_pose = self.env.ee_link.get_pose()
            baked_data["ee_pose"].append(np.concatenate([ee_pose.p, ee_pose.q]))
            self.env.robot.set_qpos(qpos)
            if self.env.robot_name == "mano":
                # NOTE: Action i is the transition from state i to state i+1
                baked_data["action"].append(self.human_robot_hand.robot.get_drive_target())
                self.env.robot.set_qvel(qvel)
            else:
                if use_human_hand:
                    if i >= 1:
                        baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
                                                                                    np.sum(contact_finger_index) > 0))
                    if i >= 2:
                        duration = self.env.frame_skip * self.scene.get_timestep()
                        finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
                        root_qvel = baked_data["action"][-1][:6]
                        self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))
                else:
                    baked_data["action"].append(self.env.robot.get_drive_target())
                    self.env.robot.set_qvel(qvel)

            # Environment observation
            baked_data["obs"].append(self.env.get_observation())
            # Environment state
            baked_data["state"].append(self.collect_env_state([manipulated_object]))

        if use_human_hand:
            baked_data["action"].append(baked_data["action"][-1])

        return baked_data


def bake_demonstration_allegro_test(retarget=False):
    from pathlib import Path

    # Recorder
    # robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    # path = Path("/home/sim/data/teleop/relocate-mustard_bottle/0021.pickle")
    path = "./sim/raw_data/xarm/less_random/pick_place_single_demo/mustard_bottle_0050.pickle"
    # all_data = np.load(str(path.resolve()), allow_pickle=True)
    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    data = all_data["data"]
    use_visual_obs = False
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    if 'finger_control_params' in meta_data.keys():
        finger_control_params = meta_data['finger_control_params']
    if 'root_rotation_control_params' in meta_data.keys():
        root_rotation_control_params = meta_data['root_rotation_control_params']
    if 'root_translation_control_params' in meta_data.keys():
        root_translation_control_params = meta_data['root_translation_control_params']
    if 'robot_arm_control_params' in meta_data.keys():
        robot_arm_control_params = meta_data['robot_arm_control_params']        

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = True
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
    
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'hammer':
        env = HammerRLEnv(**env_params)
    elif task_name == 'table_door':
        env = TableDoorRLEnv(**env_params)
    elif task_name == 'insert_object':
        env = InsertObjectRLEnv(**env_params)
    elif task_name == 'mug_flip':
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError

    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                    joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                else:
                    joint.set_drive_property(*(finger_control_params), mode="acceleration")
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                else:
                    joint.set_drive_property(*(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step
    env.reset()
    viewer = env.render(mode="human")
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, np.pi/4)
    if task_name == 'table_door':
        viewer.set_camera_xyz(-0.6, -0.6, 0.6)
        viewer.set_camera_rpy(0, -np.pi/6, -np.pi/4)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    # Retargeting
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration()

    # Debug
    a = []
    a1 = []

    # Visualization
    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)
    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])
    # actor_id = env.manipulated_object.get_id()
    # target_pose_vec = baked_data["target_pose"]
    # env.target_object.set_pose(sapien.Pose(target_pose_vec[:3], target_pose_vec[3:]))
    # articulation_id = env.manipulated_object.get_links()[0].get_id()
    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))
    delta_catesian_action = []
    for idx, (obs, qpos, state, action, ee_pose) in enumerate(zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"], baked_data["ee_pose"])):
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data['obs'])-1:
            palm_pose = env.ee_link.get_pose()
            palm_pose = robot_pose.inv() * palm_pose

            ee_pose_next = baked_data["ee_pose"][idx + 1]
            palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
            palm_next_pose = robot_pose.inv() * palm_next_pose

            palm_delta_pose = palm_pose.inv() * palm_next_pose
            delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
                delta_axis = -delta_axis
            delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
            delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

            # euler = euler_optim.optimize(transforms3d.quaternions.quat2mat(palm_next_pose.q))
            # palm_next_pose = np.concatenate((palm_next_pose.p, euler))
            # delta_pose = palm_next_pose - palm_pose
            palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
            arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
            arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]
            hand_qpos = action[env.arm_dof:]
            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            env.step(target_qpos)
            env.render()

        # # NOTE: Old Method
        # env.step(action)
        # ##
        # # env.robot.set_qpos(qpos)
        # # print(action[:6])
        # # print(env.reward())
        # # object_pose = state["actor"][actor_id]["pose"]
        # # object_pose = state["articulation"][articulation_id]["pose"]
        # # env.manipulated_object.set_pose(sapien.Pose([0,0,-0.3], [1,0,0,0]))
        # # env.manipulated_object.set_pose(sapien.Pose(object_pose[:3], object_pose[3:7]))
        # ##

        # env.render()
        # # for _ in range(3):
        # #     env.render()

def bake_demonstration_panda_test():
    from pathlib import Path

    # Recorder
    robot_name = "allegro_hand_xarm6_wrist_mounted_rotate"
    path = Path("/home/sim/data/teleop/relocate-mustard_bottle/0021.pickle")
    all_data = np.load(str(path.resolve()), allow_pickle=True)
    meta_data = all_data["meta_data"]
    data = all_data["data"]
    env = RelocateRLEnv(**meta_data["env_kwargs"], robot_name=robot_name, use_gui=True)
    player = RelocateObjectEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # env.robot.set_pose(sapien.Pose([0, 0, -2]))

    # Retargeting
    link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                  "link_2.0", "link_6.0", "link_10.0"]
    indices = [0, 1, 2, 3, 5, 6, 7, 8]
    joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
    retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                      has_joint_limits=True)
    baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)

    # Visualization
    player.scene.remove_articulation(player.human_robot_hand.robot)
    actor_id = env.manipulated_object.get_id()
    target_pose_vec = baked_data["target_pose"]
    env.target_object.set_pose(sapien.Pose(target_pose_vec[:3], target_pose_vec[3:]))
    for obs, qpos, state, action in zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"]):
        env.robot.set_qpos(obs[:22])
        # print(action[:6])
        # print(env.reward())
        object_pose = state["actor"][actor_id]["pose"]
        env.manipulated_object.set_pose(sapien.Pose(object_pose[:3], object_pose[3:7]))

        for _ in range(3):
            env.render()


def bake_demonstration_mano():
    from pathlib import Path

    # Recorder
    robot_name = "mano"
    shutil.rmtree('./temp/demos/single')
    os.makedirs('./temp/demos/single')
    path = "./sim/raw_data/free_hand/less_random/mano_pick_place_mustard_bottle/mustard_bottle_0001.pickle"
    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    data = all_data["data"]
    use_visual_obs = False

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = True
    # env_params = dict(object_name=meta_data["env_kwargs"]['object_name'], object_scale=meta_data["env_kwargs"]['object_scale'], robot_name=robot_name, 
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=meta_data["env_kwargs"]['randomness_scale'], 
    #                  use_visual_obs=use_visual_obs, use_gui=False)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'hammer':
        env = HammerRLEnv(**env_params)
    elif task_name == 'table_door':
        env = TableDoorRLEnv(**env_params)
    elif task_name == 'insert_object':
        env = InsertObjectRLEnv(**env_params)
    elif task_name == 'mug_flip':
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError
    env.reset()

    viewer = env.render(mode="human")
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, 0)    
    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError
    
    baked_data = player.bake_demonstration()

    # Visualization
    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    for _ in range(player.env.frame_skip):
        player.scene.step()
    player.scene.remove_articulation(player.human_robot_hand.robot)
    # target_pose_vec = baked_data["target_pose"]
    # env.target_object.set_pose(sapien.Pose(target_pose_vec[:3], target_pose_vec[3:]))
    # articulation_id = env.manipulated_object.get_links()[0].get_id()
    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    for obs, qpos, state, action in zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"]):

        env.step(action)

        ##
        # print(action[:6])
        # print(env.reward())
        # env.robot.set_qpos(obs[:51])
        # object_pose = state["actor"][actor_id]["pose"]
        # env.manipulated_object.set_pose(sapien.Pose(object_pose[:3], object_pose[3:7]))
        # object_qpos = state["articulation"][articulation_id]["qpos"]
        # object_qvel = state["articulation"][articulation_id]["qvel"]
        # env.manipulated_object.set_qpos(object_qpos)
        # env.manipulated_object.set_qvel(object_qvel)
        # prev_obs = obs
        ##

        for _ in range(3):
            env.render()
    
def average_angle_handqpos(hand_qpos):
    delta_angles = []
    for i in range(0,len(hand_qpos),4):
        qpos = hand_qpos[i:i+4]
        delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(qpos)
        if delta_angle > np.pi:
            delta_angle = 2 * np.pi - delta_angle
        delta_angles.append(delta_angle)
    return np.mean(delta_angles)

def bake_visual_demonstration_test(retarget=False):
    from pathlib import Path

    # Recorder
    shutil.rmtree('./temp/demos/player', ignore_errors=True)
    os.makedirs('./temp/demos/player')
    path = "./sim/raw_data/xarm/less_random/pick_place_mustard_bottle/mustard_bottle_0001.pickle"
    #path = "sim/raw_data/xarm/less_random/pick_place_tomato_soup_can/tomato_soup_can_0001.pickle"
    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    data = all_data["data"]
    use_visual_obs = True
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_free"
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params']       

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = True
    # env_params = dict(object_name=meta_data["env_kwargs"]['object_name'], object_scale=meta_data["env_kwargs"]['object_scale'], robot_name=robot_name, 
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=meta_data["env_kwargs"]['randomness_scale'], 
    #                  use_visual_obs=use_visual_obs, use_gui=False)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'hammer':
        env = HammerRLEnv(**env_params)
    elif task_name == 'table_door':
        env = TableDoorRLEnv(**env_params)
    elif task_name == 'insert_object':
        env = InsertObjectRLEnv(**env_params)
    elif task_name == 'mug_flip':
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError

    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                    joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                else:
                    joint.set_drive_property(*(finger_control_params), mode="acceleration")
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                else:
                    joint.set_drive_property(*(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step
    env.reset()
    viewer = env.render(mode="human")
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, np.pi/4)  

    
    real_camera_cfg = {
        "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
    }
    
    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }   
         
    env.setup_camera_from_config(real_camera_cfg)
    #env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    # Retargeting
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration()
    visual_baked = dict(obs=[], action=[])
    env.reset()
    player.scene.unpack(player.get_sim_data(0))
    # env.randomize_object_rotation()
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))
    delta_catesian_action = []

    ee_pose = baked_data["ee_pose"][0]
    #hand_qpos_prev = baked_data["action"][0][env.arm_dof:]
    for idx, (obs, qpos, state, action, ee_pose) in enumerate(zip(baked_data["obs"], baked_data["robot_qpos"], baked_data["state"],
                                        baked_data["action"], baked_data["ee_pose"])):
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data['obs'])-1:
            ee_pose_next = baked_data["ee_pose"][idx + 1]
            ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            
            #delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos
            #if ee_pose_delta > 0.0005 and average_angle_handqpos(delta_hand_qpos)>=np.pi/180 :
            if ee_pose_delta > 0.0003:
                ee_pose = ee_pose_next
                #hand_qpos_prev = hand_qpos

                palm_pose = env.ee_link.get_pose()
                palm_pose = robot_pose.inv() * palm_pose

                palm_next_pose = sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                palm_next_pose = robot_pose.inv() * palm_next_pose

                palm_delta_pose = palm_pose.inv() * palm_next_pose
                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(env.robot.get_qpos()[:env.arm_dof])
                arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
                arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]
                hand_qpos = action[env.arm_dof:]
                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                visual_baked["obs"].append(env.get_observation())
                visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                _, _, _, info = env.step(target_qpos)
                env.render()
            
            else:
                print("!!!!!!!!!!!!!!!!!!!!!!skip!!!!!!!!!!!!!!!!!!!!!")

        # # NOTE: Old Version
        # visual_baked["obs"].append(env.get_observation())
        # visual_baked["action"].append(action)
        # env.step(action)

        # env.render()
        # # for _ in range(3):
        # #     env.render()

    for i in range(len(visual_baked["obs"])):
        rgb = visual_baked["obs"][i]["relocate_view-rgb"]
        rgb_pic = (rgb * 255).astype(np.uint8)
        imageio.imsave("./temp/demos/player/relocate-rgb_{}.png".format(i), rgb_pic)


def bake_visual_real_demonstration_test(retarget=False, index = 0):
    from pathlib import Path

    # Recorder
    shutil.rmtree('./temp/demos/player', ignore_errors=True)
    os.makedirs('./temp/demos/player')
    #path = "./sim/raw_data/xarm/less_random/pick_place_mustard_bottle/mustard_bottle_0002.pickle"
    #path = "./sim/raw_data/xarm/less_random/pick_place_sugar_box/sugar_box_0001.pickle"
    path = "sim/raw_data/xarm/less_random/pick_place_tomato_soup_can/tomato_soup_can_0001.pickle"
    
    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    #meta_data['env_kwargs']['init_target_pos'] = sapien.Pose([-0.05, -0.105, 0], [1, 0, 0, 0])
    data = all_data["data"]
    use_visual_obs = True

    #print(meta_data)
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_free"
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params'] 
             

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = use_visual_obs
    env_params['use_gui'] = True
    # env_params = dict(object_name=meta_data["env_kwargs"]['object_name'], object_scale=meta_data["env_kwargs"]['object_scale'], robot_name=robot_name, 
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=meta_data["env_kwargs"]['randomness_scale'], 
    #                  use_visual_obs=use_visual_obs, use_gui=False)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
        print(env_params['init_target_pos'])
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'hammer':
        env = HammerRLEnv(**env_params)
    elif task_name == 'table_door':
        env = TableDoorRLEnv(**env_params)
    elif task_name == 'insert_object':
        env = InsertObjectRLEnv(**env_params)
    elif task_name == 'mug_flip':
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError
    
    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(*(1 * root_translation_control_params), mode="acceleration")
                elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                    joint.set_drive_property(*(1 * root_rotation_control_params), mode="acceleration")
                else:
                    joint.set_drive_property(*(finger_control_params), mode="acceleration")
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
                else:
                    joint.set_drive_property(*(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step
    env.reset()
    viewer = env.render(mode="human")
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, np.pi/4)  

    # real_camera_cfg = {
    #     "relocate_view": dict( pose= ROBOT2BASE * CAM2ROBOT, fov=np.deg2rad(47.4), resolution=(320, 240))
    # }
    
    real_camera_cfg = {
        "relocate_view": dict( pose= lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
    }

    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }
            
    env.setup_camera_from_config(real_camera_cfg)
    #env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'hammer':
        player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'table_door':
        player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'insert_object':
        player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'mug_flip':
        player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    # Retargeting
    using_real = True
    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                    "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name() for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                        has_joint_limits=True)
        baked_data = player.bake_demonstration(retargeting, method="tip_middle", indices=indices)
    elif using_real:
        #record_root  = "./real/raw_data/pick_place_mustard_bottle"
        record_root  = "./real/raw_data/pick_place_tomato_soup_can"
        #record_root  = "./real/raw_data/pick_place_sugar_box"
        record_path = Path(record_root) / f"{index:04}.pkl"
        #path = "./real/raw_data/pick_place_mustard_bottle/0001.pkl"
        baked_data = np.load(record_path, allow_pickle=True)
        
    visual_baked = dict(obs=[], action=[])
    env.reset()

    player.scene.unpack(player.get_sim_data(0))
    # env.randomize_object_rotation()
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)
    
    #robot_base_pose = np.array([0, -0.7, 0, 0.707, 0, 0, 0.707])
    env.robot.set_qpos(baked_data[0]["teleop_cmd"])
    #print("init_qpos: ",baked_data[0]["teleop_cmd"])

    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(-np.matmul(rotation_matrix.T,robot_pose.p),rotation_matrix.T,np.ones(3))
    
    for idx in range(len(baked_data)):

        ee_pose = env.ee_link.get_pose()
        baked_data[idx]["ee_pose"] = np.concatenate([ee_pose.p,ee_pose.q])
        #print(baked_data[idx]["ee_pose"])
        # NOTE: robot.get_qpos() version
        if idx != len(baked_data)-1:
            
            hand_qpos = baked_data[idx+1]["teleop_cmd"][env.arm_dof:]
            arm_qpos = baked_data[idx+1]["teleop_cmd"][:env.arm_dof]

            target_qpos = np.concatenate([arm_qpos, hand_qpos])
            # env.step(target_qpos)
            env.robot.set_qpos(target_qpos)
            env.render()
    
    with record_path.open("wb") as f:
        pickle.dump(baked_data, f)

    # for i in range(len(visual_baked["obs"])):
    #     rgb = visual_baked["obs"][i]["relocate_view-rgb"]
    #     rgb_pic = (rgb * 255).astype(np.uint8)
    #     imageio.imsave("./temp/demos/player/relocate-rgb_{}.png".format(i), rgb_pic)
    
    

if __name__ == '__main__':
    # bake_demonstration_adroit()
    # bake_demonstration_allegro_test()
    # bake_demonstration_svh_test()
    # bake_demonstration_ar10_test()
    # bake_demonstration_mano()
    # bake_visual_demonstration_test()
    for i in range(15):
        bake_visual_real_demonstration_test(index = i)
