import shutil
from typing import Dict, Any, Optional, List

import numpy as np
import sapien.core as sapien
import transforms3d
import pickle
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
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
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

        self.scene.unpack(self.get_sim_data(0))

        init_pose = manipulated_object.get_pose()
        baked_data["init_pose"] = np.concatenate([init_pose.p, init_pose.q])
        # self.scene.step()

        for i in range(self.meta_data["data_len"]):
            # for _ in range(self.env.frame_skip):
            #     self.scene.step()
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
                        baked_data["action"].append(
                            self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
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

class DcLawEnvPlayer(DataPlayer):
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
            #baked_data["state"].append(self.collect_env_state([manipulated_object]))

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
