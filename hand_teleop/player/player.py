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
from hand_teleop.env.rl_env.pour_env import PourBoxRLEnv
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.real_world import lab
from hand_teleop.player.data_players import *

dclaw_diverse_objects = {
    0: "dclaw_3x",
    1: "dclaw_3x_135",
    2: "dclaw_4x_60",
    3: "dclaw_4x_75",
    4: "dclaw_4x_90",
    5: "dclaw_5x_60",
    6: "dclaw_5x_72",
    7: "dclaw_5x_75"
}


def handqpos2angle(hand_qpos):
    delta_angles = []
    for i in range(0, len(hand_qpos)):
        delta_angle = hand_qpos[i]
        if delta_angle > np.pi:
            delta_angle = 2 * np.pi - delta_angle
        delta_angle = delta_angle / np.pi * 180
        delta_angles.append(np.abs(delta_angle))
    return delta_angles


def create_env_test(retarget=False, idx=2):
    # Recorder
    # shutil.rmtree('./temp/demos/player', ignore_errors=True)
    # os.makedirs('./temp/demos/player')
    # path = f"./sim/raw_data/pick_place_mustard_bottle/mustard_bottle_{idx:004d}.pickle"
    # path = f"./sim/raw_data/dclaw/dclaw_3x_{idx:004d}.pickle"
    # path = f"./sim/raw_data/pick_place/bottle_1_{idx:004d}.pickle"
    path = f"./sim/raw_data/pour/chip_can_{idx:004d}.pickle"
    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]["task_name"]
    meta_data["env_kwargs"].pop("task_name")
    data = all_data["data"]
    use_visual_obs = True
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_free"
    if "allegro" in robot_name:
        if "finger_control_params" in meta_data.keys():
            finger_control_params = meta_data["finger_control_params"]
        if "root_rotation_control_params" in meta_data.keys():
            root_rotation_control_params = meta_data["root_rotation_control_params"]
        if "root_translation_control_params" in meta_data.keys():
            root_translation_control_params = meta_data[
                "root_translation_control_params"
            ]
        if "robot_arm_control_params" in meta_data.keys():
            robot_arm_control_params = meta_data["robot_arm_control_params"]

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params["robot_name"] = robot_name
    env_params["use_visual_obs"] = use_visual_obs
    env_params["use_gui"] = False
    # env_params["object_name"] = "sugar_box"
    # env_params["object_name"] = "bleach_cleanser"
    # env_params["object_category"] = "SHAPE_NET"
    # env_params["object_name"] = "bottle_8"

    # env_params["object_name"] = "dclaw_3x_135"
    # env_params["object_name"] = "dclaw_4x_60"
    # env_params["object_name"] = "dclaw_4x_75"
    # env_params["object_name"] = "dclaw_4x_90"
    # env_params["object_name"] = "dclaw_5x_60"
    # env_params["object_name"] = "dclaw_5x_72"
    # env_params["object_name"] = "dclaw_5x_75"

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None
    if "init_obj_pos" in meta_data["env_kwargs"].keys():
        env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]
    if "init_target_pos" in meta_data["env_kwargs"].keys():
        env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]
    if task_name == "pick_place":
        env = PickPlaceRLEnv(**env_params)
    elif task_name == "dclaw":
        env = DClawRLEnv(**env_params)
    elif task_name == "pour":
        env = PourBoxRLEnv(**env_params)
    else:
        raise NotImplementedError

    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(
                        *(1 * root_translation_control_params), mode="acceleration"
                    )
                elif (
                    "x_rotation_joint" in name
                    or "y_rotation_joint" in name
                    or "z_rotation_joint" in name
                ):
                    joint.set_drive_property(
                        *(1 * root_rotation_control_params), mode="acceleration"
                    )
                else:
                    joint.set_drive_property(
                        *(finger_control_params), mode="acceleration"
                    )
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(
                        *(1 * robot_arm_control_params), mode="force"
                    )
                else:
                    joint.set_drive_property(
                        *(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step

    env.reset()
    # viewer = env.render()
    # env.viewer = viewer
    # viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    # viewer.set_camera_rpy(0, -np.pi / 6, np.pi / 4)

    real_camera_cfg = {
        "relocate_view": dict(
            pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(640, 480)
        )
    }

    env.setup_camera_from_config(real_camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {
        "rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)
    # Player
    if task_name == "pick_place":
        player = PickPlaceEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "dclaw":
        player = DcLawEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "pour":
        player = PourEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    else:
        raise NotImplementedError

    # Retargeting
    if retarget:
        link_names = [
            "palm_center",
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_14.0",
            "link_2.0",
            "link_6.0",
            "link_10.0",
        ]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name()
                       for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(
            env.robot,
            joint_names,
            link_names,
            has_global_pose_limits=False,
            has_joint_limits=True,
        )
        baked_data = player.bake_demonstration(
            retargeting, method="tip_middle", indices=indices
        )
    else:
        baked_data = player.bake_demonstration()

    env.reset()
    player.scene.unpack(player.get_sim_data(0))

    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    return env, task_name, meta_data, baked_data, path, idx


def bake_visual_demonstration_test(retarget=False, idx=2):
    env, task_name, meta_data, baked_data, path, demo_idx = create_env_test(
        retarget=retarget, idx=idx)

    robot_pose = env.robot.get_pose()
    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]

    frame_skip = 1
    rgb_pics = []

    ################################# Kinematic Augmentation####################################
    init_pose_aug_dict = {
        "init_pose_aug_obj": sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
        "init_pose_aug_target": sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
    }

    aug_step_target = 400
    init_pose_aug_obj = init_pose_aug_dict["init_pose_aug_obj"]
    meta_data["env_kwargs"]["init_obj_pos"] = (
        init_pose_aug_obj * meta_data["env_kwargs"]["init_obj_pos"]
    )
    env.manipulated_object.set_pose(meta_data["env_kwargs"]["init_obj_pos"])
    meta_data["env_kwargs"]["init_target_pos"] = sapien.Pose(
        [0.0, 0.2, env.bowl_height]
    )
    if task_name in ["pick_place", "pour"]:
        init_pose_aug_target = init_pose_aug_dict["init_pose_aug_target"]
        meta_data["env_kwargs"]["init_target_pos"] = (
            init_pose_aug_target * meta_data["env_kwargs"]["init_target_pos"]
        )
        env.target_object.set_pose(meta_data["env_kwargs"]["init_target_pos"])
        aug_target = np.array([init_pose_aug_obj.p[0], init_pose_aug_obj.p[1]])
        one_step_aug_target = np.array(
            [
                (-1 * init_pose_aug_obj.p[0] + init_pose_aug_target.p[0])
                / aug_step_target,
                (-1 * init_pose_aug_obj.p[1] + init_pose_aug_target.p[1])
                / aug_step_target,
            ]
        )
        aug_step_obj = 500
        aug_obj = np.array([0, 0])
        one_step_aug_obj = np.array(
            [
                init_pose_aug_obj.p[0] / aug_step_obj,
                init_pose_aug_obj.p[0] / aug_step_obj,
            ]
        )

    elif task_name == "dclaw":
        aug_step_obj = 50
        aug_obj = np.array([0, 0])
        one_step_aug_obj = np.array(
            [
                (init_pose_aug_obj.p[0]) / aug_step_obj,
                (init_pose_aug_obj.p[1]) / aug_step_obj,
            ]
        )

    # LIGHT AND TEXTURE RANDOMNESS
    env.random_map(2)  # hyper parameter
    ############## Add Texture Randomness ############
    env.generate_random_object_texture(2)
    ############## Add Light Randomness ############
    env.random_light(2)
    ############## Add Action Chunk ############
    valid_frame = 0
    lifted_chunk = 0
    visual_baked = dict(obs=[], action=[])
    for idx in range(0, len(baked_data["obs"]), frame_skip):
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data["obs"]) - frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(
                np.sum((ee_pose_next[:3] - ee_pose[:3]) ** 2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos

            if (
                ee_pose_delta < 0.001
                and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2
                and task_name in ["dclaw"]
            ):
                continue

            else:
                valid_frame += 1
                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos

                palm_pose = env.ee_link.get_pose()
                palm_pose = robot_pose.inv() * palm_pose

                if task_name in ["pick_place", "pour"]:
                    if env._is_object_lifted():
                        if lifted_chunk == 0:
                            lifted_chunk = int((valid_frame - 1) / 50)
                        if aug_step_target > 0:
                            aug_step_target -= 1
                            aug_target = aug_target + one_step_aug_target
                        palm_next_pose = sapien.Pose(
                            [aug_target[0], aug_target[1], 0], [1, 0, 0, 0]
                        ) * sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

                    elif not env._is_object_lifted():
                        if aug_step_obj > 0:
                            aug_step_obj -= 1
                            aug_obj = aug_obj + one_step_aug_obj
                        palm_next_pose = sapien.Pose(
                            [aug_obj[0], aug_obj[1], 0], [1, 0, 0, 0]
                        ) * sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

                if task_name == "dclaw":
                    if aug_step_obj > 0:
                        aug_step_obj -= 1
                        aug_obj = aug_obj + one_step_aug_obj
                    palm_next_pose = sapien.Pose(
                        [aug_obj[0], aug_obj[1], 0], [1, 0, 0, 0]
                    ) * sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

                palm_next_pose = robot_pose.inv() * palm_next_pose
                palm_delta_pose = palm_pose.inv() * palm_next_pose
                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(
                    palm_delta_pose.q
                )
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = (
                    palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                )
                delta_pose = np.concatenate(
                    [palm_next_pose.p - palm_pose.p,
                        delta_axis_world * delta_angle]
                )

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
                    env.robot.get_qpos()[: env.arm_dof]
                )
                arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[
                    : env.arm_dof
                ]
                arm_qpos = arm_qvel + env.robot.get_qpos()[: env.arm_dof]

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                visual_baked["obs"].append(env.get_observation())
                visual_baked["action"].append(
                    np.concatenate([delta_pose * 100, hand_qpos])
                )
                _, _, _, info = env.step(target_qpos)
                # env.render()
                # print(valid_frame)
                # print(info["object_total_rotate_angle"])
                rgb = env.get_observation(
                )["relocate_view-rgb"].cpu().detach().numpy()
                rgb_pic = (rgb * 255).astype(np.uint8)
                rgb_pics.append(rgb_pic)
    object_name = meta_data["env_kwargs"]["object_name"]
    all_data = np.load(path, allow_pickle=True)
    all_data["meta_data"] = meta_data
    all_data["meta_data"]["env_kwargs"]["task_name"] = task_name

    if info["success"]:
        with open(f"./sim/raw_data/dclaw_diverse/{object_name}_{demo_idx:04d}.pickle", "wb") as f:
            pickle.dump(all_data, f)
        imageio.mimsave(
            f"./temp/demos/player/dclaw_{object_name}_{demo_idx:04d}.mp4",
            rgb_pics,
            fps=60,
        )


def bake_visual_real_demonstration_test(retarget=False):
    from pathlib import Path

    # Recorder
    shutil.rmtree("./temp/demos/player", ignore_errors=True)
    os.makedirs("./temp/demos/player")
    path = "./sim/raw_data/pick_place_mustard_bottle/mustard_bottle_0001.pickle"
    # path = "sim/raw_data/xarm/less_random/pick_place_tomato_soup_can/tomato_soup_can_0001.pickle"
    # path = "sim/raw_data/xarm/less_random/pick_place_sugar_box/sugar_box_0001.pickle"
    # path = "sim/raw_data/dclaw/dclaw_3x_0002.pickle"

    all_data = np.load(path, allow_pickle=True)
    meta_data = all_data["meta_data"]
    task_name = meta_data["env_kwargs"]["task_name"]
    meta_data["env_kwargs"].pop("task_name")
    # meta_data['env_kwargs']['init_target_pos'] = sapien.Pose([-0.05, -0.105, 0], [1, 0, 0, 0])
    data = all_data["data"]
    use_visual_obs = True

    print(meta_data)
    if not retarget:
        robot_name = meta_data["robot_name"]
    else:
        robot_name = "allegro_hand_free"
    if "allegro" in robot_name:
        if "finger_control_params" in meta_data.keys():
            finger_control_params = meta_data["finger_control_params"]
        if "root_rotation_control_params" in meta_data.keys():
            root_rotation_control_params = meta_data["root_rotation_control_params"]
        if "root_translation_control_params" in meta_data.keys():
            root_translation_control_params = meta_data[
                "root_translation_control_params"
            ]
        if "robot_arm_control_params" in meta_data.keys():
            robot_arm_control_params = meta_data["robot_arm_control_params"]

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params["robot_name"] = robot_name
    env_params["use_visual_obs"] = use_visual_obs
    env_params["use_gui"] = True
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
    if "init_obj_pos" in meta_data["env_kwargs"].keys():
        env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]
    if "init_target_pos" in meta_data["env_kwargs"].keys():
        env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]
        print(env_params["init_target_pos"])
    if task_name == "pick_place":
        env = PickPlaceRLEnv(**env_params)
    elif task_name == "dclaw":
        env = DClawRLEnv(**env_params)
    elif task_name == "hammer":
        env = HammerRLEnv(**env_params)
    elif task_name == "table_door":
        env = TableDoorRLEnv(**env_params)
    elif task_name == "insert_object":
        env = InsertObjectRLEnv(**env_params)
    elif task_name == "mug_flip":
        env = MugFlipRLEnv(**env_params)
    else:
        raise NotImplementedError

    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(
                        *(1 * root_translation_control_params), mode="acceleration"
                    )
                elif (
                    "x_rotation_joint" in name
                    or "y_rotation_joint" in name
                    or "z_rotation_joint" in name
                ):
                    joint.set_drive_property(
                        *(1 * root_rotation_control_params), mode="acceleration"
                    )
                else:
                    joint.set_drive_property(
                        *(finger_control_params), mode="acceleration"
                    )
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(
                        *(1 * robot_arm_control_params), mode="force"
                    )
                else:
                    joint.set_drive_property(
                        *(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step
    env.reset()
    viewer = env.render()
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    viewer.set_camera_rpy(0, -np.pi / 6, np.pi / 4)

    real_camera_cfg = {
        "relocate_view": dict(
            pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(640, 480)
        )
    }

    if task_name == "table_door":
        camera_cfg = {
            "relocate_view": dict(
                position=np.array([-0.25, -0.25, 0.55]),
                look_at_dir=np.array([0.25, 0.25, -0.45]),
                right_dir=np.array([1, -1, 0]),
                fov=np.deg2rad(69.4),
                resolution=(224, 224),
            )
        }

    env.setup_camera_from_config(real_camera_cfg)
    # env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {
        "rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == "pick_place":
        player = PickPlaceEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "dclaw":
        player = DcLawEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "hammer":
        player = HammerEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "table_door":
        player = TableDoorEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "insert_object":
        player = InsertObjectEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif task_name == "mug_flip":
        player = FlipMugEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    else:
        raise NotImplementedError

    # Retargeting
    using_real = True
    if retarget:
        link_names = [
            "palm_center",
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_14.0",
            "link_2.0",
            "link_6.0",
            "link_10.0",
        ]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name()
                       for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(
            env.robot,
            joint_names,
            link_names,
            has_global_pose_limits=False,
            has_joint_limits=True,
        )
        baked_data = player.bake_demonstration(
            retargeting, method="tip_middle", indices=indices
        )
    elif using_real:
        path = "./real/raw_data/pick_place/0001.pkl"
        # path = "./real/raw_data/pick_place_tomato_soup_can/0000.pkl"
        # path = "./real/raw_data/pick_place_sugar_box/0000.pkl"
        # path = "./real/raw_data/dclaw_small_scale/0001.pkl"
        baked_data = np.load(path, allow_pickle=True)

    visual_baked = dict(obs=[], action=[])
    env.reset()

    player.scene.unpack(player.get_sim_data(0))
    # env.randomize_object_rotation()
    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    # 0000: obj_position = np.array([0.02, 0.3, 0.1])
    # 0001: obj_position = np.array([-0.05, 0.29, 0.1])
    # 0002: obj_position = np.array([0.08, 0.22, 0.1])
    # 0003: obj_position = np.array([-0.06, 0.25, 0.1])
    # 0004: obj_position = np.array([0, 0.28, 0.1])
    # 0005: obj_position = np.array([0.07, 0.29, 0.1])
    # 0006: obj_position = np.array([0.09, 0.27, 0.1])
    # 0007: obj_position = np.array([0, 0.27, 0.1])
    # 0008: obj_position = np.array([-0.09, 0.24, 0.1])
    # 0009: obj_position = np.array([-0.02, 0.25, 0.1])
    # 0010: obj_position = np.array([0.05, 0.24, 0.1])
    # 0011: obj_position = np.array([-0.08, 0.26, 0.1])
    # 0012: obj_position = np.array([0, 0.26, 0.1])
    # 0013: obj_position = np.array([0, 0.26, 0.1])
    # 0014: obj_position = np.array([-0.02, 0.24, 0.1])
    # 0015: obj_position = np.array([-0.02, 0.25, 0.1])
    # 0016: obj_position = np.array([-0.02, 0.25, 0.1])
    # 0017: obj_position = np.array([-0.02, 0.28, 0.1])
    # y: 0.32 - 0.26
    obj_position = np.array([-0.05, 0.29, 0.1])
    euler = np.deg2rad(30)
    orientation = transforms3d.euler.euler2quat(0, 0, euler)
    obj_pose = sapien.Pose(obj_position, orientation)
    env.manipulated_object.set_pose(obj_pose)

    # robot_base_pose = np.array([0, -0.7, 0, 0.707, 0, 0, 0.707])
    env.robot.set_qpos(baked_data[0]["teleop_cmd"])
    print("init_qpos: ", baked_data[0]["teleop_cmd"])

    robot_pose = env.robot.get_pose()
    rotation_matrix = transforms3d.quaternions.quat2mat(robot_pose.q)
    world_to_robot = transforms3d.affines.compose(
        -np.matmul(rotation_matrix.T,
                   robot_pose.p), rotation_matrix.T, np.ones(3)
    )

    ee_pose = baked_data[0]["ee_pose"]
    hand_qpos_prev = baked_data[0]["teleop_cmd"][env.arm_dof:]

    frame_skip = 1
    rgb_pics = []
    for idx in range(0, len(baked_data), frame_skip):
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data) - frame_skip:
            ee_pose_next = np.array(baked_data[idx + frame_skip]["ee_pose"])
            ee_pose_delta = np.sqrt(
                np.sum((ee_pose_next[:3] - ee_pose[:3]) ** 2))
            hand_qpos = baked_data[idx]["teleop_cmd"][env.arm_dof:]
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos

            object_pose = env.manipulated_object.pose.p
            palm_pose = env.ee_link.get_pose()

            # dist_object_hand = np.linalg.norm(object_pose - palm_pose.p)

            if (
                ee_pose_delta < 0.001
                and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2
            ):
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!skip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            else:
                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos

                palm_pose = robot_pose.inv() * palm_pose

                palm_next_pose = sapien.Pose(
                    ee_pose_next[0:3], ee_pose_next[3:7])
                palm_next_pose = robot_pose.inv() * palm_next_pose

                palm_delta_pose = palm_pose.inv() * palm_next_pose
                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(
                    palm_delta_pose.q
                )
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = (
                    palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
                )
                delta_pose = np.concatenate(
                    [palm_next_pose.p - palm_pose.p,
                        delta_axis_world * delta_angle]
                )

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
                    env.robot.get_qpos()[: env.arm_dof]
                )
                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
                    env.robot.get_qpos()[: env.arm_dof]
                )
                arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[
                    : env.arm_dof
                ]
                arm_qpos = arm_qvel + env.robot.get_qpos()[: env.arm_dof]

                # target_arm_qpos = baked_data[idx+1]["teleop_cmd"][:env.arm_dof]
                # diff = target_arm_qpos - env.robot.get_qpos()[:env.arm_dof]
                # qvel = diff / lab.arm_control_step / lab.safety_factor
                # qvel = np.clip(qvel, -0.3, 0.3)
                # arm_qpos = qvel + env.robot.get_qpos()[:env.arm_dof]

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                _, _, _, info = env.step(target_qpos)
                env.render()

                rgb = env.get_observation(
                )["relocate_view-rgb"].cpu().detach().numpy()
                rgb_pic = (rgb * 255).astype(np.uint8)
                rgb_pics.append(rgb_pic)

        imageio.mimsave("./temp/demos/player/relocate-rgb.mp4",
                        rgb_pics, fps=30)


if __name__ == "__main__":
    # bake_demonstration_adroit()
    # bake_demonstration_allegro_test()
    # bake_demonstration_svh_test()
    # bake_demonstration_ar10_test()
    # bake_demonstration_mano()
    # for i in range(1, 51):
    #     bake_visual_demonstration_test(retarget=False, idx=i)
    bake_visual_demonstration_test()
    # bake_visual_real_demonstration_test()
