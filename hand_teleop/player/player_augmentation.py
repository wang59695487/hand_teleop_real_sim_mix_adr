import shutil
from typing import Dict, Any, Optional, List

import numpy as np
from numpy import random
import sapien.core as sapien
import transforms3d
import pickle
import os
import imageio
from tqdm import tqdm
import copy
from argparse import ArgumentParser

from hand_teleop.env.rl_env.base import BaseRLEnv, compute_inverse_kinematics
from hand_teleop.utils.common_robot_utils import LPFilter
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.real_world import lab
from hand_teleop.player.player import *


def aug_in_non_sensitive_chunk(lifted_chunk, chunk_sensitivity):
    aug_step_obj_length = 3  # hyperparameter
    aug_step_target_length = 3  # hyperparameter
    aug_obj_chunk = []
    aug_target_chunk = []
    chunk_sensitivity_obj = np.argsort(
        -np.array(chunk_sensitivity[:lifted_chunk]))
    aug_obj_chunk = chunk_sensitivity_obj[:aug_step_obj_length] if aug_step_obj_length <= lifted_chunk else chunk_sensitivity_obj
    chunk_sensitivity_target = np.argsort(
        -np.array(chunk_sensitivity[lifted_chunk:len(chunk_sensitivity)]))
    aug_target_chunk = chunk_sensitivity_target[:aug_step_target_length] if aug_step_target_length <= len(
        chunk_sensitivity)-lifted_chunk else chunk_sensitivity_target

    return aug_obj_chunk, aug_target_chunk


def create_env(args, demo, retarget=False):

    robot_name = args['robot_name']

    if robot_name == 'mano':
        assert retarget == False
    # Get env params
    meta_data = demo["meta_data"]
    if not retarget:
        assert robot_name == meta_data["robot_name"]
    task_name = meta_data["env_kwargs"]['task_name']
    meta_data["env_kwargs"].pop('task_name')
    meta_data["task_name"] = task_name

    data = demo["data"]
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
    env_params['use_gui'] = False

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
    if 'pick_place' in task_name:
        if args['object_name'] == 'diverse_objects':
            bottle_id = np.random.randint(0, 10)
            if bottle_id == 9:
                env_params["object_category"] = "YCB"
                env_params["object_name"] = "mustard_bottle"
            else:
                env_params["object_category"] = "SHAPE_NET"
                env_params["object_name"] = "bottle_{}".format(bottle_id)
        env = PickPlaceRLEnv(**env_params)
    elif 'dclaw' in task_name:
        if args['object_name'] == 'diverse_objects':
            dclaw_id = np.random.randint(0, 8)
            env_params["object_name"] = dclaw_diverse_objects[dclaw_id]

        env = DClawRLEnv(**env_params)
    elif 'pour' in task_name:
        env = PourBoxRLEnv(**env_params)
        meta_data["env_kwargs"]["init_target_pos"] = env.target_pose
    else:
        raise NotImplementedError

    if not retarget:
        if "free" in robot_name:
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(
                        *(1 * root_translation_control_params), mode="acceleration")
                elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                    joint.set_drive_property(
                        *(1 * root_rotation_control_params), mode="acceleration")
                else:
                    joint.set_drive_property(
                        *(finger_control_params), mode="acceleration")
            env.rl_step = env.simple_sim_step
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in env.robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(
                        *(1 * robot_arm_control_params), mode="force")
                else:
                    joint.set_drive_property(
                        *(1 * finger_control_params), mode="force")
            env.rl_step = env.simple_sim_step

    env.reset()

    if "multi_view" in args['task_name']:
        ry = random.uniform(-15, 15)
        rz = random.uniform(-15, 15)
        quat = transforms3d.euler.euler2quat(
            0, np.deg2rad(ry), np.deg2rad(rz))
        aug_view_pose = sapien.Pose([0.05, 0.05, 0], quat)
    else:
        aug_view_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    real_camera_cfg = {
        "relocate_view": dict(
            pose=aug_view_pose*lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224)
        )
    }
    env.setup_camera_from_config(real_camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {
        "rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if 'pick_place' in task_name:
        player = PickPlaceEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif 'dclaw' in task_name:
        player = DcLawEnvPlayer(meta_data, data, env,
                                zero_joint_pos=env_params["zero_joint_pos"])
    elif 'pour' in task_name:
        player = PourEnvPlayer(meta_data, data, env,
                               zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

    if retarget:
        link_names = ["palm_center", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_14.0",
                      "link_2.0", "link_6.0", "link_10.0"]
        indices = [0, 1, 2, 3, 5, 6, 7, 8]
        joint_names = [joint.get_name()
                       for joint in env.robot.get_active_joints()]
        retargeting = PositionRetargeting(env.robot, joint_names, link_names, has_global_pose_limits=False,
                                          has_joint_limits=True)
        baked_data = player.bake_demonstration(
            retargeting, method="tip_middle", indices=indices)
    else:
        baked_data = player.bake_demonstration()

    env.reset()
    player.scene.unpack(player.get_sim_data(0))

    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    env.reset()
    env.robot.set_qpos(baked_data["robot_qpos"][0])
    if baked_data["robot_qvel"] != []:
        env.robot.set_qvel(baked_data["robot_qvel"][0])

    return env, task_name, meta_data, baked_data


def generate_sim_aug_in_play_demo(args, demo, demo_idx, init_pose_aug_target, init_pose_aug_obj, var_adr_light, frame_skip=1, retarget=False, is_video=False):

    env, task_name, meta_data, baked_data = create_env(
        args, demo=demo, retarget=retarget)

    robot_pose = env.robot.get_pose()

    visual_baked = dict(obs=[], action=[], robot_qpos=[])

    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]

    ################################# Kinematic Augmentation####################################
    if task_name in ["pick_place", 'pour']:
        if args["sensitivity_check"]:
            sensitive_chunk_file = f"{args['sim_dataset_folder']}/meta_data.pickle"
            with open(sensitive_chunk_file, 'rb') as file:
                sensitive_chunk_data = pickle.load(file)
            aug_obj_chunk, aug_target_chunk = aug_in_non_sensitive_chunk(sensitive_chunk_data['lifted_chunks'][demo_idx],
                                                                        chunk_sensitivity=sensitive_chunk_data['chunks_sensitivity'][demo_idx])
            aug_step_obj = len(aug_obj_chunk)*50
            aug_step_target = len(aug_target_chunk)*50
            meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug_target * \
                meta_data["env_kwargs"]['init_target_pos']
            env.target_object.set_pose(meta_data["env_kwargs"]['init_target_pos'])
            aug_target = np.array([init_pose_aug_obj.p[0], init_pose_aug_obj.p[1]])
            one_step_aug_target = np.array([(-1*init_pose_aug_obj.p[0]+init_pose_aug_target.p[0]) /
                                            aug_step_target, (-1*init_pose_aug_obj.p[1]+init_pose_aug_target.p[1])/aug_step_target])
        aug_step_obj = 150
        aug_step_target = 200
        meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug_target * \
            meta_data["env_kwargs"]['init_target_pos']
        env.target_object.set_pose(meta_data["env_kwargs"]['init_target_pos'])
        aug_target = np.array([init_pose_aug_obj.p[0], init_pose_aug_obj.p[1]])
        one_step_aug_target = np.array([(-1*init_pose_aug_obj.p[0]+init_pose_aug_target.p[0]) /
                                       aug_step_target, (-1*init_pose_aug_obj.p[1]+init_pose_aug_target.p[1])/aug_step_target])
    elif 'dclaw' in task_name:
        aug_step_obj = 100

    meta_data["env_kwargs"]['init_obj_pos'] = init_pose_aug_obj * \
        meta_data["env_kwargs"]['init_obj_pos']
    env.manipulated_object.set_pose(meta_data["env_kwargs"]['init_obj_pos'])
    if 'pour' in task_name:
        for i in range(len(env.boxes)):
            env.boxes[i].set_pose(meta_data["env_kwargs"]['init_obj_pos'])

    ################# Avoid the case that the object is already close to the target or there is no chunk for augmentation################
    if (task_name in ["pick_place", "pour"] and env._is_close_to_target()):
        return visual_baked, meta_data, False
    elif 'dclaw' in task_name:
        env.object_total_rotate_angle = 0
        env.object_angle = env.get_object_rotate_angle()

    aug_obj = np.array([0, 0])
    one_step_aug_obj = np.array(
        [init_pose_aug_obj.p[0]/aug_step_obj, init_pose_aug_obj.p[1]/aug_step_obj])

    valid_frame = 0
    stop_frame = 0

    if args['randomness_rank'] >= 2:
        env.random_map(var_adr_light)
        env.random_light(var_adr_light)
        env.generate_random_object_texture(var_adr_light)
    rgb_pics = []

    ################# Add 100 Interpolation steps ################
    start_idx = 0
    for i in tqdm(range(0, 100, frame_skip)):
        if i % 10 == 0:
            start_idx += 1
        ee_pose_next = baked_data["ee_pose"][start_idx + frame_skip]
        hand_qpos = baked_data["action"][start_idx][env.arm_dof:]
        palm_pose = env.ee_link.get_pose()
        palm_pose = robot_pose.inv() * palm_pose
        aug_step_obj -= 1
        aug_obj = aug_obj + one_step_aug_obj
        palm_next_pose = sapien.Pose([aug_obj[0], aug_obj[1], 0], [
            1, 0, 0, 0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
        ########## Add Light Randomness ############
        if args['randomness_rank'] >= 2 and i % 50 == 0:
            env.random_light(var_adr_light)
            env.generate_random_object_texture(var_adr_light)

        palm_next_pose = robot_pose.inv() * palm_next_pose
        palm_delta_pose = palm_pose.inv() * palm_next_pose

        delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(
            palm_delta_pose.q)
        if delta_angle > np.pi:
            delta_angle = 2 * np.pi - delta_angle
            delta_axis = -delta_axis
        delta_axis_world = palm_pose.to_transformation_matrix()[
            :3, :3] @ delta_axis
        delta_pose = np.concatenate(
            [palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

        palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
            env.robot.get_qpos()[:env.arm_dof])
        arm_qvel = compute_inverse_kinematics(
            delta_pose, palm_jacobian)[:env.arm_dof]
        arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]

        target_qpos = np.concatenate([arm_qpos, hand_qpos])
        observation = env.get_observation()
        visual_baked["obs"].append(observation)
        visual_baked["action"].append(
            np.concatenate([delta_pose*100, hand_qpos]))
        # Using robot qpos version
        visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                          env.ee_link.get_pose().p, env.ee_link.get_pose().q]))
        _, _, _, info = env.step(target_qpos)
        if is_video:
            rgb = env.get_observation(
            )["relocate_view-rgb"].cpu().detach().numpy()
            rgb_pic = (rgb * 255).astype(np.uint8)
            rgb_pics.append(rgb_pic)

    ee_pose = baked_data["ee_pose"][start_idx]
    sim_ee_delta_pose_bound = 0.001 if task_name != "dclaw" else 0.0005
    for idx in tqdm(range(start_idx+1, len(baked_data["obs"]), frame_skip)):
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data['obs'])-frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(
                np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos

            if ee_pose_delta < sim_ee_delta_pose_bound and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2:
                continue

            else:
                valid_frame += 1
                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos
                palm_pose = env.ee_link.get_pose()
                palm_pose = robot_pose.inv() * palm_pose
                if task_name in ['pick_place', 'pour']:
                    if env._is_object_lifted():
                        if aug_step_target > 0:
                            aug_step_target -= 1
                            # print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_target)
                            aug_target = aug_target + one_step_aug_target
                        palm_next_pose = sapien.Pose([aug_target[0], aug_target[1], 0], [
                                                     1, 0, 0, 0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

                    elif not env._is_object_lifted():
                        if aug_step_obj > 0:
                            aug_step_obj -= 1
                            # print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_obj)
                            aug_obj = aug_obj + one_step_aug_obj
                        palm_next_pose = sapien.Pose([aug_obj[0], aug_obj[1], 0], [
                                                     1, 0, 0, 0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                elif task_name == 'dclaw':
                    if aug_step_obj > 0:
                        aug_step_obj -= 1
                        # print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_obj)
                        aug_obj = aug_obj + one_step_aug_obj
                    palm_next_pose = sapien.Pose([aug_obj[0], aug_obj[1], 0], [
                                                 1, 0, 0, 0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

                ########### Add Light Randomness ############
                if args['randomness_rank'] >= 2 and valid_frame % 50 == 0:
                    env.random_light(var_adr_light)
                    env.generate_random_object_texture(var_adr_light)

                palm_next_pose = robot_pose.inv() * palm_next_pose
                palm_delta_pose = palm_pose.inv() * palm_next_pose

                delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(
                    palm_delta_pose.q)
                if delta_angle > np.pi:
                    delta_angle = 2 * np.pi - delta_angle
                    delta_axis = -delta_axis
                delta_axis_world = palm_pose.to_transformation_matrix()[
                    :3, :3] @ delta_axis
                delta_pose = np.concatenate(
                    [palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])

                palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
                    env.robot.get_qpos()[:env.arm_dof])
                arm_qvel = compute_inverse_kinematics(
                    delta_pose, palm_jacobian)[:env.arm_dof]
                arm_qpos = arm_qvel + env.robot.get_qpos()[:env.arm_dof]

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                observation = env.get_observation()
                visual_baked["obs"].append(observation)
                visual_baked["action"].append(
                    np.concatenate([delta_pose*100, hand_qpos]))
                # Using robot qpos version
                visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                                  env.ee_link.get_pose().p, env.ee_link.get_pose().q]))
                _, _, _, info = env.step(target_qpos)
                if is_video:
                    rgb = env.get_observation(
                    )["relocate_view-rgb"].cpu().detach().numpy()
                    rgb_pic = (rgb * 255).astype(np.uint8)
                    rgb_pics.append(rgb_pic)

                info_success = info['success']

                if info_success:
                    if task_name in ["dclaw"]:
                        break
                    stop_frame += 1

                if stop_frame >= 30:
                    break

    if valid_frame < 300:
        return visual_baked, meta_data, False
    if not is_video:
        return visual_baked, meta_data, info_success
    else:
        return info_success, rgb_pics


def player_augmenting(args):

    np.random.seed(args['seed'])

    demo_files = []
    sim_data_path = f"{args['sim_demo_folder']}"
    for file_name in os.listdir(sim_data_path):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(sim_data_path, file_name))
    print('Augmenting sim demos and creating the dataset:')
    print('---------------------')

    for i in range(400):
        for demo_id, file_name in enumerate(demo_files):
            random.seed()
            demo_idx = file_name.split("/")[-1].split(".")[0]
            num_success = 0
          
            with open(file_name, 'rb') as file:
                demo = pickle.load(file)

            if args['task_name'] in ['pick_place', 'dclaw']:
                x1, x2, y1, y2 = np.random.uniform(-0.12, 0.12, 4)
            elif args['task_name'] == 'pour':
                x1 = np.random.uniform(0.12, 0.2)
                y1 = np.random.uniform(-0.2, 0.02)
                x2 = np.random.uniform(-0.12, 0.12)
                y2 = np.random.uniform(-0.02, 0.12)

            if np.fabs(x1) <= 0.02 and np.fabs(y1) <= 0.02:
                continue
            elif np.fabs(x2) <= 0.02 and np.fabs(y2) <= 0.02:
                continue

            all_data = copy.deepcopy(demo)

            out_folder = f"./sim/raw_augmentation_action/{args['task_name']}_{args['object_name']}_aug/"
            os.makedirs(out_folder, exist_ok=True)

            init_pose_aug_obj = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
            init_pose_aug_target = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

            info_success, video = generate_sim_aug_in_play_demo(
                args, all_data, demo_idx, init_pose_aug_target, init_pose_aug_obj, var_adr_light=3, is_video=args['save_video'])
            if args['save_video']:
                os.makedirs(f"./temp/demos/aug_{args['object_name']}/", exist_ok=True)
                imageio.mimsave(
                    f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{info_success}_{num_success}_x1{x1:.2f}_y1{y1:.2f}_x2{x2:.2f}_y2{y2:.2f}.mp4", video, fps=120)
            print(info_success)
            if info_success:
                print("##############SUCCESS##############")
                num_success += 1
                print("##########This is {}th try and {}th success##########".format(
                    i+1, num_success))
                    # imageio.mimsave(
                    #     f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{num_success}_x1{x1:.2f}_y1{y1:.2f}_x2{x2:.2f}_y2{y2:.2f}.mp4", video, fps=120)

            if num_success == args['kinematic_aug']:
                break


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=20230923, type=int)
    parser.add_argument("--sim-demo-folder", default=None)
    parser.add_argument("--sim-dataset-folder", default=None, type=str)
    parser.add_argument("--task-name", required=True, type=str)
    parser.add_argument("--object-name", required=True, type=str)
    parser.add_argument("--kinematic-aug", default=100, type=int)
    parser.add_argument("--frame-skip", default=1, type=int)
    parser.add_argument("--retarget", default=False, type=bool)
    parser.add_argument("--delta-ee-pose-bound", default=0, type=float)
    parser.add_argument("--randomness-rank", default=1, type=int)
    parser.add_argument("--save-video", default=False, type=bool)
    parser.add_argument("--sensitivity-check", default=False, type=bool)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    args = {
        'seed': args.seed,
        'sim_demo_folder': args.sim_demo_folder,
        'sim_dataset_folder': args.sim_dataset_folder,
        'task_name': args.task_name,
        'object_name': args.object_name,
        'kinematic_aug': args.kinematic_aug,
        'frame_skip': args.frame_skip,
        'delta_ee_pose_bound': args.delta_ee_pose_bound,
        'randomness_rank': args.randomness_rank,
        'retarget': args.retarget,
        'robot_name': "xarm6_allegro_modified_finger",
        'save_video': args.save_video,
        'sensitivity_check': args.sensitivity_check,
    }

    player_augmenting(args)
