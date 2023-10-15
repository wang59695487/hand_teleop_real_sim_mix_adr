import shutil
from typing import Dict, Any, Optional, List

import numpy as np
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
    aug_step_obj_length = 5 #hyperparameter
    aug_step_plate_length = 6 #hyperparameter
    aug_obj_chunk = []
    aug_plate_chunk = []
    chunk_sensitivity_obj = np.argsort(-np.array(chunk_sensitivity[1:lifted_chunk]))
    aug_obj_chunk = chunk_sensitivity_obj[:aug_step_obj_length]
    chunk_sensitivity_plate = np.argsort(-np.array(chunk_sensitivity[lifted_chunk:len(chunk_sensitivity)]))
    aug_plate_chunk = chunk_sensitivity_plate[:aug_step_plate_length]
    
    return aug_obj_chunk, aug_plate_chunk

def create_env(args, demo, retarget=False):
    
    robot_name=args['robot_name']
    
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
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'dclaw':
        env = DClawRLEnv(**env_params)
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
    
    real_camera_cfg = {"relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))}   
    env.setup_camera_from_config(real_camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'dclaw':
        player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    else:
        raise NotImplementedError

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

def generate_sim_aug_in_play_demo(args, demo, demo_idx, init_pose_aug_plate, init_pose_aug_obj, var_adr_light, frame_skip=1, retarget=False):

    env, task_name, meta_data, baked_data = create_env(args, demo=demo, retarget=retarget)
    
    robot_pose = env.robot.get_pose()

    visual_baked = dict(obs=[], action=[],robot_qpos=[])

    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]
    
    #################################Kinematic Augmentation####################################
    if task_name == 'pick_place':
        sensitive_chunk_file = f"{args['sim_dataset_folder']}/meta_data.pickle" 
        with open(sensitive_chunk_file,'rb') as file:
            sensitive_chunk_data = pickle.load(file)
        aug_obj_chunk, aug_plate_chunk = aug_in_non_sensitive_chunk(sensitive_chunk_data['lifted_chunks'][demo_idx], 
                                                                    chunk_sensitivity = sensitive_chunk_data['chunks_sensitivity'][demo_idx])
        aug_step_obj = len(aug_obj_chunk)*50
        aug_step_plate = len(aug_plate_chunk)*50
        meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug_plate * meta_data["env_kwargs"]['init_target_pos']
        env.plate.set_pose(meta_data["env_kwargs"]['init_target_pos'])
        aug_plate = np.array([init_pose_aug_obj.p[0],init_pose_aug_obj.p[1]])
        one_step_aug_plate = np.array([(-1*init_pose_aug_obj.p[0]+init_pose_aug_plate.p[0])/aug_step_plate, (-1*init_pose_aug_obj.p[1]+init_pose_aug_plate.p[1])/aug_step_plate])
    elif task_name == 'dclaw':
        aug_step_obj=50
    
    meta_data["env_kwargs"]['init_obj_pos'] = init_pose_aug_obj * meta_data["env_kwargs"]['init_obj_pos']
    env.manipulated_object.set_pose(meta_data["env_kwargs"]['init_obj_pos'])
    
    #################Avoid the case that the object is already close to the target or there is no chunk for augmentation################
    if (task_name == 'pick_place' and env._is_close_to_target()) or len(aug_obj_chunk)==0 or len(aug_plate_chunk)==0:
        return visual_baked, meta_data, False

    aug_obj = np.array([0,0])
    one_step_aug_obj = np.array([init_pose_aug_obj.p[0]/aug_step_obj, init_pose_aug_obj.p[0]/aug_step_obj])
    
    valid_frame = 0
    stop_frame = 0
    
    if args['randomness_rank'] >= 2:
        env.random_map(var_adr_light)
    
    chunk_idx = 0    
    for idx in tqdm(range(0,len(baked_data["obs"]),frame_skip)):
        # NOTE: robot.get_qpos() version
        if idx < len(baked_data['obs'])-frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos
            
            if ee_pose_delta < 0.001 and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2:
                continue

            else:
                valid_frame += 1

                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos
                palm_pose = env.ee_link.get_pose()
                palm_pose = robot_pose.inv() * palm_pose
                if task_name == 'pick_place':
                    if env._is_object_lifted():
                        if aug_step_plate > 0 and chunk_idx in aug_plate_chunk:
                            aug_step_plate -= 1
                            #print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_plate)
                            aug_plate = aug_plate + one_step_aug_plate
                        palm_next_pose = sapien.Pose([aug_plate[0],aug_plate[1],0],[1,0,0,0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                        
                    elif not env._is_object_lifted():
                        if aug_step_obj > 0 and chunk_idx in aug_obj_chunk:
                            aug_step_obj -= 1
                            #print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_obj)
                            aug_obj = aug_obj + one_step_aug_obj
                        palm_next_pose = sapien.Pose([aug_obj[0],aug_obj[1],0],[1,0,0,0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                elif task_name == 'dclaw':
                    if aug_step_obj > 0:
                        aug_step_obj -= 1
                        #print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_obj)
                        aug_obj = aug_obj + one_step_aug_obj
                    palm_next_pose = sapien.Pose([aug_obj[0],aug_obj[1],0],[1,0,0,0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                    
                ########### Add Light Randomness ############
                if args['randomness_rank'] >= 2 and valid_frame%50 == 0:
                    env.random_light(var_adr_light)
                    env.generate_random_object_texture(var_adr_light)
                    chunk_idx += 1
                    
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

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                observation = env.get_observation()
                visual_baked["obs"].append(observation)
                visual_baked["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                # Using robot qpos version
                visual_baked["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),
                                                    env.ee_link.get_pose().p,env.ee_link.get_pose().q]))
                _, _, _, info = env.step(target_qpos)

                if task_name == 'pick_place':
                    info_success = info["is_object_lifted"] and env._object_target_distance() <= 0.2 and env._is_object_plate_contact()    
                elif task_name == 'dclaw':
                    info_success = info['success']

                if info_success:
                    stop_frame += 1

                if stop_frame >= 30:
                    break
                
    if valid_frame < 300:           
        return visual_baked, meta_data, False
    
    return visual_baked, meta_data, info_success


def generate_sim_aug(args, all_data, init_pose_aug_plate, init_pose_aug_obj, aug_step_plate=400, aug_step_obj=500, retarget=False, frame_skip=1):

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
    if task_name == 'pick_place':
        env = PickPlaceRLEnv(**env_params)
    elif task_name == 'dclaw':
        env = DClawRLEnv(**env_params)
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
    
    real_camera_cfg = {
        "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
    }
    
    if task_name == 'table_door':
         camera_cfg = {
        "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }   
         
    env.setup_camera_from_config(real_camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)

    # Player
    if task_name == 'pick_place':
        player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
    elif task_name == 'dclaw':
        player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
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
    robot_pose = env.robot.get_pose()

    data = {"simulation": [], "action": [], "robot_qpos": []}
    rgb_pics = []

    ee_pose = baked_data["ee_pose"][0]
    hand_qpos_prev = baked_data["action"][0][env.arm_dof:]
   
    meta_data["env_kwargs"]['init_target_pos'] = init_pose_aug_plate * meta_data["env_kwargs"]['init_target_pos']
    meta_data["env_kwargs"]['init_obj_pos'] = init_pose_aug_obj * meta_data["env_kwargs"]['init_obj_pos']
    env.plate.set_pose(meta_data["env_kwargs"]['init_target_pos'])
    env.manipulated_object.set_pose(meta_data["env_kwargs"]['init_obj_pos'])

    #################Avoid the case that the object is already close to the target################
    if task_name == 'pick_place' and env._is_close_to_target():
        return False, _ , rgb_pics
    
    aug_obj = np.array([0,0])
    one_step_aug_obj = np.array([init_pose_aug_obj.p[0]/aug_step_obj, init_pose_aug_obj.p[0]/aug_step_obj])
    aug_plate = np.array([init_pose_aug_obj.p[0],init_pose_aug_obj.p[1]])
    one_step_aug_plate = np.array([(-1*init_pose_aug_obj.p[0]+init_pose_aug_plate.p[0])/aug_step_plate, (-1*init_pose_aug_obj.p[1]+init_pose_aug_plate.p[1])/aug_step_plate])

    stop_frame = 0
    valid_frame = 0
    for idx in tqdm(range(0,len(baked_data["obs"]),frame_skip)):

        # NOTE: robot.get_qpos() version
        if idx < len(baked_data['obs'])-frame_skip:
            ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
            ee_pose_delta = np.sqrt(np.sum((ee_pose_next[:3] - ee_pose[:3])**2))
            hand_qpos = baked_data["action"][idx][env.arm_dof:]
            delta_hand_qpos = hand_qpos - hand_qpos_prev if idx!=0 else hand_qpos

            if ee_pose_delta < args['delta_ee_pose_bound'] and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2:
                #print("!!!!!!!!!!!!!!!!!!!!!!skip!!!!!!!!!!!!!!!!!!!!!")
                continue
                
            else:
                valid_frame += 1
                
                ee_pose = ee_pose_next
                hand_qpos_prev = hand_qpos
                palm_pose = env.ee_link.get_pose()
                palm_pose = robot_pose.inv() * palm_pose

                if env._is_object_lifted():
                    if aug_step_plate > 0:
                        aug_step_plate -= 1
                        # print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_plate)
                        aug_plate = aug_plate + one_step_aug_plate
                    palm_next_pose = sapien.Pose([aug_plate[0],aug_plate[1],0],[1,0,0,0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])
                    
                elif not env._is_object_lifted():
                    if aug_step_obj > 0:
                        aug_step_obj -= 1
                        # print("!!!!!!!!!!!!!!!!!!!!!!Alter!!!!!!!!!!!!!!!!!!!!!", aug_step_obj)
                        aug_obj = aug_obj + one_step_aug_obj
                    palm_next_pose = sapien.Pose([aug_obj[0],aug_obj[1],0],[1,0,0,0])*sapien.Pose(ee_pose_next[0:3], ee_pose_next[3:7])

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

                target_qpos = np.concatenate([arm_qpos, hand_qpos])
                data["simulation"].append(env.scene.pack())
                data["action"].append(np.concatenate([delta_pose*100, hand_qpos]))
                data["robot_qpos"].append(np.concatenate([env.robot.get_qpos(),env.ee_link.get_pose().p,env.ee_link.get_pose().q]))
                rgb_pics.append(env.get_observation()["relocate_view-rgb"].cpu().detach().numpy())
                _, _, _, info = env.step(target_qpos)
                
                if task_name == 'pick_place':
                    info_success = info["is_object_lifted"] and env._object_target_distance() <= 0.2 and env._is_object_plate_contact()

                if info_success:
                    stop_frame += 1

                if  stop_frame >= 30:
                    break
                                    
    meta_data["env_kwargs"]['task_name'] = task_name
    augment_data = {'data': data, 'meta_data': meta_data}

    
    for i in range(len(rgb_pics)):
        rgb = rgb_pics[i]
        rgb_pics[i] = (rgb * 255).astype(np.uint8)
    
    if valid_frame < 300:
        return False, augment_data, rgb_pics

    return info_success, augment_data, rgb_pics


def player_augmenting(args):

    np.random.seed(args['seed'])

    demo_files = []
    sim_data_path = f"{args['sim_demo_folder']}/{args['task_name']}_{args['object_name']}/"
    for file_name in os.listdir(sim_data_path ):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(sim_data_path, file_name))
    print('Augmenting sim demos and creating the dataset:')
    print('---------------------')

    for demo_id, file_name in enumerate(demo_files):

        num_test = 0
        # if demo_id <= 5:
        #     continue
        with open(file_name, 'rb') as file:
            demo = pickle.load(file)

        for i in range(400):
            
            x1,x2,y1,y2 = np.random.uniform(-0.12,0.12,4)
        
            if np.fabs(x1) <= 0.01 and np.fabs(y1) <= 0.01:
                continue
            elif np.fabs(x2) <= 0.01 and np.fabs(y2) <= 0.01:
                continue

            all_data = copy.deepcopy(demo)

            out_folder = f"./sim/raw_augmentation_action/{args['task_name']}_{args['object_name']}_aug/"
            os.makedirs(out_folder, exist_ok=True)

            # if len(os.listdir(out_folder)) == 0:
            #     num_test = "0001"
            # else:
            #     pkl_files = os.listdir(out_folder)
            #     last_num = sorted([int(x.replace(".pickle", "").split("_")[-1]) for x in pkl_files])[-1]
            #     num_test = str(last_num + 1).zfill(4)
            init_pose_aug_obj = sapien.Pose([x1, y1, 0], [1, 0, 0, 0])
            init_pose_aug_plate = sapien.Pose([x2, y2, 0], [1, 0, 0, 0])
            info_success, data, video = generate_sim_aug(args, all_data=all_data, init_pose_aug_plate=init_pose_aug_plate,init_pose_aug_obj=init_pose_aug_obj,frame_skip=args['frame_skip'], retarget=args['retarget'])
            #imageio.mimsave(f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{num_test}_x{x:.2f}_y{y:.2f}.mp4", video, fps=120)
            if info_success:

                print("##############SUCCESS##############")
                num_test += 1
                print("##########This is {}th try and {}th success##########".format(i+1,num_test))

                imageio.mimsave(f"./temp/demos/aug_{args['object_name']}/demo_{demo_id+1}_{num_test}_x1{x1:.2f}_y1{y1:.2f}_x2{x2:.2f}_y2{y2:.2f}.mp4", video, fps=120)
                
                dataset_folder = f"{out_folder}/demo_{demo_id+1}_{num_test}.pickle"

                with open(dataset_folder,'wb') as file:
                    pickle.dump(data, file)
            
            if num_test == args['kinematic_aug']:
                break
            
            
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=20230923, type=int)
    parser.add_argument("--sim-demo-folder",default='./sim/raw_data/', type=str)
    parser.add_argument("--task-name", required=True, type=str)
    parser.add_argument("--object-name", required=True, type=str)
    parser.add_argument("--kinematic-aug", default=100, type=int)
    parser.add_argument("--frame-skip", default=1, type=int)
    parser.add_argument("--retarget", default=False, type=bool)
    parser.add_argument("--delta-ee-pose-bound", default=0, type=float)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
     
    args = {
        'seed': args.seed,
        'sim_demo_folder' : args.sim_demo_folder,
        'task_name': args.task_name,
        'object_name': args.object_name,
        'kinematic_aug': args.kinematic_aug,
        'frame_skip': args.frame_skip,
        'delta_ee_pose_bound': args.delta_ee_pose_bound,
        'retarget': args.retarget
    }

    player_augmenting(args)
        

    
    
