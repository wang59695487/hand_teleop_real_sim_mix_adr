import numpy as np
import torch
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import dataset.bc_dataset as bc_dataset

from main.policy.bc_agent import make_agent
from feature_extractor import generate_feature_extraction_model
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.real_world import task_setting
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.player.randomization_utils import *
from hand_teleop.player.player import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(32)

def apply_IK_get_real_action(action,env,qpos,use_visual_obs):
    if not use_visual_obs:
        action = action/10
    # action = action/10    
    delta_pose = np.squeeze(action)[:env.arm_dof]/100
    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(qpos[:env.arm_dof])
    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[:env.arm_dof]
    arm_qpos = arm_qvel + qpos[:env.arm_dof]
    hand_qpos = np.squeeze(action)[env.arm_dof:]
    target_qpos = np.concatenate([arm_qpos, hand_qpos])
    return target_qpos


def main(args):
    checkpoint = torch.load(args["weight_path"])
    saved_args = checkpoint['args']
    dataset_folder = args['dataset_folder']
    with open("{}/{}_meta_data.pickle".format(dataset_folder, args["backbone_type"]),'rb') as file:
        meta_data = pickle.load(file)
    # --Create Env and Robot-- #
    robot_name = args["robot_name"]
    # task_name = meta_data['task_name']
    task_name = "pick_place"
    if 'randomness_scale' in meta_data["env_kwargs"].keys():
        randomness_scale = meta_data["env_kwargs"]['randomness_scale']
    else:
        randomness_scale = 1
    rotation_reward_weight = 0
    use_visual_obs = args['use_visual_obs']
    if 'allegro' in robot_name:
        if 'finger_control_params' in meta_data.keys():
            finger_control_params = meta_data['finger_control_params']
        if 'root_rotation_control_params' in meta_data.keys():
            root_rotation_control_params = meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in meta_data.keys():
            root_translation_control_params = meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in meta_data.keys():
            robot_arm_control_params = meta_data['robot_arm_control_params']            

    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = True
    # env_params['use_gui'] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]

    if 'init_obj_pos' in meta_data["env_kwargs"].keys():
        print('Found initial object pose')
        env_params['init_obj_pos'] = meta_data["env_kwargs"]['init_obj_pos']
        object_pos = meta_data["env_kwargs"]['init_obj_pos']

    if 'init_target_pos' in meta_data["env_kwargs"].keys():
        print('Found initial target pose')
        env_params['init_target_pos'] = meta_data["env_kwargs"]['init_target_pos']
        target_pos = meta_data["env_kwargs"]['init_target_pos']

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
    env.seed(0)
    env.reset()

    # viewer = env.render(mode="human")
    # add_default_scene_light(env.scene, env.renderer)
    # env.viewer = viewer
    # # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    # viewer.set_camera_xyz(-0.6, 0.6, 0.6)
    # viewer.set_camera_rpy(0, -np.pi/6, np.pi/4)
    # if task_name == 'table_door':
    #     viewer.set_camera_xyz(-0.6, -0.6, 0.6)
    #     viewer.set_camera_rpy(0, -np.pi/6, -np.pi/4)

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

    if args['use_visual_obs']:
        # Create camera
        # camera_cfg = {
        #     "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
        #                             right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        # }
        camera_cfg = {
            "relocate_view": dict(position=np.array([0.25, 0.25, 0.45]), look_at_dir=np.array([-0.25, -0.25, -0.35]),
                                    right_dir=np.array([-1, 1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        }
        if task_name == 'table_door':
            camera_cfg = {
            "relocate_view": dict(position=np.array([-0.25, -0.25, 0.55]), look_at_dir=np.array([0.25, 0.25, -0.45]),
                                    right_dir=np.array([1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
            }           
        env.setup_camera_from_config(camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)
        # viewer.toggle_axes(False)
        # viewer.toggle_camera_lines(False)

    with open('{}/{}_dataset.pickle'.format(dataset_folder, args["backbone_type"]), 'rb') as file:
        print('dataset_folder: {}'.format(dataset_folder))
        dataset = pickle.load(file)
        print(dataset.keys())
        if 'state' in dataset.keys():
            init_robot_qpos = dataset['state'][0][-7-env.robot.dof:-7]
            state_shape = len(dataset['state'][0])
            concatenated_obs_shape = None
            # print('State shape: {}'.format(state_shape))
        else:
            init_robot_qpos = dataset['obs'][0][-7-env.robot.dof:-7]
            concatenated_obs_shape = len(dataset['obs'][0])
            state_shape = None
        action_shape = len(dataset['action'][0])

    # --Create and Configure Agent-- #
    agent = make_agent(concatenated_obs_shape=concatenated_obs_shape, state_shape=state_shape, action_shape=action_shape, args=saved_args, frame_stack=4)
    agent.load(weight_path=args['weight_path'])
    if concatenated_obs_shape != None:
        feature_extractor = generate_feature_extraction_model(backbone_type=args['model_backbone'])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

    manual_action = False
    env.robot.set_qpos(init_robot_qpos)
    eval_idx = 0
    for x in np.arange(-0.1, 0.0, 0.02): # -0.15, 0.18, 0.03  # -0.1, 0.0, 0.02
        for y in np.arange(0.1, 0.2, 0.02): # 0.05, 0.2, 0.05 # 0.1, 0.2, 0.02
            video = []
            if args['random_obj_pos']:
                idx = np.random.randint(len(meta_data['init_obj_poses']))
                sampled_pos = meta_data['init_obj_poses'][idx]
                object_pos = sampled_pos.p + np.array([np.random.normal(0,0.003), np.random.normal(0,0.003), 0])
                object_pos = sapien.Pose(p=object_pos,q=sampled_pos.q)
            print('Object Pos: {}'.format(object_pos))
            env.reset()
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
            env.robot.set_qpos(init_robot_qpos)
            env.manipulated_object.set_pose(object_pos)
            for _ in range(10*env.frame_skip):
                env.scene.step()
            obs = env.get_observation()
            if args['use_visual_obs']:
                features = []
                robot_states = []
                next_robot_states = []
                rgb_imgs = []
                next_rgb_imgs = []
            else:
                oracle_obs = []
            for i in range(1500):
                video.append(obs["relocate_view-rgb"])
                if concatenated_obs_shape != None:
                    assert args['adapt'] == False
                    if args['use_visual_obs']:
                        features, robot_states, obs = bc_dataset.get_stacked_data_from_obs(rgb_imgs=features, robot_states=robot_states, obs=obs, i=i, concatenate=True, feature_extractor=feature_extractor)
                        obs = obs.reshape((1,-1))
                    else:
                        oracle_obs.append(obs)
                        j = len(oracle_obs)-1
                        if j==0:
                            stacked_obs = np.concatenate((oracle_obs[j],oracle_obs[j],oracle_obs[j],oracle_obs[j]))    
                        elif j==1:
                            stacked_obs = np.concatenate((oracle_obs[j-1],oracle_obs[j],oracle_obs[j],oracle_obs[j]))
                        elif j==2:
                            stacked_obs = np.concatenate((oracle_obs[j-2],oracle_obs[j-1],oracle_obs[j],oracle_obs[j]))         
                        else:
                            stacked_obs = np.concatenate((oracle_obs[j-3],oracle_obs[j-2],oracle_obs[j-1],oracle_obs[j]))
                        obs = torch.from_numpy(stacked_obs).to(device)
                        obs = obs.reshape((1,-1))
                # print('State shape: {}'.format(state_shape))
                if state_shape != None:
                    rgb_imgs, robot_states, stacked_imgs, stacked_states = bc_dataset.get_stacked_data_from_obs(rgb_imgs=rgb_imgs, robot_states=robot_states, obs=obs, i=i, concatenate=False)
                
                if manual_action:
                    action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
                else:
                    agent.train(train_visual_encoder=False, train_state_encoder=False, train_policy=False, train_inv=False)
                    if concatenated_obs_shape != None:
                        action = agent.validate(concatenated_obs=obs, mode='test')
                    else:
                        action = agent.validate(obs=stacked_imgs, state=stacked_states, mode='test')
                    action = action.cpu().numpy()
                    # NOTE For new version, uncomment below!
                    real_action = apply_IK_get_real_action(action, env, env.robot.get_qpos(), use_visual_obs=use_visual_obs)
                    
                # next_obs, reward, done, _ = env.step(action)
                # NOTE For new version, uncomment below!
                next_obs, reward, done, _ = env.step(real_action)
                # TODO: Check how this new action should be used in PAD!
                if args['adapt']:
                    action = torch.from_numpy(action).to(device)
                    next_rgb_imgs, next_robot_states, stacked_next_imgs, stacked_next_states = bc_dataset.get_stacked_data_from_obs(rgb_imgs=next_rgb_imgs, robot_states=next_robot_states, obs=next_obs, i=i, concatenate=False)
                    agent.train(train_visual_encoder=True, train_state_encoder=True, train_policy=False, train_inv=True)
                    agent.update_inv(h=stacked_imgs, s=stacked_states, next_h=stacked_next_imgs, next_s=stacked_next_states, action=action)

                # env.render()

                obs = deepcopy(next_obs)
            
            video = (np.stack(video) * 255).astype(np.uint8)
            video_path = args["weight_path"].replace(".pt", f"_{eval_idx}.mp4")
            imageio.mimsave(video_path, video, fps=120)
            eval_idx += 1


if __name__ == '__main__':
    '''
    - If evaluating the model trained on a single demo, make sure to set 'random_obj_pos': False
    - If evaluating the model trained with states, make sure to set 'use_visual_obs': False
    - Make sure the 'robot_name' is the same as the one in player. If no retargeting used during demo replay, it should be the same as meta_data['robot_name']
    - If 'use_visual_obs': True, make sure the 'model_backbone' is the same as the one in player,.
    '''
    args = {
        'model_dir' : 'policy_only_xarm_less_random_pick_place_mustard_bottle_700',
        "dataset_folder": "sim/baked_data/xarm_less_random_pick_place_mustard_bottle",
        "weight_path": "trained_models/policy_only_xarm_less_random_pick_place_mustard_bottle_1400/bc_model.pt",
        "backbone_type": "MoCo50",
        'robot_name': 'xarm6_allegro_modified_finger',
        'use_visual_obs': True,
        'adapt': False,
        'random_obj_pos' : True,
        'model_backbone': 'MoCo50', 
    }
    main(args)

