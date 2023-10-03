import numpy as np
import torch
import pickle
import os
import imageio
import sapien.core as sapien

from behavior_cloning import BehaviorCloning, BehaviorCloningMultiModal
from feature_extractor import ImageEncoder, StateDecoder, generate_feature_extraction_model
from sapien.utils import Viewer

from hand_teleop.env.rl_env.base import BaseRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.relocate_env import RelocateRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv
from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.real_world import task_setting
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.player.player import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_bc(args):
    DATAPATH = './sim/baked_data/{}.pickle'.format(args['dataset_name'])
    model_path = './trained_models/{}_{}.pt'.format(args['task_props'], args['model_backbone'])
    file = open(DATAPATH, 'rb')
    baked_data = pickle.load(file)
    init_robot_qpos = baked_data["obs"][0][-58:-7]
    meta_data = baked_data['meta_data']
    task_name = args['task_name']
    
    # robot_name = "allegro_hand_free"
    # path = './test_baked.pickle'
    # file = open(path, 'rb')
    # baked_data = pickle.load(file)
    # init_robot_qpos = baked_data["obs"][0][:22]
    # init_object_qpos = baked_data["obs"][0][-5]

    robot_name = "mano"
    # path = "./test_visual.pickle"
    # all_data = np.load(path, allow_pickle=True)
    # meta_data = all_data["meta_data"]
    # meta_data["env_kwargs"].pop('task_name')
    # data = all_data["data"]

    # Create env
    env_params = meta_data["env_kwargs"]
    env_params['robot_name'] = robot_name
    env_params['use_visual_obs'] = args['use_visual_obs']
    env_params['use_gui'] = True
    # env_params = dict(object_name=meta_data['env_kwargs']['object_name'], object_scale=meta_data['env_kwargs']['object_scale'], robot_name=robot_name,
    #                  rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=randomness_scale, use_visual_obs=use_visual_obs, use_gui=True)
    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    if task_name == 'pick_place':
        object_names = ['tomato_soup_can', "mustard_bottle", "potted_meat_can"]
        env_params['object_name'] = object_names[np.random.choice(len(object_names))]
        env_params['object_name'] = object_names[2]
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

    # Set the player (if needed)
    # data = baked_data['data']
    # if task_name == 'pick_place':
    #     player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # elif task_name == 'hammer':
    #     player = HammerEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # elif task_name == 'table_door':
    #     player = TableDoorEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # elif task_name == 'insert_object':
    #     player = InsertObjectEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # elif task_name == 'mug_flip':
    #     player = FlipMugEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # else:
    #     raise NotImplementedError
    # _ = player.bake_demonstration()
    # player.scene.remove_articulation(player.human_robot_hand.robot)

    viewer = env.render(mode="human")
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, 0)

    if args['use_visual_obs']:
        # Create camera
        # camera_cfg = {
        #     "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
        #                             right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
        # }
        camera_cfg = {
            "relocate_view": dict(position=np.array([0.25, 0.25, 0.35]), look_at_dir=np.array([-0.25, -0.25, -0.35]),
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
        viewer.toggle_axes(False)
        viewer.toggle_camera_lines(False)

        # Generate the models for inference for visual obs
        feature_extractor = generate_feature_extraction_model(backbone_type=args['model_backbone'])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
        if args['stack_frames']:
            if args['model_backbone'] == 'ResNet34' or args['model_backbone'] == 'MoCo18':
                in_dim = 2280
            else:
                in_dim = 8424
        else:
            if args['encode']:
                in_dim = 2280
            else:
                in_dim = 2106
    else:
        in_dim = 63
    model = BehaviorCloning(in_dim, 51)
    best_checkpoint = torch.load(model_path)
    print('Loaded Model: {}'.format(model_path.split('/')[-1]))
    model.load_state_dict(best_checkpoint['model'])
    model = model.to(device)
    model.eval()

    if args['encode']:
        image_encoder = ImageEncoder()
        best_checkpoint = torch.load('./trained_models/trained_image_encoder.pt')
        image_encoder.load_state_dict(best_checkpoint['model'])
        image_encoder = image_encoder.to(device)
        image_encoder.eval()
        state_decoder = StateDecoder()
        best_checkpoint = torch.load('./trained_models/trained_state_decoder.pt')
        state_decoder.load_state_dict(best_checkpoint['model'])
        state_decoder = state_decoder.to(device)
        state_decoder.eval()

    done = False
    manual_action = False
    action = np.zeros(env.robot.dof)

    env.robot.set_qpos(init_robot_qpos)
    plate_poses = np.array([[-0.1, -0.15, 0.1],[-0.1, -0.2, 0.1],[0.1, -0.15, 0.1],[0.1, -0.2, 0.1]]) #[0, -0.18, 0.1]
    plate_poses = [np.array([0.05, -0.2, 0.1])]

    trial = 0
    while not viewer.closed:
        for plate_pos in plate_poses:
            for x in np.arange(-0.1, 0.0, 0.02): # -0.15, 0.18, 0.03  # -0.1, 0.0, 0.02
                for y in np.arange(0.1, 0.2, 0.02): # 0.05, 0.2, 0.05 # 0.1, 0.2, 0.02
                    with torch.no_grad():
                        random_object_pos = np.array([x, y, 0.1])
                        print('Object Pose: {}'.format(random_object_pos))
                        print('Plate Pose: {}'.format(plate_pos))            
                        reward_sum = 0
                        env.reset()
                        env.manipulated_object.set_pose(sapien.Pose(random_object_pos))
                        env.plate.set_pose(sapien.Pose(plate_pos))
                        # env.randomize_object_rotation()
                        for _ in range(10*env.frame_skip):
                            env.scene.step()
                        env.robot.set_qpos(init_robot_qpos)
                        obs = env.get_observation()
                        features = []
                        robot_states = []
                        for i in range(400):
                            if args['use_visual_obs']:
                                robot_state = obs["state"]
                                img = obs["relocate_view-rgb"]
                                img = np.moveaxis(img, -1, 0)
                                img = torch.from_numpy(img)
                                img = img.reshape(1, 3, 224, 224)
                                img = img.to(device)
                                with torch.no_grad():
                                    feature = feature_extractor(img)
                                feature = feature.cpu().detach().numpy().reshape(-1)

                                # Save the frames - for debugging
                                if args['save_frames']:
                                    rgb = img.cpu().detach().numpy()
                                    rgb = np.squeeze(rgb)
                                    rgb = np.moveaxis(rgb,0,-1)
                                    rgb_pic = (rgb * 255).astype(np.uint8)
                                    imageio.imsave("./temp/eval/relocate-rgb_{}.png".format(i), rgb_pic)

                                if args['encode']:
                                    feature = torch.from_numpy(feature).to(device)
                                    feature = image_encoder.encode(feature)
                                    feature = feature.cpu().detach().numpy()
                                    robot_state = torch.from_numpy(robot_state).to(device)
                                    robot_state = state_decoder.decode(robot_state)
                                    robot_state = robot_state.cpu().detach().numpy()
                                    
                                features.append(feature)
                                robot_states.append(robot_state)
                                if args['stack_frames']:
                                    if i == 0:
                                        obs = np.concatenate((features[i],features[i],features[i],features[i],robot_states[i],robot_states[i],robot_states[i],robot_states[i]))
                                    elif i == 1:
                                        obs = np.concatenate((features[i-1],features[i],features[i],features[i],robot_states[i-1],robot_states[i],robot_states[i],robot_states[i]))
                                    elif i == 2:
                                        obs = np.concatenate((features[i-2],features[i-1],features[i],features[i],robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i]))
                                    else:
                                        obs = np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i]))
                                else:
                                        obs = np.concatenate((feature, robot_state))
                            if manual_action:
                                action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
                            else:
                                obs = obs.reshape((1,-1))
                                obs = torch.from_numpy(obs).to(device).float()
                                action = model(obs)
                                action = action.cpu().numpy()
                            obs, reward, done, _ = env.step(action)
                            reward_sum += reward
                            env.render()
                            if env.viewer.window.key_down("enter"):
                                manual_action = True
                            elif env.viewer.window.key_down("p"):
                                manual_action = False
                    print(f"Reward: {reward_sum}")
                    trial +=1
        break

        # with torch.no_grad():
        #     reward_sum = 0
        #     env.reset()
        #     for _ in range(10*env.frame_skip):
        #         env.scene.step()
        #     env.robot.set_qpos(init_robot_qpos)
        #     obs = env.get_observation()
        #     features = []
        #     robot_states = []
        #     for i in range(400):
        #         if args['use_visual_obs']:
        #             robot_state = obs["state"]
        #             img = obs["relocate_view-rgb"]
        #             img = np.moveaxis(img, -1, 0)
        #             img = torch.from_numpy(img)
        #             img = img.reshape(1, 3, 224, 224)
        #             img = img.to(device)
        #             with torch.no_grad():
        #                 feature = feature_extractor(img)
        #             feature = feature.cpu().detach().numpy().reshape(-1)

        #             # Save the frames - for debugging
        #             # rgb = img.cpu().detach().numpy()
        #             # rgb = np.squeeze(rgb)
        #             # rgb = np.moveaxis(rgb,0,-1)
        #             # rgb_pic = (rgb * 255).astype(np.uint8)
        #             # imageio.imsave("./temp/eval/relocate-rgb_{}.png".format(i), rgb_pic)

        #             if args['encode']:
        #                 feature = torch.from_numpy(feature).to(device)
        #                 feature = image_encoder.encode(feature)
        #                 feature = feature.cpu().detach().numpy()
        #                 robot_state = torch.from_numpy(robot_state).to(device)
        #                 robot_state = state_decoder.decode(robot_state)
        #                 robot_state = robot_state.cpu().detach().numpy()
                        
        #             features.append(feature)
        #             robot_states.append(robot_state)
        #             if args['stack_frames']:
        #                 if i == 0:
        #                     obs = np.concatenate((features[i],features[i],features[i],features[i],robot_states[i],robot_states[i],robot_states[i],robot_states[i]))
        #                 elif i == 1:
        #                     obs = np.concatenate((features[i-1],features[i],features[i],features[i],robot_states[i-1],robot_states[i],robot_states[i],robot_states[i]))
        #                 elif i == 2:
        #                     obs = np.concatenate((features[i-2],features[i-1],features[i],features[i],robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i]))
        #                 else:
        #                     obs = np.concatenate((features[i-3],features[i-2],features[i-1],features[i],robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i]))
        #             else:
        #                     obs = np.concatenate((feature, robot_state))
        #         if manual_action:
        #             action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
        #         else:
        #             obs = obs.reshape((1,-1))
        #             obs = torch.from_numpy(obs).to(device).float()
        #             action = model(obs)
        #             action = action.cpu().numpy()
        #         obs, reward, done, _ = env.step(action)
        #         reward_sum += reward
        #         env.render()
        #         if env.viewer.window.key_down("enter"):
        #             manual_action = True
        #         elif env.viewer.window.key_down("p"):
        #             manual_action = False
        # print(f"Reward: {reward_sum}")
        # trial +=1

    print("Number of Trials: {}".format(trial))

if __name__ == '__main__':
    args = {
        'task_name': 'pick_place',
        'dataset_name': 'dataset_lessrandom_pick_place',
        'task_props': 'lessrandom_pick_place',
        'model_backbone': 'MoCo50',
        'use_visual_obs': True,
        'stack_frames': True,
        'encode': False,
        'save_frames': False
    }
    eval_bc(args)
