import numpy as np
import torch
import pickle
import os
import imageio
import sapien.core as sapien

from main.behavior_cloning import BehaviorCloning
from main.train_from_scratch_bc import EmbeddingNet
from sapien.utils import Viewer

from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pen_draw_env import PenDrawRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.real_world import task_setting
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.player.player import LaptopEnvPlayer, PickPlaceEnvPlayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    dataset_path = './sim/baked_data/{}.pickle'.format(args['dataset_name'])
    model_path = 'trained_models/{}.pt'.format(args['model_name'])
    with open(dataset_path,'rb') as file:
        baked_data = pickle.load(file)
    init_robot_qpos = baked_data["state"][0][:51]
    meta_data = baked_data['meta_data']

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
    rotation_reward_weight = 0
    randomness_scale = 1
    env_params = dict(object_name=meta_data['env_kwargs']['object_name'], object_scale=meta_data['env_kwargs']['object_scale'], robot_name=robot_name,
                     rotation_reward_weight=rotation_reward_weight, constant_object_state=False, randomness_scale=randomness_scale, use_visual_obs=True, use_gui=True)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
        env = PickPlaceRLEnv(**env_params)
    else:
        env = LaptopRLEnv(use_gui=True, robot_name=robot_name)
    env.seed(0)
    env.reset()

    # Set the player (if needed)
    # player = LaptopEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=meta_data["zero_joint_pos"])
    # _ = player.bake_demonstration()
    # player.scene.remove_articulation(player.human_robot_hand.robot)

    viewer = env.render(mode="human")
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer
    # viewer.set_camera_xyz(0.4, 0.2, 0.5)
    # viewer.set_camera_rpy(0, -np.pi/4, 5*np.pi/6)
    viewer.set_camera_xyz(-0.6, 0, 0.6)
    viewer.set_camera_rpy(0, -np.pi/6, 0)

    # Create camera
    # camera_cfg = {
    #     "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
    #                             right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
    # }
    camera_cfg = {
        "relocate_view": dict(position=np.array([0.25, 0.25, 0.35]), look_at_dir=np.array([-0.25, -0.25, -0.35]),
                                right_dir=np.array([-1, 1, 0]), fov=np.deg2rad(69.4), resolution=(224, 224))
    }
    env.setup_camera_from_config(camera_cfg)

    # Specify modality
    empty_info = {}  # level empty dict for now, reserved for future
    camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
    env.setup_visual_obs_config(camera_info)
    viewer.toggle_axes(False)
    viewer.toggle_camera_lines(False)

    # Load models
    embedding_model = EmbeddingNet(args['embedding_name'],
                                   in_channels=3,
                                   pretrained=True,
                                   train=False)
    actor_model = BehaviorCloning(3432, 51)
    checkpoint = torch.load(model_path)
    embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
    embedding_model = embedding_model.to(device)
    embedding_model.eval()
    actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
    actor_model = actor_model.to(device)
    actor_model.eval()

    done = False
    manual_action = False
    action = np.zeros(51)
    # action = np.zeros(22)

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
                        rgb_imgs = []
                        for i in range(400):
                            robot_states.append(obs["state"])
                            rgb_imgs.append(obs["relocate_view-rgb"])

                            # Save the frames - for debugging
                            # rgb_pic = (obs["relocate_view-rgb"] * 255).astype(np.uint8)
                            # imageio.imsave("./temp/eval/relocate-rgb_{}.png".format(i), rgb_pic)

                            if i==0:
                                stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
                                stacked_states = np.concatenate((robot_states[i],robot_states[i],robot_states[i],robot_states[i]))        
                            elif i==1:
                                stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
                                stacked_states = np.concatenate((robot_states[i-1],robot_states[i],robot_states[i],robot_states[i]))         
                            elif i==2:
                                stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i],rgb_imgs[i]), axis=-1), -1, 0)
                                stacked_states = np.concatenate((robot_states[i-2],robot_states[i-1],robot_states[i],robot_states[i]))       
                            else:
                                stacked_imgs = np.moveaxis(np.concatenate((rgb_imgs[i-3],rgb_imgs[i-2],rgb_imgs[i-1],rgb_imgs[i]), axis=-1), -1, 0)       
                                stacked_states = np.concatenate((robot_states[i-3],robot_states[i-2],robot_states[i-1],robot_states[i]))        

                            stacked_imgs = torch.from_numpy(stacked_imgs).to(device)
                            stacked_imgs = torch.unsqueeze(stacked_imgs, 0)
                            stacked_states = torch.from_numpy(stacked_states).to(device)
                            stacked_states = torch.unsqueeze(stacked_states, 0)

                            embedding_batch = []
                            for frame_id in range(args['frame_stack']):
                                embedding_i = embedding_model(stacked_imgs[:, frame_id*3:(frame_id+1)*3].permute(0, 2, 3, 1)) # obs: bxfsx256x256 -> bx256x256xfs
                                embedding_batch.append(embedding_i)
                            embedding_batch = torch.from_numpy(np.array(embedding_batch)).to(device)
                            embedding_batch = embedding_batch.view(1, -1) # concat frames
                            final_obs = torch.cat((embedding_batch,stacked_states), dim=1)

                            if manual_action:
                                action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
                            else:
                                # final_obs = final_obs.reshape((1,-1))
                                final_obs = final_obs.to(device).float()
                                action = actor_model(final_obs)
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
    print("Number of Trials: {}".format(trial))    

if __name__ == '__main__':
    args = {
        'model_name' : 'train_from_scratch_trial1200',
        'dataset_name' : 'dataset_for_train_from_scratch',
        'embedding_name' : 'ours',
        'frame_stack': 4,
    }
    test(args)    