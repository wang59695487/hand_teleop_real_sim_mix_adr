import glob
import os
import pickle
from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Optional

import imageio
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import transforms3d
import wandb
from numpy import random

from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.dclaw_env import DClawRLEnv
from hand_teleop.player.vec_player import VecPlayer
from hand_teleop.player.player import PickPlaceEnvPlayer, DcLawEnvPlayer
from hand_teleop.real_world import lab
from hand_teleop.render.render_player import RenderPlayer
from losses import ACTLoss
from main.eval import apply_IK_get_real_action
from model import Agent


class VecTrainer:

    def __init__(self, args):
        self.args = args

        self.epoch_start = 0

        self.load_data(args)
        self.img_preprocess = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.init_model(args)
        self.criterion = ACTLoss(args.w_kl_loss)
        if args.dann:
            self.domain_criterion = nn.BCEWithLogitsLoss()
        if args.finetune_backbone:
            self.optimizer = AdamW(self.model.parameters(), args.max_lr,
                weight_decay=args.wd_coef)
        else:
            params = []
            for name, p in self.model.named_parameters():
                if "backbone" not in name:
                    params.append({"params": p})
            self.optimizer = AdamW(params, args.max_lr,
                weight_decay=args.wd_coef)

        if self.args.max_lr > self.args.min_lr:
            self.scheduler = CosineAnnealingLR(self.optimizer,
                self.args.epochs * len(self.demo_paths_train) // self.args.grad_acc,
                self.args.min_lr)

        self.start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.args.ckpt is None:
            self.log_dir = f"logs/{self.args.task}_{self.start_time}"
            if not self.args.debug:
                os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = os.path.dirname(self.args.ckpt)

        if not args.wandb_off:
            wandb.init(
                project="hand-teleop-vector",
                name=os.path.basename(self.log_dir),
                config=self.args
            )

        self.num_render_workers = args.n_renderers
        self.vec_player: Optional[VecPlayer] = None

    def init_model(self, args):
        self.model = Agent(args).to(args.device)

    def load_data(self, args):
        demo_paths_all = sorted(glob.glob(os.path.join(args.demo_folder,
            "*.pickle")))
        if args.small_scale:
            demo_paths = sorted(glob.glob(os.path.join(args.demo_folder,
                "*_1.pickle")))
        elif args.scale < 100:
            demo_indices = [int(os.path.basename(x).split(".")[0].split("_")[-1]) for x in demo_paths_all]
            demo_paths = [path for path, idx in zip(demo_paths_all, demo_indices) if idx <= args.scale]
        elif args.scale == 100:
            demo_paths = demo_paths_all
        else:
            demo_paths = demo_paths_all
        if args.one_demo:
            self.demo_paths_train = [demo_paths[0]]
            self.demo_paths_val = [demo_paths[0]]
        else:
            train_idx, val_idx = train_test_split(list(range(len(demo_paths))),
                test_size=args.val_pct, random_state=args.seed)
            self.demo_paths_train = [demo_paths[i] for i in train_idx]
            self.demo_paths_val = [demo_paths[i] for i in val_idx]

        self.sample_demo = self.load_demo(self.demo_paths_train[0])
        self.meta_data = self.sample_demo["meta_data"]
        self.init_robot_qpos = self.sample_demo["data"][0]["robot_qpos"]\
            [:self.args.robot_dof]

    def load_demo(self, demo_path):
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)

        if isinstance(demo["data"], dict):
            new_data = [{k: demo["data"][k][i] for k in demo["data"].keys()}
                for i in range(len(demo["data"]["simulation"]))]
            demo["data"] = new_data

        return demo

    def init_player(self, demo):
        meta_data = deepcopy(demo["meta_data"])
        robot_name = self.args.robot
        data = demo["data"]
        use_visual_obs = True
        if "finger_control_params" in meta_data.keys():
            finger_control_params = meta_data["finger_control_params"]
        if "root_rotation_control_params" in meta_data.keys():
            root_rotation_control_params = meta_data["root_rotation_control_params"]
        if "root_translation_control_params" in meta_data.keys():
            root_translation_control_params = meta_data["root_translation_control_params"]
        if "robot_arm_control_params" in meta_data.keys():
            robot_arm_control_params = meta_data["robot_arm_control_params"]            

        # Create env
        env_params = meta_data["env_kwargs"]
        if "task_name" in env_params:
            env_params.pop("task_name")
        env_params["robot_name"] = robot_name
        env_params["use_visual_obs"] = use_visual_obs
        env_params["use_gui"] = False

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

        if self.args.task == "pick_place":
            env = PickPlaceRLEnv(**env_params)
        elif self.args.task == "dclaw":
            env = DClawRLEnv(**env_params)
        else:
            raise NotImplementedError

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
            "relocate_view": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT,
            fov=lab.fov, resolution=(224, 224))
        }
        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info}}
        env.setup_visual_obs_config(camera_info)

        # Player
        if self.args.task == "pick_place":
            player = PickPlaceEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        elif self.args.task == "dclaw":
            player = DcLawEnvPlayer(meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"])
        else:
            raise NotImplementedError

        return player

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model"])
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not ckpt_path.endswith("best.pth"):
            self.epoch_start = int(os.path.basename(ckpt_path) \
                .split(".")[0].split("_")[1]) - 1
        self.log_dir = os.path.dirname(ckpt_path)

    def save_checkpoint(self, epoch):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.log_dir, f"model_{epoch}.pth")
        torch.save(state_dict, save_path)

    def generate_random_object_pose(self, randomness_scale=1):
        random.seed(self.args.seed)
        pos_x = random.uniform(-0.1, 0.1) * randomness_scale
        pos_y = random.uniform(0.2, 0.3) * randomness_scale
        position = np.array([pos_x, pos_y, 0.1])
        # euler = self.np_random.uniform(low=np.deg2rad(15), high=np.deg2rad(25))
        if self.object_name != "sugar_box":
            euler = random.uniform(np.deg2rad(15), np.deg2rad(25))
        else:
            euler = random.uniform(np.deg2rad(80), np.deg2rad(90))
        orientation = transforms3d.euler.euler2quat(0, 0, euler)
        random_pose = sapien.Pose(position, orientation)
        return random_pose

    @torch.no_grad()
    def eval_in_env(self, epoch, x_steps, y_steps):
        self.model.eval()

        robot_name = self.args.robot
        task_name = self.args.task
        rotation_reward_weight = 0
        use_visual_obs = True
        if 'finger_control_params' in self.meta_data.keys():
            finger_control_params = self.meta_data['finger_control_params']
        if 'root_rotation_control_params' in self.meta_data.keys():
            root_rotation_control_params = self.meta_data['root_rotation_control_params']
        if 'root_translation_control_params' in self.meta_data.keys():
            root_translation_control_params = self.meta_data['root_translation_control_params']
        if 'robot_arm_control_params' in self.meta_data.keys():
            robot_arm_control_params = self.meta_data['robot_arm_control_params']            

        env_params = self.meta_data["env_kwargs"]
        env_params['robot_name'] = robot_name
        env_params['use_visual_obs'] = True
        env_params['use_gui'] = False
        env_params['light_mode'] = "default" if self.args.eval_rnd_lvl < 3 else "random"

        env_params["device"] = "cuda"

        if 'init_obj_pos' in self.meta_data["env_kwargs"].keys():
            print('Found initial object pose')
            env_params['init_obj_pos'] = self.meta_data["env_kwargs"]['init_obj_pos']
            object_pos = self.meta_data["env_kwargs"]['init_obj_pos']

        if 'init_target_pos' in self.meta_data["env_kwargs"].keys():
            print('Found initial target pose')
            env_params['init_target_pos'] = self.meta_data["env_kwargs"]['init_target_pos']
            target_pos = self.meta_data["env_kwargs"]['init_target_pos']

        if task_name == 'pick_place':
            env = PickPlaceRLEnv(**env_params)
        elif task_name == 'dclaw':
            env = DClawRLEnv(**env_params)
        else:
            raise NotImplementedError
        env.seed(0)
        env.reset()

        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for joint in env.robot.get_active_joints():
            name = joint.get_name()
            if name in arm_joint_names:
                joint.set_drive_property(*(1 * robot_arm_control_params), mode="force")
            else:
                joint.set_drive_property(*(1 * finger_control_params), mode="force")
        env.rl_step = env.simple_sim_step

        real_camera_cfg = {
            "relocate_view": dict( pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=lab.fov, resolution=(224, 224))
        }

        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)

        if task_name == "pick_place":
            init_robot_qpos = [0, (-45/180)*np.pi, 0, 0, (45/180)*np.pi, (-90/180)*np.pi] + [0] * 16
        elif task_name == "dclaw":
            init_robot_qpos = [0, (20/180)*np.pi, -(85/180)*np.pi, 0, (112/180)*np.pi, -np.pi / 2] + [0] * 16

        env.robot.set_qpos(init_robot_qpos)

        eval_idx = 0
        avg_success = 0
        progress = tqdm(total=x_steps * y_steps)

        # since in simulation, we always use simulated data, so sim_real_label is always 0
        sim_real_label = [0]
        var_object = 0 if self.args.eval_rnd_lvl < 4 else 0.1
        for x in np.linspace(-0.1 - var_object, 0.1 + var_object, x_steps):        # -0.08 0.08 /// -0.05 0
            for y in np.linspace(0.2 - var_object, 0.3 + var_object, y_steps):  # 0.12 0.18 /// 0.12 0.32
                video = []
                object_p = np.array([x, y, 0.1])
                object_pos = sapien.Pose(p=object_p, q=np.array([1, 0, 0, 0]))
                print('Object Pos: {}'.format(object_pos))

                env.reset()

                ########### Add Plate Randomness ############
                if self.args.eval_rnd_lvl in [2,3,6] and task_name == "pick_place":
                    ########### Randomize the plate pose ############
                    var_plate = 0.05 if self.args.eval_rnd_lvl in [2] else 0.1
                    print("############################Randomize the plate pose##################")
                    x2 = np.random.uniform(-var_plate, var_plate)
                    y2 = np.random.uniform(-var_plate, var_plate)
                    plate_random_plate = sapien.Pose([-0.005+x2, -0.12+y2, 0],[1,0,0,0]) 
                    env.plate.set_pose(plate_random_plate)
                    print('Target Pos: {}'.format(plate_random_plate))
                else:
                    env.plate.set_pose(sapien.Pose([-0.005, -0.12, 0],[1,0,0,0]))

                ############## Add Texture Randomness ############
                if self.args.eval_rnd_lvl in [4,6] :
                    #env.random_light(self.args.rnd_lvl-2)
                    env.generate_random_object_texture(2)
                
                ############## Add Light Randomness ############
                if self.args.eval_rnd_lvl in [5,6] :
                    env.random_light(2)

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
                success = False
                all_time_actions = torch.zeros([self.args.max_eval_steps,
                    self.args.max_eval_steps + self.args.n_queries, self.args.action_dims]).to(self.args.device)
                for i in range(self.args.max_eval_steps):
                    video.append(obs["relocate_view-rgb"].cpu().detach().numpy())

                    image = torch.moveaxis(obs["relocate_view-rgb"], -1, 0)[None, ...]
                    robot_qpos = np.concatenate([env.robot.get_qpos(),env.ee_link.get_pose().p,env.ee_link.get_pose().q])
                    robot_qpos = torch.from_numpy(robot_qpos[None, ...]).to(self.args.device)

                    action = self.model.get_action(image, robot_qpos)
                    all_time_actions[[i], i:i + self.args.n_queries] = action
                    actions_for_curr_step = all_time_actions[:, i]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = self.args.w_action_ema
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    raw_action = raw_action.cpu().detach().numpy()
                    real_action = apply_IK_get_real_action(raw_action, env, env.robot.get_qpos(), use_visual_obs=use_visual_obs)

                    next_obs, reward, done, info = env.step(real_action)
                    if epoch != "best":
                        if task_name == "pick_place":
                            info_success = info["is_object_lifted"] and info["success"]
                        elif task_name == "dclaw":
                            info_success = info["success"]
                    else:
                        info_success = info["success"]
                    
                    success = success or info_success
                    if success:
                        break

                    obs = deepcopy(next_obs)
                
                #If it did not lift the object, consider it as 0.25 success
                if epoch != "best" and info["success"]:
                    avg_success += 0.25
                avg_success += int(success)
                video = (np.stack(video) * 255).astype(np.uint8)

                #only save video if success or in the final_success evaluation
                #if success or epoch == "best":
                if task_name == "pick_place":
                    is_lifted = info["is_object_lifted"]
                    video_path = os.path.join(self.log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{is_lifted}_{(x, y)}_level{self.args.eval_rnd_lvl}.mp4")
                elif task_name == "dclaw":
                    total_angle = info["object_total_rotate_angle"]
                    video_path = os.path.join(self.log_dir, f"epoch_{epoch}_{eval_idx}_{success}_{total_angle}_level{self.args.eval_rnd_lvl}.mp4")
                if not self.args.debug:
                    imageio.mimsave(video_path, video, fps=120)
                eval_idx += 1
                progress.update()

        avg_success /= eval_idx
        metrics = {
            "avg_success": avg_success
        }
        progress.close()

        return metrics

    def render_images(self, demo, indices):
        self.vec_player.load_player_data([demo] * self.num_render_workers)

        data_len = len(indices)

        image_tensor = []
        for i_worker in range(data_len // self.num_render_workers):
            beg = i_worker * self.num_render_workers
            end = (i_worker + 1) * self.num_render_workers
            batch_indices = indices[beg:end]

            self.vec_player.set_sim_data_async(batch_indices)
            self.vec_player.set_sim_data_wait()
            self.vec_player.render_async()
            image_dict = self.vec_player.render_wait()
            images = image_dict["Color"].contiguous()
            images = images[:, 0, :, :, :3].permute((0, 3, 1, 2)).clone()

            image_tensor.append(images)

        image_tensor = torch.cat(image_tensor).detach()

        return image_tensor

    def get_obj_poses(self, demo, indices):
        obj_poses = torch.tensor([demo["data"][idx]["simulation"]["actor"][4][:7]
            for idx in indices]).to(self.args.device)

        return obj_poses

    def _train_epoch(self):
        loss_dict_train = {}
        batch_cnt = 0

        self.model.train()
        if not self.args.finetune_backbone:            
            self.model.policy_net.model.backbones.eval()

        for i in tqdm(range(len(self.demo_paths_train))):
            cur_demo = self.load_demo(self.demo_paths_train[i])
            demo_len = len(cur_demo["data"])
            if demo_len <= self.args.min_demo_len:
                continue

            begs = np.random.choice(demo_len - self.args.n_queries + 1,
                self.args.batch_size)
            ends = begs + self.args.n_queries

            image_tensor = self.render_images(cur_demo, begs)
            image_tensor = self.img_preprocess(image_tensor)

            robot_qpos_tensor = torch.from_numpy(np.stack([
                cur_demo["data"][t]["robot_qpos"] for t in begs]))
            action_tensor = torch.from_numpy(np.stack([
                np.stack([x["action"] for x in cur_demo["data"][b:e]])
                for b, e in zip(begs, ends)]))
            robot_qpos_tensor = robot_qpos_tensor.to(self.args.device)
            action_tensor = action_tensor.to(self.args.device)
            is_pad = torch.zeros(action_tensor.size()[:-1],
                dtype=torch.bool).to(self.args.device)
            if self.args.rnd_len:
                rnd_ends = np.random.randint(self.args.min_eps_len,
                    self.args.n_queries, size=self.args.batch_size)
                for i in range(self.args.batch_size):
                    action_tensor[i, rnd_ends[i]:] = 0
                    is_pad[i, rnd_ends[i]:] = True

            if self.args.dann:
                actions_pred, mu, log_var, domain_preds = self.model(image_tensor,
                    robot_qpos_tensor, action_tensor, is_pad)
                # TODO: mix real and sim and get domain labels
                loss_dict = self.criterion(actions_pred, action_tensor, is_pad, mu, log_var)
                domains = torch.zeros(actions_pred.size(0), 1, dtype=torch.float,
                    device=self.args.device)
                domain_loss = self.domain_criterion(domain_preds, domains)
                loss_dict["domain_loss"] = domain_loss
                loss_dict["loss"] += loss_dict["domain_loss"]
            else:
                actions_pred, mu, log_var = self.model(image_tensor,
                    robot_qpos_tensor, action_tensor, is_pad)
                loss_dict = self.criterion(actions_pred, action_tensor, is_pad, mu, log_var)

            loss_dict["loss"].backward()
            for k in loss_dict.keys():
                loss_dict_train[f"{k}/train"] = loss_dict_train.get(f"{k}/train", 0) + loss_dict[k].detach().cpu().item()
            batch_cnt += 1

            if not self.args.wandb_off:
                wandb.log({
                    "running_loss": loss_dict["loss"].detach().cpu().item(),
                    "running_kl_div": loss_dict["kl_div"].detach().cpu().item(),
                    "running_action_loss": loss_dict["action_loss"].detach().cpu().item(),
                })

            # gradient accumulation check
            if (i + 1) % self.args.grad_acc == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if hasattr(self, "scheduler"):
                    self.scheduler.step()

            if i >= 10 and self.args.debug:
                break

        for k in loss_dict_train.keys():
            loss_dict_train[k] /= batch_cnt

        return loss_dict_train

    @torch.no_grad()
    def _eval_epoch(self):
        loss_dict_val = {}
        batch_cnt = 0

        self.model.eval()

        for i in tqdm(range(len(self.demo_paths_val))):
            cur_demo = self.load_demo(self.demo_paths_val[i])
            demo_len = len(cur_demo["data"])
            if demo_len <= self.args.min_demo_len:
                continue
            begs = np.arange(0, demo_len - self.args.n_queries,
                self.args.n_queries)
            if begs.shape[0] % self.args.n_renderers != 0:
                residual = self.args.n_renderers - begs.shape[0] % self.args.n_renderers
                begs = np.concatenate((begs, begs[:residual]))
            ends = begs + self.args.n_queries

            image_tensor = self.render_images(cur_demo, begs)
            image_tensor = self.img_preprocess(image_tensor)

            robot_qpos_tensor = torch.from_numpy(np.stack([
                cur_demo["data"][t]["robot_qpos"] for t in begs]))
            action_tensor = torch.from_numpy(np.stack([
                np.stack([x["action"] for x in cur_demo["data"][b:e]])
                for b, e in zip(begs, ends)]))
            robot_qpos_tensor = robot_qpos_tensor.to(self.args.device)
            action_tensor = action_tensor.to(self.args.device)
            is_pad = torch.zeros(action_tensor.size()[:-1],
                dtype=torch.bool).to(self.args.device)

            if self.args.dann:
                actions_pred, mu, log_var, domain_preds = self.model(image_tensor,
                    robot_qpos_tensor, action_tensor, is_pad)
                # TODO: mix real and sim and get domain labels
                loss_dict = self.criterion(actions_pred, action_tensor, is_pad, mu, log_var)
                domains = torch.zeros(actions_pred.size(0), 1, dtype=torch.float,
                    device=self.args.device)
                domain_loss = self.domain_criterion(domain_preds, domains)
                loss_dict["domain_loss"] = domain_loss
                loss_dict["loss"] += loss_dict["domain_loss"]
            else:
                actions_pred, mu, log_var = self.model(image_tensor,
                    robot_qpos_tensor, action_tensor, is_pad)
                loss_dict = self.criterion(actions_pred, action_tensor, is_pad, mu, log_var)

            for k in loss_dict.keys():
                loss_dict_val[f"{k}/val"] = loss_dict_val.get(f"{k}/val", 0) + loss_dict[k].detach().cpu().item()
            batch_cnt += 1

            if i >= 10 and self.args.debug:
                break

        for k in loss_dict_val.keys():
            loss_dict_val[k] /= batch_cnt

        return loss_dict_val

    def train(self):
        best_success = 0

        player_create_fn = [partial(RenderPlayer.from_demo, demo=self.sample_demo,
            robot_name=self.args.robot) for i in range(self.num_render_workers)]
        self.vec_player = VecPlayer(player_create_fn)

        for i in range(self.epoch_start, self.args.epochs):
            metrics_train = self._train_epoch()
            metrics_val = self._eval_epoch()
            metrics = {"epoch": i + 1}
            metrics.update(metrics_train)
            metrics.update(metrics_val)

            if not self.args.debug:
                self.save_checkpoint("latest")

            if (i + 1) % self.args.eval_freq == 0\
                    and (i + 1) >= self.args.eval_beg:
                if not self.args.debug:
                    self.save_checkpoint(i + 1)
                env_metrics = self.eval_in_env(i + 1,
                    self.args.eval_x_steps, self.args.eval_y_steps)
                # env_metrics = self.vector_eval_in_env(i + 1,
                #     self.args.eval_x_steps, self.args.eval_y_steps)
                metrics.update(env_metrics)

                if metrics["avg_success"] > best_success:
                    if not self.args.debug:
                        self.save_checkpoint("best")
                    best_success = metrics["avg_success"]

            if not self.args.wandb_off:
                wandb.log(metrics)


if __name__ == "__main__":
    with open("real/pick_place_mustard_bottle_large_scale/0000.pkl", "rb") as f:
        data = pickle.load(f)
    print(1)