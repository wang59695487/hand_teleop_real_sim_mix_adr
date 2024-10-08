import numpy as np
import torch
import os
import pickle
import h5py
import random
from copy import deepcopy

import multiprocessing as mp
from functools import partial
from multiprocessing.connection import Connection
from typing import Callable, List, Tuple

import sapien.core as sapien
import tqdm

from hand_teleop.player.randomization_utils import *
from hand_teleop.player.player import *


def apply_IK_get_real_action(action, env, qpos):
    delta_pose = np.squeeze(action)[: env.arm_dof] / 100
    palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
        qpos[: env.arm_dof]
    )
    arm_qvel = compute_inverse_kinematics(
        delta_pose, palm_jacobian)[: env.arm_dof]
    arm_qpos = arm_qvel + qpos[: env.arm_dof]
    hand_qpos = np.squeeze(action)[env.arm_dof:]
    target_qpos = np.concatenate([arm_qpos, hand_qpos])
    return target_qpos


def create_env(args):
    with open("{}/meta_data.pickle".format(args["sim_dataset_folder"]), "rb") as file:
        meta_data = pickle.load(file)

    # --Create Env and Robot-- #
    robot_name = args["robot_name"]
    task_name = meta_data["task_name"]

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
    env_params["use_visual_obs"] = True
    env_params["use_gui"] = False

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]

    if "init_obj_pos" in meta_data["env_kwargs"].keys():
        print("Found initial object pose")
        env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]

    if "init_target_pos" in meta_data["env_kwargs"].keys() and "dclaw" not in task_name:
        print("Found initial target pose")
        env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]

    if "pick_place" in task_name:
        if args['object_name'] == 'diverse_objects':
            bottle_id = np.random.randint(0, 10)
            if bottle_id == 9:
                env_params["object_category"] = "YCB"
                env_params["object_name"] = "mustard_bottle"
            else:
                env_params["object_category"] = "SHAPE_NET"
                env_params["object_name"] = "bottle_{}".format(bottle_id)
        env = PickPlaceRLEnv(**env_params)
    elif "dclaw" in task_name:
        if args['object_name'] == 'diverse_objects':
            dclaw_id = np.random.randint(0, 8)
            env_params["object_name"] = dclaw_diverse_objects[dclaw_id]
        env = DClawRLEnv(**env_params)
    elif "pour" in task_name:
        env = PourBoxRLEnv(**env_params)
    else:
        raise NotImplementedError

    env.seed(0)
    env.reset()

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

    if args["use_visual_obs"]:

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
        ),
        # "record_view": dict(
        #     pose=aug_view_pose*lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(50.4), resolution=(640, 480)
        # )
    }


        env.setup_camera_from_config(real_camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {
            "rgb": empty_info, "segmentation": empty_info}}
        env.setup_visual_obs_config(camera_info)
        # camera_info = {"record_view": {
        # "rgb": empty_info, "segmentation": empty_info}}
        # env.setup_visual_obs_config(camera_info)

    return env


def eval_in_env(
    args, log_dir, epoch, eval_idx, x, y, randomness_rank, policy, avg_success
):
    env = create_env(args)
    with open("{}/meta_data.pickle".format(args["sim_dataset_folder"]), "rb") as file:
        meta_data = pickle.load(file)

    if randomness_rank >= 2:
        env.random_map(2)  # hyper parameter

    ############## Add Texture Randomness ############
    if randomness_rank >= 2:
        env.generate_random_object_texture(2)

    ############## Add Light Randomness ############
    if randomness_rank >= 2:
        env.random_light(2)

    env.reset()

    robot_name = args["robot_name"]

    ################################# Initial robot control parameters#################################
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

    ########### Initialize the Robot Qpose ############
    task_name = meta_data["task_name"]

    if "dclaw" in task_name:
        init_robot_qpos = [
            0,
            (20 / 180) * np.pi,
            -(85 / 180) * np.pi,
            0,
            (112 / 180) * np.pi,
            -np.pi / 2,
        ] + [0] * 16
    else:
        init_robot_qpos = [
            0,
            (-45 / 180) * np.pi,
            0,
            0,
            (45 / 180) * np.pi,
            (-90 / 180) * np.pi,
        ] + [0] * 16
    env.robot.set_qpos(init_robot_qpos)

    ########### Initialize the object pose ############
    idx = np.random.randint(len(meta_data["init_obj_poses"]))
    sampled_pos = meta_data["init_obj_poses"][idx]
    object_p = np.array([x, y, sampled_pos.p[-1]])
    object_pos = (
        sapien.Pose(p=object_p, q=sampled_pos.q)
        # if task_name in ["pick_place", "pour"]
        # else sapien.Pose(p=object_p, q=[0.707, 0, 0, 0.707])
    )
    print("Object Pos: {}".format(object_pos))

    env.manipulated_object.set_pose(object_pos)
    if "pour" in task_name:
        for i in range(len(env.boxes)):
            env.boxes[i].set_pose(object_pos)

    ########### Add target-object Randomness ############
    if "dclaw" in task_name:
        env.object_total_rotate_angle = 0
        env.object_angle = env.get_object_rotate_angle()
    else:
        if randomness_rank > 2:
            ########### Randomize the target-object  ############
            var_target = [0.08, 0.2] if randomness_rank < 4 else [0.16, 0.2]
            print(
                "############################Randomize the target pose##################"
            )
            x2 = np.random.uniform(-var_target[0], var_target[0])
            y2 = np.random.uniform(0, var_target[1])
            if "pick_place" in task_name:
                aug_random_target = sapien.Pose(
                    [-0.005 + x2, -0.1 - y2, 0], [1, 0, 0, 0]
                )
            elif "pour" in task_name:
                aug_random_target = sapien.Pose(
                    [0 + x2, 0.2 + y2, env.bowl_height], [1, 0, 0, 0]
                )
            dist_xy = np.linalg.norm(
                object_pos.p[:2] - aug_random_target.p[:2])
            if dist_xy >= 0.25:
                env.target_object.set_pose(aug_random_target)
            else:
                if "pick_place" in task_name:
                    env.target_object.set_pose(
                        sapien.Pose([-0.005, -0.12, 0], [1, 0, 0, 0])
                    )
                elif "pour" in task_name:
                    env.target_object.set_pose(
                        sapien.Pose([0, 0.2, env.bowl_height], [1, 0, 0, 0])
                    )

            print("Target Pos: {}".format(aug_random_target))
        else:
            if "pick_place" in task_name:
                env.target_object.set_pose(
                    sapien.Pose([-0.005, -0.12, 0], [1, 0, 0, 0])
                )
            elif "pour" in task_name:
                env.target_object.set_pose(
                    sapien.Pose([0, 0.2, env.bowl_height], [1, 0, 0, 0])
                )

        for _ in range(10 * env.frame_skip):
            env.scene.step()



    obs = env.get_observation()
    success = False
    # max_time_step = 1300
    max_time_step = 1100
    action_dim = 22
    all_time_actions = torch.zeros(
        [max_time_step, max_time_step + args["num_queries"], action_dim]
    ).cuda()
    video = []
    stop_frame = 0
    for i in range(max_time_step):
        video_frame = obs["relocate_view-rgb"].cpu().detach().numpy()
        video.append((video_frame * 255).astype(np.uint8))

        img = torch.moveaxis(obs["relocate_view-rgb"], -1, 0)[None, ...]
        robot_qpos = np.concatenate(
            [env.robot.get_qpos(), env.ee_link.get_pose().p,
             env.ee_link.get_pose().q]
        )

        feature = img[None, ...]
        robot_qpos = robot_qpos[None, ...]
        robot_qpos = torch.from_numpy(robot_qpos).to(img.device)

        with torch.inference_mode():
            action = policy(feature, robot_qpos)

        all_time_actions[[i], i: i + args["num_queries"]] = action
        actions_for_curr_step = all_time_actions[:, i]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        raw_action = (actions_for_curr_step *
                      exp_weights).sum(dim=0, keepdim=True)
        raw_action = raw_action.cpu().detach().numpy()
        real_action = apply_IK_get_real_action(
            raw_action, env, env.robot.get_qpos())

        next_obs, reward, done, info = env.step(real_action)

        info_success = info["success"]

        success = success or info_success
        if success:
            stop_frame += 1
            if task_name in ["dclaw"]:
                break
        if stop_frame == 50:
            break

        obs = deepcopy(next_obs)

    # only save video if success or in the final_success evaluation
    # if success:
    if "pick_place" in task_name:
        is_lifted = info["is_object_lifted"]
        video_path = os.path.join(
            log_dir,
            f"epoch_{epoch}_{eval_idx}_rank{randomness_rank}_{success}_{is_lifted}.mp4",
        )
    elif "pour" in task_name:
        num_in_bowl = info["num_box_in_bowl"]
        video_path = os.path.join(
            log_dir,
            f"epoch_{epoch}_{eval_idx}_rank{randomness_rank}_{success}_{num_in_bowl}.mp4",
        )
    elif "dclaw" in task_name:
        total_angle = info["object_total_rotate_angle"]
        video_path = os.path.join(
            log_dir,
            f"epoch_{epoch}_{eval_idx}_rank{randomness_rank}_{success}_{total_angle}.mp4",
        )
    # imageio version 2.28.1 imageio-ffmpeg version 0.4.8 scikit-image version 0.20.0
    imageio.mimsave(video_path, video, fps=60)
    avg_success.append(int(success))

    return int(success)


class Eval_player:
    device: torch.device
    remotes: List[Connection] = []
    work_remotes: List[Connection] = []
    processes: List[mp.Process] = []

    def __init__(self, num_workers: int, args, policy):
        ###### Initial the workers######
        self.num_workers = num_workers
        self.policy = policy
        self.args = args
        ctx = mp.get_context("forkserver")
        self.eval_pool = ctx.Pool(self.num_workers)

    def eval_init(self):
        ctx = mp.get_context("forkserver")
        manager = ctx.Manager()
        self.avg_success = {
            1: manager.list(),
            2: manager.list(),
            3: manager.list(),
            4: manager.list(),
            5: manager.list(),
        }

    def eval_start(self, log_dir, epoch, eval_idx, x, y, randomness_rank):
        self.eval_pool.apply_async(
            eval_in_env,
            args=(
                self.args,
                log_dir,
                epoch,
                eval_idx,
                x,
                y,
                randomness_rank,
                self.policy,
                self.avg_success[randomness_rank],
            ),
        )

    def eval_get_result(self):
        return self.avg_success

    def eval_terminate(self):
        self.eval_pool.close()
        self.eval_pool.terminate()
        ctx = mp.get_context("forkserver")
        self.eval_pool = ctx.Pool(self.num_workers)
