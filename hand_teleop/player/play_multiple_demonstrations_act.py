import os
import numpy as np
import h5py
from tqdm import tqdm
from argparse import ArgumentParser
import random

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import v2

from hand_teleop.player.player import *
from hand_teleop.player.randomization_utils import *
from hand_teleop.real_world import lab

from dataset.act_dataset import set_seed
from models.vision.feature_extractor import generate_features


def play_multiple_sim_visual(args):
    set_seed(args["seed"])
    demo_files = []
    dataset_folder = args["out_folder"]
    if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
        shutil.rmtree(dataset_folder)
    os.makedirs(dataset_folder)
    if args["with_features"]:
        assert args["backbone_type"] != None

    for file_name in os.listdir(args["sim_demo_folder"]):
        if ".pickle" in file_name:
            demo_files.append(os.path.join(args["sim_demo_folder"], file_name))

    print("Replaying the sim demos and creating the dataset:")
    print("---------------------")
    init_obj_poses = []
    lifted_chunks = {}
    chunks_sensitivity = {}
    total = 0
    total_episodes = 0
    dataset_path = "{}/dataset.h5".format(args["out_folder"])
    file1 = h5py.File(dataset_path, "w")

    for _, file_name in enumerate(demo_files):
        print(file_name)
        demo_idx = file_name.split("/")[-1].split(".")[0]
        iteration = 2 if "multi_view" in args['task_name'] else 1
        for it in range(iteration):
            with open(file_name, "rb") as file:
                demo = pickle.load(file)
                (
                    visual_baked,
                    meta_data,
                    info_success,
                    lifted_chunk,
                    chunk_sensitivity,
                ) = play_one_real_sim_visual_demo(args=args, demo=demo)
                chunks_sensitivity[demo_idx] = chunk_sensitivity
                lifted_chunks[demo_idx] = lifted_chunk
                if not info_success:
                    continue
                total += 1
                init_obj_poses.append(meta_data["env_kwargs"]["init_obj_pos"])
            total_episodes, obs, action, robot_qpos = stack_and_save_frames(
                visual_baked, total_episodes, args, file1
            )

    print("Fail sim demos:", len(demo_files)*iteration - total)
    print("---------------------")

    print("Dataset ready:")
    print("----------------------")
    print("Number of demos: {}".format(total))
    print("Number of datapoints: {}".format(
        total_episodes * args["chunk_size"]))
    print("Shape of observations: {}".format(obs.shape))
    print("Shape of Robot_qpos: {}".format(robot_qpos.shape))
    print("Action dimension: {}".format(len(action)))
    meta_data_path = "{}/meta_data.pickle".format(dataset_folder)
    meta_data["init_obj_poses"] = init_obj_poses
    meta_data["chunks_sensitivity"] = chunks_sensitivity
    meta_data["lifted_chunks"] = lifted_chunks
    meta_data["num_img_aug"] = args["num_data_aug"]
    meta_data["total_episodes"] = total_episodes
    file1.close()
    with open(meta_data_path, "wb") as file:
        pickle.dump(meta_data, file)
    print("dataset is saved in the folder: {}".format(dataset_folder))


def play_multiple_real_visual(args):
    demo_files = []
    dataset_folder = args["out_folder"]
    if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
        shutil.rmtree(dataset_folder)
    os.makedirs(dataset_folder)
    if args["with_features"]:
        assert args["backbone_type"] != None

    for file_name in os.listdir(args["real_demo_folder"]):
        if ".pkl" in file_name:
            demo_files.append(os.path.join(
                args["real_demo_folder"], file_name))

    print("Replaying the real demos and creating the dataset:")
    print("---------------------")

    init_obj_poses = []
    total_episodes = 0
    dataset_path = "{}/dataset.h5".format(args["out_folder"])
    file1 = h5py.File(dataset_path, "w")
    for _, file_name in enumerate(demo_files):
        print("file_name: ", file_name)
        image_file = file_name.strip(".pkl")
        print("image_file: ", image_file)

        with open(file_name, "rb") as file:
            real_demo = pickle.load(file)
            path = "./sim/raw_data/{}_{}/{}_0004.pickle".format(
                "pick_place", "mustard_bottle", "mustard_bottle")
            # print("sim_file: ", path)
            demo = np.load(path, allow_pickle=True)
            visual_baked, meta_data = play_one_real_sim_visual_demo(
                args=args,
                demo=demo,
                real_demo=real_demo,
                real_images=image_file,
                using_real_data=True,
            )
            init_obj_poses.append(meta_data["env_kwargs"]["init_obj_pos"])
            # visual_baked_demos.append(visual_baked)
        total_episodes, obs, action, robot_qpos = stack_and_save_frames(
            visual_baked, total_episodes, args, file1, using_real_data=True
        )
        # since here we are using real data, we set sim_real_label = 1

    print("Dataset ready:")
    print("----------------------")
    print("Number of demos: {}".format(len(demo_files)))
    print("Number of datapoints: {}".format(
        total_episodes * args["chunk_size"]))
    print("Shape of observations: {}".format(obs.shape))
    print("Shape of Robot_qpos: {}".format(robot_qpos.shape))
    print("Action dimension: {}".format(len(action)))
    meta_data_path = "{}/meta_data.pickle".format(dataset_folder)
    meta_data["init_obj_poses"] = init_obj_poses
    meta_data["num_img_aug"] = args["num_data_aug"]
    meta_data["total_episodes"] = total_episodes
    file1.close()
    with open(meta_data_path, "wb") as file:
        pickle.dump(meta_data, file)
    print("dataset is saved in the folder: {}".format(dataset_folder))


def play_one_real_sim_visual_demo(
    args, demo, real_demo=None, real_images=None, using_real_data=False
):
    robot_name = args["robot_name"]
    retarget = args["retarget"]
    frame_skip = args["frame_skip"]

    if robot_name == "mano":
        assert retarget == False
    # Get env params
    meta_data = demo["meta_data"]
    if not retarget:
        assert robot_name == meta_data["robot_name"]
    task_name = meta_data["env_kwargs"]["task_name"]
    meta_data["env_kwargs"].pop("task_name")
    meta_data["task_name"] = task_name
    data = demo["data"]

    use_visual_obs = True
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

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    if robot_name == "mano":
        env_params["zero_joint_pos"] = meta_data["zero_joint_pos"]
    else:
        env_params["zero_joint_pos"] = None

    if "init_obj_pos" in meta_data["env_kwargs"].keys():
        print("Found initial object pose")
        env_params["init_obj_pos"] = meta_data["env_kwargs"]["init_obj_pos"]

    if "init_target_pos" in meta_data["env_kwargs"].keys():
        print("Found initial target pose")
        env_params["init_target_pos"] = meta_data["env_kwargs"]["init_target_pos"]

    if "pick_place" in task_name:
        env = PickPlaceRLEnv(**env_params)
    elif "dclaw" in task_name:
        env = DClawRLEnv(**env_params)
    elif "pour" in task_name:
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
    if "pick_place" in task_name:
        player = PickPlaceEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif "dclaw" in task_name:
        player = DcLawEnvPlayer(
            meta_data, data, env, zero_joint_pos=env_params["zero_joint_pos"]
        )
    elif "pour" in task_name:
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
    elif using_real_data:
        baked_data = real_demo
    else:
        baked_data = player.bake_demonstration()

    visual_baked = dict(obs=[], action=[], robot_qpos=[])
    env.reset()
    player.scene.unpack(player.get_sim_data(0))

    for _ in range(player.env.frame_skip):
        player.scene.step()
    if player.human_robot_hand is not None:
        player.scene.remove_articulation(player.human_robot_hand.robot)

    if using_real_data:
        env.robot.set_qpos(baked_data[0]["teleop_cmd"])
        print("init_qpos: ", baked_data[0]["teleop_cmd"])
    else:
        env.robot.set_qpos(baked_data["robot_qpos"][0])
        if baked_data["robot_qvel"] != []:
            env.robot.set_qvel(baked_data["robot_qvel"][0])

    robot_pose = env.robot.get_pose()

    if using_real_data:
        ee_pose = baked_data[0]["ee_pose"]
        hand_qpos_prev = baked_data[0]["teleop_cmd"][env.arm_dof:]
    else:
        ee_pose = baked_data["ee_pose"][0]
        hand_qpos_prev = baked_data["action"][0][env.arm_dof:]

    valid_frame = 0
    lifted_chunk = 0
    stop_frame = 0
    if using_real_data:
        for idx in tqdm(range(len(baked_data))):
            # NOTE: robot.get_qpos() version
            if idx != len(baked_data) - 1:
                ee_pose_next = np.array(baked_data[idx + 1]["ee_pose"])
                ee_pose_delta = np.sqrt(
                    np.sum((ee_pose_next[:3] - ee_pose[:3]) ** 2))
                hand_qpos = baked_data[idx + 1]["teleop_cmd"][env.arm_dof:]

                delta_hand_qpos = hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos

                if (
                    ee_pose_delta <= args["real_delta_ee_pose_bound"]
                    and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2
                ):
                    continue
                else:
                    valid_frame += 1
                    ee_pose = ee_pose_next
                    hand_qpos_prev = hand_qpos

                    palm_pose = env.ee_link.get_pose()
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
                        palm_pose.to_transformation_matrix()[
                            :3, :3] @ delta_axis
                    )
                    delta_pose = np.concatenate(
                        [palm_next_pose.p - palm_pose.p,
                            delta_axis_world * delta_angle]
                    )

                    palm_jacobian = (
                        env.kinematic_model.compute_end_link_spatial_jacobian(
                            env.robot.get_qpos()[: env.arm_dof]
                        )
                    )
                    palm_jacobian = (
                        env.kinematic_model.compute_end_link_spatial_jacobian(
                            env.robot.get_qpos()[: env.arm_dof]
                        )
                    )
                    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[
                        : env.arm_dof
                    ]
                    arm_qpos = arm_qvel + env.robot.get_qpos()[: env.arm_dof]

                    observation = env.get_observation()

                    rgb_pic = torchvision.io.read_image(
                        path=os.path.join(real_images, "frame%04i.png" % idx),
                        mode=torchvision.io.ImageReadMode.RGB,
                    )
                    rgb_pic = v2.Pad(padding=[0, 80])(rgb_pic)
                    rgb_pic = v2.Resize(size=[224, 224])(rgb_pic)
                    rgb_pic = rgb_pic.permute(1, 2, 0)
                    # print("rgb_pic: ", rgb_pic.shape)
                    rgb_pic = (rgb_pic / 255.0).type(torch.float32)

                    observation["relocate_view-rgb"] = rgb_pic
                    visual_baked["obs"].append(observation)
                    action = np.concatenate([delta_pose * 100, hand_qpos])
                    visual_baked["action"].append(action.astype(np.float32))
                    # Using robot qpos version
                    visual_baked["robot_qpos"].append(
                        np.concatenate(
                            [
                                env.robot.get_qpos(),
                                env.ee_link.get_pose().p,
                                env.ee_link.get_pose().q,
                            ]
                        )
                    )

                    target_qpos = np.concatenate([arm_qpos, hand_qpos])
                    env.robot.set_qpos(target_qpos)
                    if valid_frame >= 1500 and args['task_name'] == "dclaw":
                        break

        return visual_baked, meta_data

    else:
        for idx in tqdm(range(0, len(baked_data["obs"]), frame_skip)):
            # NOTE: robot.get_qpos() version
            if idx != len(baked_data["obs"]) - frame_skip:
                ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
                ee_pose_delta = np.sqrt(
                    np.sum((ee_pose_next[:3] - ee_pose[:3]) ** 2))
                hand_qpos = baked_data["action"][idx][env.arm_dof:]
                delta_hand_qpos = hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos

                if (
                    ee_pose_delta <= args["sim_delta_ee_pose_bound"]
                    and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2
                ):
                    continue
                else:
                    valid_frame += 1

                    if task_name in ["pick_place", "pour"]:
                        if env._is_object_lifted() and lifted_chunk == 0:
                            lifted_chunk = int((valid_frame - 1) / 50)

                    ee_pose = ee_pose_next
                    hand_qpos_prev = hand_qpos

                    palm_pose = env.ee_link.get_pose()
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
                        palm_pose.to_transformation_matrix()[
                            :3, :3] @ delta_axis
                    )
                    delta_pose = np.concatenate(
                        [palm_next_pose.p - palm_pose.p,
                            delta_axis_world * delta_angle]
                    )

                    palm_jacobian = (
                        env.kinematic_model.compute_end_link_spatial_jacobian(
                            env.robot.get_qpos()[: env.arm_dof]
                        )
                    )
                    arm_qvel = compute_inverse_kinematics(delta_pose, palm_jacobian)[
                        : env.arm_dof
                    ]
                    arm_qpos = arm_qvel + env.robot.get_qpos()[: env.arm_dof]
                    target_qpos = np.concatenate([arm_qpos, hand_qpos])

                    observation = env.get_observation()
                    visual_baked["obs"].append(observation)
                    visual_baked["action"].append(
                        np.concatenate([delta_pose * 100, hand_qpos])
                    )
                    # Using robot qpos version
                    visual_baked["robot_qpos"].append(
                        np.concatenate(
                            [
                                env.robot.get_qpos(),
                                env.ee_link.get_pose().p,
                                env.ee_link.get_pose().q,
                            ]
                        )
                    )

                    _, _, _, info = env.step(target_qpos)

                    info_success = info["success"]

                    if info_success:
                        stop_frame += 1

                    if stop_frame >= 30:
                        break

        chunk_sensitivity = []
        # if info_success and task_name in ["pick_place", "pour"]:
        #     # assign weights for action chunk
        #     total_frame = len(visual_baked["obs"])
        #     for i in tqdm(range(total_frame // 50)):
        #         for var in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        #             valid_frame = 0
        #             env.reset()
        #             for idx in range(0, len(baked_data["obs"]), frame_skip):
        #                 # NOTE: robot.get_qpos() version
        #                 if idx < len(baked_data["obs"]) - frame_skip:
        #                     ee_pose_next = baked_data["ee_pose"][idx + frame_skip]
        #                     ee_pose_delta = np.sqrt(
        #                         np.sum((ee_pose_next[:3] - ee_pose[:3]) ** 2)
        #                     )
        #                     hand_qpos = baked_data["action"][idx][env.arm_dof:]
        #                     delta_hand_qpos = (
        #                         hand_qpos - hand_qpos_prev if idx != 0 else hand_qpos
        #                     )

        #                     if (
        #                         ee_pose_delta < 0.001
        #                         and np.mean(handqpos2angle(delta_hand_qpos)) <= 1.2
        #                         and task_name in ["pick_place", "dclaw"]
        #                     ):
        #                         continue
        #                     else:
        #                         valid_frame += 1
        #                         ee_pose = ee_pose_next
        #                         hand_qpos_prev = hand_qpos

        #                         palm_pose = env.ee_link.get_pose()
        #                         palm_pose = robot_pose.inv() * palm_pose

        #                         palm_next_pose = sapien.Pose(
        #                             ee_pose_next[0:3], ee_pose_next[3:7]
        #                         )
        #                         palm_next_pose = robot_pose.inv() * palm_next_pose

        #                         palm_delta_pose = palm_pose.inv() * palm_next_pose
        #                         (
        #                             delta_axis,
        #                             delta_angle,
        #                         ) = transforms3d.quaternions.quat2axangle(
        #                             palm_delta_pose.q
        #                         )
        #                         if delta_angle > np.pi:
        #                             delta_angle = 2 * np.pi - delta_angle
        #                             delta_axis = -delta_axis
        #                         delta_axis_world = (
        #                             palm_pose.to_transformation_matrix()[
        #                                 :3, :3]
        #                             @ delta_axis
        #                         )
        #                         delta_pose = np.concatenate(
        #                             [
        #                                 palm_next_pose.p - palm_pose.p,
        #                                 delta_axis_world * delta_angle,
        #                             ]
        #                         )
        #                         ############################## Action Chunk Test#################################
        #                         if valid_frame > i * 50 and valid_frame <= (i + 1) * 50:
        #                             delta_pose = delta_pose * var
        #                             delta_hand_qpos = delta_hand_qpos * var
        #                             hand_qpos = hand_qpos_prev + delta_hand_qpos

        #                         palm_jacobian = env.kinematic_model.compute_end_link_spatial_jacobian(
        #                             env.robot.get_qpos()[: env.arm_dof]
        #                         )
        #                         arm_qvel = compute_inverse_kinematics(
        #                             delta_pose, palm_jacobian
        #                         )[: env.arm_dof]
        #                         arm_qpos = (
        #                             arm_qvel +
        #                             env.robot.get_qpos()[: env.arm_dof]
        #                         )

        #                         target_qpos = np.concatenate(
        #                             [arm_qpos, hand_qpos])

        #                         _, _, _, info = env.step(target_qpos)

        #             if not info["success"] or var == 2.0:
        #                 chunk_sensitivity.append(var)
        #                 break

        return visual_baked, meta_data, info_success, lifted_chunk, chunk_sensitivity


def stack_and_save_frames(
    visual_baked, total_episode, args, file1, using_real_data=False
):
    # 0 menas sim, 1 means real here
    visual_demo_with_features = generate_features(
        visual_baked=visual_baked,
        backbone_type=args["backbone_type"],
        num_data_aug=args["num_data_aug"],
        augmenter=args["image_augmenter"],
        using_features=args["with_features"],
    )
    assert len(visual_demo_with_features["obs"]) % args["num_data_aug"] == 0
    data_aug_length = int(
        len(visual_demo_with_features["obs"]) / args["num_data_aug"])
    obs = []
    action = []
    robot_qpos = []
    for ii in range(args["num_data_aug"]):
        obs.extend(
            visual_demo_with_features["obs"][
                ii * data_aug_length: (ii + 1) * data_aug_length
            ]
        )
        action.extend(
            visual_demo_with_features["action"][
                ii * data_aug_length: (ii + 1) * data_aug_length
            ]
        )
        robot_qpos.extend(
            visual_demo_with_features["robot_qpos"][
                ii * data_aug_length: (ii + 1) * data_aug_length
            ]
        )

    for episode in range(len(obs) // args["chunk_size"]):
        obs_chunk = obs[
            episode * args["chunk_size"]: (episode + 1) * args["chunk_size"]
        ]
        action_chunk = action[
            episode * args["chunk_size"]: (episode + 1) * args["chunk_size"]
        ]
        robot_qpos_chunk = robot_qpos[
            episode * args["chunk_size"]: (episode + 1) * args["chunk_size"]
        ]
        sim_real_label_chunk = (
            np.array([1 for _ in range(len(obs_chunk))])
            if using_real_data
            else torch.tensor([0 for _ in range(len(obs_chunk))])
        )
        # visual_training_set = {"obs": obs_chunk, "action": action_chunk, "robot_qpos": robot_qpos_chunk, "sim_real_label": sim_real_label_chunk}
        g1 = file1.create_group("episode_{}".format(episode + total_episode))
        g1.create_dataset("obs", data=obs_chunk)
        g1.create_dataset("action", data=action_chunk)
        g1.create_dataset("robot_qpos", data=robot_qpos_chunk)
        g1.create_dataset("sim_real_label", data=sim_real_label_chunk)

    return episode + total_episode + 1, obs[0], action[0], robot_qpos[0]


def stack_and_save_frames_aug(
    visual_baked, total_episode, args, file1, using_real_data=False
):
    # 0 menas sim, 1 means real here
    obs = []
    for observation in visual_baked["obs"]:
        img = observation["relocate_view-rgb"]
        img = torch.moveaxis(img, -1, 0)[None, ...]
        img = img.detach().cpu().numpy()
        obs.append(img)

    action = visual_baked["action"]
    robot_qpos = visual_baked["robot_qpos"]

    for episode in range(len(obs) // args["num_queries"]):
        obs_chunk = obs[
            episode * args["num_queries"]: (episode + 1) * args["num_queries"]
        ]
        action_chunk = action[
            episode * args["num_queries"]: (episode + 1) * args["num_queries"]
        ]
        robot_qpos_chunk = robot_qpos[
            episode * args["num_queries"]: (episode + 1) * args["num_queries"]
        ]
        sim_real_label_chunk = (
            np.array([1 for _ in range(len(obs_chunk))])
            if using_real_data
            else torch.tensor([0 for _ in range(len(obs_chunk))])
        )
        if f"episode_{episode+total_episode}" in file1.keys():
            del file1[f"episode_{episode+total_episode}"]
        g1 = file1.create_group(f"episode_{episode+total_episode}")
        g1.create_dataset("obs", data=obs_chunk)
        g1.create_dataset("action", data=action_chunk)
        g1.create_dataset("robot_qpos", data=robot_qpos_chunk)
        g1.create_dataset("sim_real_label", data=sim_real_label_chunk)

    # if len(obs) % args["num_queries"] != 0:
    #     obs_chunk = obs[(episode + 1) * args["num_queries"] :]
    #     obs_chunk = obs_chunk + [
    #         obs_chunk[-1] for _ in range(args["num_queries"] - len(obs_chunk))
    #     ]
    #     action_chunk = action[(episode + 1) * args["num_queries"] :]
    #     action_chunk = action_chunk + [
    #         action_chunk[-1] for _ in range(args["num_queries"] - len(action_chunk))
    #     ]
    #     robot_qpos_chunk = robot_qpos[(episode + 1) * args["num_queries"] :]
    #     robot_qpos_chunk = robot_qpos_chunk + [
    #         robot_qpos_chunk[-1]
    #         for _ in range(args["num_queries"] - len(robot_qpos_chunk))
    #     ]
    #     sim_real_label_chunk = (
    #         np.array([1 for _ in range(len(obs_chunk))])
    #         if using_real_data
    #         else torch.tensor([0 for _ in range(len(obs_chunk))])
    #     )
    #     episode = episode + 1
    #     if f"episode_{episode+total_episode}" in file1.keys():
    #         del file1[f"episode_{episode+total_episode}"]
    #     g1 = file1.create_group(f"episode_{episode+total_episode}")
    #     g1.create_dataset("obs", data=obs_chunk)
    #     g1.create_dataset("action", data=action_chunk)
    #     g1.create_dataset("robot_qpos", data=robot_qpos_chunk)
    #     g1.create_dataset("sim_real_label", data=sim_real_label_chunk)

    return episode + total_episode + 1, obs[0], action[0], robot_qpos[0]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--backbone-type", default=None)
    parser.add_argument("--sim-delta-ee-pose-bound",
                        default="0.0005", type=float)
    parser.add_argument("--real-delta-ee-pose-bound",
                        default="0.0005", type=float)
    parser.add_argument("--frame-skip", default="1", type=int)
    parser.add_argument("--chunk-size", default="50", type=int)
    parser.add_argument("--img-data-aug", default="1", type=int)
    parser.add_argument("--sim-folder", default=None)
    parser.add_argument("--real-folder", default=None)
    parser.add_argument("--task-name", default="pick_place")
    parser.add_argument("--object-name", default="mustard_bottle")
    parser.add_argument("--out-folder", required=True)
    parser.add_argument("--with-features", default=False)
    parser.add_argument("--seed", default=20230929, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Since we are doing online retargeting currently, 'retargeting' argument should be False all the time in this file.
    # It might be better to save each frame one by one if you need the images itself. you can save it all as one file if you are just using the features.
    args = parse_args()

    args = {
        "sim_demo_folder": args.sim_folder,
        "real_demo_folder": args.real_folder,
        "task_name": args.task_name,
        "object_name": args.object_name,
        "robot_name": "xarm6_allegro_modified_finger",
        "with_features": args.with_features,
        "backbone_type": args.backbone_type,
        "retarget": False,
        "save_each_frame": False,
        "domain_randomization": False,
        "randomization_prob": 0.2,
        "num_data_aug": args.img_data_aug,
        "image_augmenter": T.AugMix(),
        "sim_delta_ee_pose_bound": args.sim_delta_ee_pose_bound,
        "real_delta_ee_pose_bound": args.real_delta_ee_pose_bound,
        "frame_skip": args.frame_skip,
        "chunk_size": args.chunk_size,
        "out_folder": args.out_folder,
        "seed": args.seed,
    }

    if args["sim_demo_folder"] is not None and args["real_demo_folder"] is None:
        print(
            "##########################Using Only Sim##################################"
        )
        play_multiple_sim_visual(args)
    elif args["sim_demo_folder"] is None and args["real_demo_folder"] is not None:
        print(
            "##########################Using Only Real##################################"
        )
        play_multiple_real_visual(args)
