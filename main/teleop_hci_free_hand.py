import os
from pathlib import Path

import numpy as np
import sapien.core as sapien
import torch
import transforms3d.euler

from hand_detector.hand_monitor import Record3DSingleHandMotionControl
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.env.sim_env.pick_place_env import PickPlaceEnv
from hand_teleop.env.sim_env.laptop_env import LaptopEnv
from hand_teleop.env.sim_env.insert_object_env import InsertObjectEnv
from hand_teleop.env.sim_env.hammer_env import HammerEnv
from hand_teleop.env.sim_env.mug_flip_env import MugFlipEnv
from hand_teleop.env.rl_env.base import compute_inverse_kinematics
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.kinematics_helper import PartialKinematicModel
# from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.player.recorder import DataRecorder
from hand_teleop.teleop.teleop_gui import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from hand_teleop.utils.common_robot_utils import load_robot, generate_arm_robot_hand_info

from hand_teleop.retargeting import build_default_retargeting, LPFilter

## NOTE: THIS SCRIPT CAN ONLY CONTROL FREE HAND / MANO HAND!!!
def main():
    # Setup
    demo_index = 0
    frame_skip = 5

    task_name = "pick_place"
    object_names = ['tomato_soup_can', 'bleach_cleanser', 'mug', 'banana', "mustard_bottle", "potted_meat_can"]
    object_name = object_names[-2]
    operator = "test"
    # robot_name = "mano"
    robot_name = "allegro_hand_free"
    hand_mode = "right_hand"
    object_scale = 0.8
    randomness_scale = 1

    folder_name = task_name
    if task_name == "relocate" or task_name == "pick_place":
        folder_name = '{}_{}'.format(folder_name, object_name)
    out_folder = f"./sim/raw_data/free_hand/less_random/{folder_name}/"
    os.makedirs(out_folder, exist_ok=True)
    if len(os.listdir(out_folder)) == 0:
        num_test = "0001"
    else:
        pkl_files = os.listdir(out_folder)
        last_num = sorted([int(x.replace(".pickle", "").split("_")[-1]) for x in pkl_files])[-1]
        num_test = str(last_num + 1).zfill(4)
    print(num_test)
    demo_name = num_test
    if task_name == "table_door":
        env = TableDoorEnv(frame_skip=frame_skip)
        env_dict = dict(task_name=task_name, frame_skip=frame_skip, init_obj_pos=env.init_pose)
    elif task_name == "relocate":
        env = RelocateEnv(frame_skip=frame_skip, object_name=object_name, randomness_scale=1, object_scale=0.8)
        env.reset_env()
        env_dict = dict(task_name=task_name, object_name=object_name, object_scale=object_scale, frame_skip=frame_skip)
        demo_name = '{}_{}'.format(object_name, demo_name)
    elif task_name == "laptop":
        env = LaptopEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(task_name=task_name)
    elif task_name == "pick_place":
        env = PickPlaceEnv(object_name=object_name,frame_skip=frame_skip, object_scale=object_scale, randomness_scale=randomness_scale)
        env.reset_env()
        env_dict = dict(task_name=task_name, object_name=object_name, object_scale=object_scale, randomness_scale=randomness_scale, init_obj_pos=env.init_pose, init_target_pos=env.target_pose)
        demo_name = '{}_{}'.format(object_name, demo_name)
    elif task_name == 'insert_object':
        env = InsertObjectEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(task_name=task_name, frame_skip=frame_skip, init_obj_pos=env.init_pose, init_target_pos=env.target_pose)
    elif task_name == 'hammer':
        env = HammerEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(task_name=task_name, frame_skip=frame_skip, init_obj_pos=env.init_pose, init_target_pos=env.target_pose)
    elif task_name == 'mug_flip':
        env = MugFlipEnv(frame_skip=frame_skip, object_scale=1)
        env.reset_env()
        env_dict = dict(task_name=task_name, frame_skip=frame_skip, object_scale=1, init_obj_pos=env.init_pose)
    else:
        raise NotImplementedError

    # print(env.manipulated_object.get_pose())
    # root_data_path = "/home/sim/data/teleop"
    # path = Path.home() / "data" / "teleop" / "hci" / operator
    # path = path / f"{task_name}-{robot_name}-{num_test}.pickle"
    path = "./sim/raw_data/free_hand/less_random/{}/{}.pickle".format(folder_name, demo_name)
    # path = "./test_demo.pickle"
    asset_path = Path(__file__).parent.parent / "assets" # check this

    env.seed(int(demo_index))
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)

    if task_name == "table_door":
        gui.viewer.set_camera_xyz(-0.6, 0, 0.6)
        gui.viewer.set_camera_rpy(0, -np.pi/6, 0)
    else:
        gui.viewer.set_camera_xyz(-0.6, 0, 0.6)
        # gui.viewer.set_camera_xyz(-1, 0, 0.6)
        gui.viewer.set_camera_rpy(0, -np.pi/6, 0)

    # Perception
    motion_control = Record3DSingleHandMotionControl(hand_mode=hand_mode, show_hand=True, need_init=True,
                                                     virtual_video_file="")

    # Renderer
    scene = env.scene
    viz_mat_hand_init = gui.context.create_material(np.array([0, 0, 0, 0]), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8,
                                                    0)
    # viz_mat_hand_init = create_visual_material(env.renderer, 0.0, 0.8, 0, np.array([0.96, 0.75, 0.69, 1]))
    # viz_mat_hand_init = env.renderer.create_material()

    # Recorder
    # os.makedirs(str(path.parent), exist_ok=True)
    # recorder = DataRecorder(filename=str(path.resolve()), scene=scene)
    recorder = DataRecorder(filename=path, scene=scene)

    # Robot
    # robot_filter = LPFilter(10, 8)
    weight = 1
    if "allegro" in robot_name:
        arm_scale = 50
        trans_scale = 50
        rot_scale = 50
        fing_scale = 50
    else:
        arm_scale = 10
        trans_scale = 10
        rot_scale = 10
        fing_scale = 10
    robot_arm_control_params = np.array([60000, 1000, 2000]) * arm_scale  # This PD is far larger than real to improve stability
    root_translation_control_params = np.array([10000, 1000, 2000]) * trans_scale
    root_rotation_control_params = np.array([2000, 200, 400]) * rot_scale
    finger_control_params = np.array([2000, 60, 10]) * fing_scale

    # Init
    create_robot = False
    steps = 0
    env_init_pos = np.array([-0.3, 0, 0.2])
    if hand_mode == 'right_hand':
        env_init_ori = transforms3d.euler.euler2quat(0, np.pi / 2, 0)
    else:
        env_init_ori = transforms3d.euler.euler2quat(0, np.pi / 2, np.pi)
    # env_init_pos = np.zeros(3)
    rgb, depth = motion_control.camera.fetch_rgb_and_depth()
    locked_indices = []
    scene.step()
    while not gui.closed:
        for _ in range(frame_skip):
            scene.step()
        gui.render(additional_views=[rgb[..., ::-1]])
        steps += 1

        if not motion_control.initialized:
            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]
            if not success:
                continue

            viz_mat_hand_init.set_base_color(motion_control.init_process_color)
            rotate_pose = sapien.Pose(q=[0.9238, 0, 0.3826, 0], p=[0.2, 0, -0.1])
            gui.update_mesh(motion_data["vertices"], motion_data["faces"], viz_mat=viz_mat_hand_init,
                            clear_context=True, pose=sapien.Pose(env_init_pos + np.array([-0.2, 0, 0])) * rotate_pose)
        else:
            if not create_robot:
                if robot_name == "mano":
                    zero_joint_pos = motion_control.compute_hand_zero_pos()
                    mano_robot = MANORobotHand(env.scene, env.renderer, init_joint_pos=zero_joint_pos, hand_mode=hand_mode,
                                               control_interval=frame_skip * scene.get_timestep(), scale=1)
                    robot = mano_robot.robot
                    robot.set_pose(sapien.Pose(env_init_pos, env_init_ori))
                else:
                    if 'left' in hand_mode:
                        robot_name = '{}_left'.format(robot_name)
                    robot = load_robot(env.scene, robot_name)
                    # Robot
                    retargeting, link_hand_indices = build_default_retargeting(asset_path,robot_name)
                    joint_filter = LPFilter(alpha=0.5)
                    
                    if "free" in robot_name:
                        for joint in robot.get_active_joints():
                            name = joint.get_name()
                            if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                                joint.set_drive_property(*(weight * root_translation_control_params), mode="acceleration")
                            elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                                joint.set_drive_property(*(weight * root_rotation_control_params), mode="acceleration")
                            else:
                                joint.set_drive_property(*(weight * finger_control_params), mode="acceleration")
                        robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)))        
                    elif "xarm" in robot_name:
                        raise NotImplementedError
                #         arm_filter = LPFilter(alpha=0.5)
                #         info = generate_arm_robot_hand_info()[robot_name]
                #         arm_dof = info.arm_dof
                #         hand_dof = info.hand_dof
                #         start_joint_name = robot.get_joints()[1].get_name()
                #         end_joint_name = robot.get_active_joints()[arm_dof - 1].get_name()
                #         kinematic_model = PartialKinematicModel(robot, start_joint_name, end_joint_name)
                #         arm_joint_names = [f"joint{i}" for i in range(1, 8)]
                #         for joint in robot.get_active_joints():
                #             name = joint.get_name()
                #             if name in arm_joint_names:
                #                 joint.set_drive_property(*(weight * robot_arm_control_params), mode="force")
                #             else:
                #                 joint.set_drive_property(*(weight * finger_control_params), mode="force")
                #         # prev_root_joint_qpos = np.concatenate([env_init_pos, np.array([0, np.pi / 2, 0])])
                #         init_robot = False
                #         robot.set_pose(sapien.Pose(np.array([-0.50, 0, 0.2]), transforms3d.euler.euler2quat(0, 0, 0)))
                #         # robot.set_pose(sapien.Pose(np.array([-0.5, 0, 0]), transforms3d.euler.euler2quat(0, 0, 0)))
                # # robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, 0, 0)))

                create_robot = True

                def change_locked():
                    locked_indices.clear()
                    contact_finger_indices = mano_robot.check_contact_finger([env.target_object])
                    locked_indices.extend(contact_finger_indices)
                    mano_robot.highlight_finger_color(contact_finger_indices)

                def clear_locked():
                    locked_indices.clear()
                    mano_robot.clear_finger_color()

                gui.register_keydown_action('z', change_locked)
                gui.register_keydown_action('x', clear_locked)

            success, motion_data = motion_control.step()
            rgb = motion_data["rgb"]

            # Data recording.py
            record_data = motion_data.copy()
            record_data.pop("rgb")
            record_data.pop("depth")
            record_data.update({"success": success})
            recorder.step(record_data)
            if not success:
                continue

            root_joint_qpos = motion_control.compute_operator_space_root_qpos(motion_data)
            # root_joint_qpos[:3] += np.array([0.1, 0, 0.1])
            root_joint_qpos *= 1

            if robot_name == "mano":
                if hand_mode == "left_hand":
                    root_joint_qpos[1:3] = -1*root_joint_qpos[1:3]
                finger_joint_qpos = mano_robot.compute_qpos(motion_data["pose_params"][3:])
                robot_qpos = np.concatenate([root_joint_qpos, finger_joint_qpos])
            else:
                origin_indices = link_hand_indices[0, :]
                task_indices = link_hand_indices[1, :]
                origin_joint_pos = motion_data["joint"][origin_indices]
                human_hand_origin_joint_pos = motion_control.compute_operator_space_joint_pos(origin_joint_pos)
                task_joint_pos = motion_data["joint"][task_indices]
                human_hand_task_joint_pos = motion_control.compute_operator_space_joint_pos(task_joint_pos)
                ref_value = human_hand_task_joint_pos - human_hand_origin_joint_pos
                rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
                ref_value = ref_value @ rotation.T
                retargeted_qpos = retargeting.retarget(ref_value, fixed_qpos=np.array([]))
                retargeted_qpos = joint_filter.next(retargeted_qpos)
                robot_qpos = retargeted_qpos
                if "free" in robot_name:
                    if hand_mode == "left_hand":
                        root_joint_qpos[-2:] = -1*root_joint_qpos[-2:]
                    robot_qpos[:6] = root_joint_qpos
                else:
                    raise NotImplementedError
                    # delta_target_pos = root_joint_qpos - prev_root_joint_qpos
                    # temp = delta_target_pos[0]
                    # delta_target_pos[0] = delta_target_pos[2]
                    # delta_target_pos[2] = temp
                    # # delta_target_pos[:-3] = np.flip(delta_target_pos[:-3], 0)
                    # delta_target_pos[-3:] = np.flip(delta_target_pos[-3:], 0)
                    # # delta_target_pos[-1] = -delta_target_pos[-1]
                    # delta_target_pos[2] = -delta_target_pos[2]
                    # palm_jacobian = kinematic_model.compute_end_link_spatial_jacobian(robot.get_qpos()[:arm_dof])
                    # delta_arm_qpos = compute_inverse_kinematics(delta_target_pos, palm_jacobian)[:arm_dof]
                    # delta_arm_qpos = np.clip(delta_arm_qpos, -np.pi, np.pi)
                    # arm_qpos = delta_arm_qpos + robot.get_qpos()[:arm_dof]
                    # arm_qpos = arm_filter.next(arm_qpos)
                    # robot_qpos[:arm_dof] = arm_qpos
                    # prev_root_joint_qpos = root_joint_qpos
                # robot.set_qpos(robot_qpos)
                # print(robot.get_pose())
                # robot.set_drive_target(robot_qpos)
                # robot_drive_qpos = robot_filter.next(robot_qpos)
                robot.set_drive_target(robot_qpos)

            if np.abs(robot.get_qpos().mean()) < 1e-5:
                robot.set_qpos(robot_qpos)

            if robot_name == "mano":
                mano_robot.control_robot(robot_qpos, confidence=motion_data["confidence"],
                                         lock_indices=locked_indices)

            # Create SAPIEN mesh for rendering
            gui.update_mesh(motion_data["vertices"], motion_data["faces"], viz_mat=viz_mat_hand_init,
                            clear_context=True,
                            pose=sapien.Pose(root_joint_qpos[:3] + np.array([0, -0.5, 0]) + env_init_pos))

    print(len(recorder.data_list))
    meta_data = dict(env_class=env.__class__.__name__, env_kwargs=env_dict, operator=operator, robot_name=robot_name,
                     shape_param=motion_control.calibrated_shape_params, hand_mode=hand_mode,
                     zero_joint_pos=motion_control.compute_hand_zero_pos())
    if "free" in robot_name:
        meta_data['finger_control_params'] = weight * finger_control_params
        meta_data['root_rotation_control_params'] = weight * root_rotation_control_params
        meta_data['root_translation_control_params'] = weight * root_translation_control_params
    recorder.dump(meta_data)


if __name__ == '__main__':
    main()
