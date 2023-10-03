from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d.euler

from hand_detector.hand_monitor import Record3DSingleHandMotionControl
from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.kinematics.mano_robot_hand import MANORobotHand
from hand_teleop.kinematics.retargeting_optimizer import PositionRetargeting
from hand_teleop.player.recorder import DataRecorder
from hand_teleop.teleop.teleop_gui import GUIBase, DEFAULT_TABLE_TOP_CAMERAS
from hand_teleop.utils.common_robot_utils import load_robot, LPFilter


def main():
    # Setup
    frame_skip = 5
    demo_index = "0049"
    object_names = ['tomato_soup_can', 'bleach_cleanser', 'mug', 'banana', "mustard_bottle", "potted_meat_can"]
    object_name = object_names[-2]
    robot_name = "mano"
    task_name = "table_door"
    object_scale = 0.8
    if robot_name == "mano":
        if task_name == "relocate":
            task_full_name = f"relocate-{object_name}"
            env_dict = dict(frame_skip=frame_skip, object_name=object_name, object_scale=object_scale)
        elif task_name == "table_door":
            task_full_name = f"table_door"
            env_dict = dict(frame_skip=frame_skip)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    root_data_path = "/home/sim/data/teleop"
    path = Path(root_data_path) / task_full_name
    path = path / f"{demo_index}.pickle"

    if task_name == "table_door":
        env = TableDoorEnv(**env_dict, use_gui=True)
    elif task_name == "relocate":
        env = RelocateEnv(**env_dict, use_gui=True)
    else:
        raise NotImplementedError
    env.reset_env()

    env.seed(int(demo_index))
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)

    if task_name == "table_door":
        gui.viewer.set_camera_rpy(0, -0.7, 0.01)
        gui.viewer.set_camera_xyz(-0.4, 0, 0.45)
    else:
        raise NotImplementedError

    # Perception
    motion_control = Record3DSingleHandMotionControl(hand_mode="right_hand", show_hand=True, need_init=True,
                                                     virtual_video_file="")

    # Renderer
    scene = env.scene
    viz_mat_hand_init = gui.context.create_material(np.array([0, 0, 0, 0]), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8,
                                                    0)
    # Recorder
    recorder = DataRecorder(filename=str(path.resolve()), scene=scene)

    # Robot
    robot_filter = LPFilter(10, 4)
    weight = 1
    root_translation_control_params = np.array([0, 200, 200])
    root_rotation_control_params = np.array([0, 50, 50])
    finger_control_params = np.array([200, 60, 10])

    # Init
    create_robot = False
    steps = 0
    env_init_pos = np.array([-0.4, 0, 0.2])
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
                            clear_context=True, pose=sapien.Pose(env_init_pos) * rotate_pose)
        else:
            if not create_robot:
                if robot_name == "mano":
                    zero_joint_pos = motion_control.compute_hand_zero_pos()
                    mano_robot = MANORobotHand(env.scene, env.renderer, init_joint_pos=zero_joint_pos,
                                               control_interval=frame_skip * scene.get_timestep(), full_dof=True,
                                               scale=1)
                    robot = mano_robot.robot
                    robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)))
                else:
                    robot = load_robot(env.scene, robot_name)
                    # Robot
                    if robot_name == "adroit_hand_free":
                        link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"] + ["thmiddle", "ffmiddle",
                                                                                              "mfmiddle",
                                                                                              "rfmiddle", "lfmiddle"]
                        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
                        link_hand_indices = [0, 4, 8, 12, 16, 20] + [2, 6, 10, 14, 18]
                    elif robot_name == "allegro_hand_free":
                        link_names = ["base_link", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
                        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
                        link_hand_indices = [0, 4, 8, 12, 16]
                    elif robot_name == "svh_hand_free":
                        link_names = ["right_hand_base_link", "right_hand_c", "right_hand_t", "right_hand_s",
                                      "right_hand_r", "right_hand_q"]
                        # link_names += ["right_hand_b", "right_hand_p", "right_hand_o", "right_hand_n", "right_hand_i"]
                        joint_names = [joint.get_name() for joint in robot.get_active_joints()]
                        link_hand_indices = [0, 4, 8, 12, 16, 20]
                    else:
                        raise NotImplementedError
                    retargeting = PositionRetargeting(robot, joint_names, link_names, has_global_pose_limits=False,
                                                      has_joint_limits=True)
                    link_hand_indices = np.array(link_hand_indices)
                    for joint in robot.get_active_joints():
                        name = joint.get_name()
                        if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                            joint.set_drive_property(*(weight * root_translation_control_params), mode="acceleration")
                        elif "x_rotation_joint" in name or "y_rotation_joint" in name or "z_rotation_joint" in name:
                            joint.set_drive_property(*(weight * root_rotation_control_params), mode="acceleration")
                        else:
                            joint.set_drive_property(*(weight * finger_control_params), mode="acceleration")

                    robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)))

                # robot.set_pose(sapien.Pose(env_init_pos, transforms3d.euler.euler2quat(0, 0, 0)))

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
            root_joint_qpos *= 1

            if robot_name == "mano":
                finger_joint_qpos = mano_robot.compute_qpos(motion_data["pose_params"][3:])
                robot_qpos = np.concatenate([root_joint_qpos, finger_joint_qpos])
            else:
                joint_pos = motion_data["joint"][link_hand_indices]
                human_hand_joints = motion_control.compute_operator_space_joint_pos(joint_pos)
                rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
                human_hand_joints = human_hand_joints @ rotation.T
                robot_qpos = retargeting.retarget(human_hand_joints, fixed_qpos=np.array([]))
                robot_qpos[:6] = root_joint_qpos
                robot_drive_qpos = robot_filter.next(robot_qpos)
                robot.set_drive_target(robot_drive_qpos)

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
    meta_data = dict(env_class=env.__class__.__name__, env_kwargs=env_dict, robot_name=robot_name,
                     shape_param=motion_control.calibrated_shape_params,
                     zero_joint_pos=motion_control.compute_hand_zero_pos())
    recorder.dump(meta_data)


if __name__ == '__main__':
    main()
