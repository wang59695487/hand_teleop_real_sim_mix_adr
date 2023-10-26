import os
import numpy as np
import time
import sapien.core as sapien
import transforms3d
from argparse import ArgumentParser

from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.relocate_env import RelocateEnv
from hand_teleop.env.sim_env.table_door_env import TableDoorEnv
from hand_teleop.env.sim_env.pick_place_env import PickPlaceEnv
from hand_teleop.env.sim_env.dclaw_env import DClawEnv
from hand_teleop.env.sim_env.pour_env import PourBoxEnv
from hand_teleop.env.sim_env.laptop_env import LaptopEnv
from hand_teleop.env.sim_env.insert_object_env import InsertObjectEnv
from hand_teleop.env.sim_env.hammer_env import HammerEnv
from hand_teleop.env.sim_env.mug_flip_env import MugFlipEnv
from hand_teleop.player.recorder import DataRecorder
from hand_teleop.teleop.teleop_gui import (
    GUIBase,
    DEFAULT_TABLE_TOP_CAMERAS,
    DEFAULT_ROTATING_CAMERAS,
)
from hand_teleop.utils.common_robot_utils import load_robot
from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.teleop.teleop_client import TeleopClient


## NOTE: THIS SCRIPT CAN ONLY CONTROL HAND CONNECTED TO THE ARM, FOR FREE HANDS/MANO USE teleop_hci_free_hand.py!!!
def main():
    parser = ArgumentParser()
    parser.add_argument("--object", default=None, type=str)
    args = parser.parse_args()
    robot_name = "xarm6_allegro_modified_finger"
    # robot_name = "xarm6_allegro_wrist_mounted_rotate"
    demo_index = 0

    task_name = "pour"
    if task_name == "dclaw":
        frame_skip = 10
    elif task_name in ["pick_place", "pour", "reorientation"]:
        frame_skip = 5

    object_names = [
        "tomato_soup_can",
        "bleach_cleanser",
        "mug",
        "banana",
        "mustard_bottle",
        "potted_meat_can",
        "sugar_box",
        "chip_can",
    ]
    if args.object is not None:
        object_name = args.object
        assert object_name in object_names
    else:
        object_name = object_names[-1]
    operator = "test"
    hand_mode = "right_hand"
    object_scale = 1
    randomness_scale = 1

    folder_name = task_name
    if task_name == "relocate" or task_name == "pick_place":
        folder_name = "{}_{}".format(folder_name, object_name)
    out_folder = f"./sim/raw_data/{folder_name}/"
    os.makedirs(out_folder, exist_ok=True)
    if len(os.listdir(out_folder)) == 0:
        num_test = "0001"
    else:
        pkl_files = os.listdir(out_folder)
        last_num = sorted(
            [int(x.replace(".pickle", "").split("_")[-1]) for x in pkl_files]
        )[-1]
        num_test = str(last_num + 1).zfill(4)
    print(num_test)
    demo_name = num_test
    if task_name == "table_door":
        env = TableDoorEnv(frame_skip=frame_skip)
        env_dict = dict(
            task_name=task_name, frame_skip=frame_skip, init_obj_pos=env.init_pose
        )
    elif task_name == "relocate":
        env = RelocateEnv(
            frame_skip=frame_skip,
            object_name=object_name,
            randomness_scale=1,
            object_scale=0.8,
        )
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            object_name=object_name,
            object_scale=object_scale,
            frame_skip=frame_skip,
        )
        demo_name = "{}_{}".format(object_name, demo_name)
    elif task_name == "laptop":
        env = LaptopEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(task_name=task_name)
    elif task_name == "pick_place":
        env = PickPlaceEnv(
            object_name=object_name,
            object_seed=demo_name,
            frame_skip=frame_skip,
            object_scale=object_scale,
            randomness_scale=randomness_scale,
        )
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            object_name=object_name,
            object_scale=object_scale,
            randomness_scale=randomness_scale,
            init_obj_pos=env.init_pose,
            init_target_pos=env.target_pose,
        )
        demo_name = "{}_{}".format(object_name, demo_name)
    elif task_name == "dclaw":
        object_name = "dclaw_3x"
        env = DClawEnv(
            object_seed=demo_name,
            frame_skip=frame_skip,
            object_scale=object_scale,
            randomness_scale=randomness_scale,
        )
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            object_name=object_name,
            object_scale=object_scale,
            randomness_scale=randomness_scale,
            init_obj_pos=env.init_pose,
        )
        demo_name = "{}_{}".format(object_name, demo_name)
    elif task_name == "pour":
        object_name = "chip_can"
        env = PourBoxEnv(
            object_seed=demo_name,
            frame_skip=frame_skip,
            randomness_scale=randomness_scale,
        )
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            object_name=object_name,
            randomness_scale=randomness_scale,
            init_obj_pos=env.init_pose,
        )
        demo_name = "{}_{}".format(object_name, demo_name)
    elif task_name == "insert_object":
        env = InsertObjectEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            frame_skip=frame_skip,
            init_obj_pos=env.init_pose,
            init_target_pos=env.target_pose,
        )
    elif task_name == "hammer":
        env = HammerEnv(frame_skip=frame_skip)
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            frame_skip=frame_skip,
            init_obj_pos=env.init_pose,
            init_target_pos=env.target_pose,
        )
    elif task_name == "mug_flip":
        env = MugFlipEnv(frame_skip=frame_skip, object_scale=1)
        env.reset_env()
        env_dict = dict(
            task_name=task_name,
            frame_skip=frame_skip,
            object_scale=1,
            init_obj_pos=env.init_pose,
        )
    else:
        raise NotImplementedError

    # print(env.manipulated_object.get_pose())
    # root_data_path = "/home/sim/data/teleop"
    # path = Path.home() / "data" / "teleop" / "hci" / operator
    # path = path / f"{task_name}-{robot_name}-{num_test}.pickle"
    path = "./sim/raw_data/{}/{}.pickle".format(folder_name, demo_name)
    # path = "./test_demo.pickle"

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
    robot_arm_control_params = (
        np.array([60000, 1000, 2000]) * arm_scale
    )  # This PD is far larger than real to improve stability
    root_translation_control_params = np.array([10000, 1000, 2000]) * trans_scale
    root_rotation_control_params = np.array([2000, 200, 400]) * rot_scale
    finger_control_params = np.array([2000, 60, 10]) * fing_scale
    if robot_name == "mano":
        raise NotImplementedError
    elif "free" in robot_name:
        raise NotImplementedError
    else:
        if "left" in hand_mode:
            robot_name = "{}_left".format(robot_name)
        robot = load_robot(env.scene, robot_name)
        qpos = [
            0,
            (-45 / 180) * np.pi,
            0,
            0,
            (45 / 180) * np.pi,
            (-90 / 180) * np.pi,
        ] + [0] * 16
        if task_name == "dclaw":
            qpos = [
                0,
                (20 / 180) * np.pi,
                -(85 / 180) * np.pi,
                0,
                (112 / 180) * np.pi,
                -np.pi / 2,
            ] + [0] * 16
        qpos = np.array(qpos)
        robot.set_qpos(qpos)
        robot.set_drive_target(qpos)

        wrist = [link for link in robot.get_links() if link.get_name() == "wrist"]
        wrist_pose = wrist[0].get_pose()

        # Robot
        if "free" in robot_name:
            env_init_pos = np.array([-0.3, 0, 0.2])
            for joint in robot.get_active_joints():
                name = joint.get_name()
                if "x_joint" in name or "y_joint" in name or "z_joint" in name:
                    joint.set_drive_property(
                        *(weight * root_translation_control_params), mode="acceleration"
                    )
                elif (
                    "x_rotation_joint" in name
                    or "y_rotation_joint" in name
                    or "z_rotation_joint" in name
                ):
                    joint.set_drive_property(
                        *(weight * root_rotation_control_params), mode="acceleration"
                    )
                else:
                    joint.set_drive_property(
                        *(weight * finger_control_params), mode="acceleration"
                    )
            robot.set_pose(
                sapien.Pose(
                    env_init_pos, transforms3d.euler.euler2quat(0, np.pi / 2, 0)
                )
            )
        elif "xarm" in robot_name:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
            for joint in robot.get_active_joints():
                name = joint.get_name()
                if name in arm_joint_names:
                    joint.set_drive_property(
                        *(weight * robot_arm_control_params), mode="force"
                    )
                else:
                    joint.set_drive_property(
                        *(weight * finger_control_params), mode="force"
                    )
            robot.set_pose(
                sapien.Pose(
                    np.array([-0.55, 0, 0.00855]),
                    transforms3d.euler.euler2quat(0, 0, 0),
                )
            )

    # Robot initial state
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    init_qpos = robot.get_qpos()
    init_ee_pose = np.concatenate((wrist_pose.p, wrist_pose.q))

    # Teleop Client
    client = TeleopClient(port=5500, cmd_dim=robot.dof, host="localhost")
    client.send_init_config(
        init_ee_pose=init_ee_pose,
        robot_base_pose=np.array([0, 0, 0, 1, 0, 0, 0]),
        init_qpos=init_qpos,
        joint_names=joint_names,
    )

    env.seed(int(demo_index))
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer)
    if task_name == "dclaw":
        for name, params in DEFAULT_ROTATING_CAMERAS.items():
            gui.create_camera(**params)
    else:
        for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
            gui.create_camera(**params)

    if task_name == "table_door":
        gui.viewer.set_camera_xyz(-0.6, 0, 0.6)
        gui.viewer.set_camera_rpy(0, -np.pi / 6, 0)
    else:
        gui.viewer.set_camera_xyz(-0.6, 0, 0.6)
        gui.viewer.set_camera_rpy(0, -np.pi / 6, 0)

    # Renderer
    scene = env.scene
    # viz_mat_hand_init = gui.context.create_material(np.array([0, 0, 0, 0]), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8,
    #                                                 0)
    # viz_mat_hand_init = create_visual_material(env.renderer, 0.0, 0.8, 0, np.array([0.96, 0.75, 0.69, 1]))
    # viz_mat_hand_init = env.renderer.create_material()

    # Recorder
    # os.makedirs(str(path.parent), exist_ok=True)
    # recorder = DataRecorder(filename=str(path.resolve()), scene=scene)
    recorder = DataRecorder(filename=path, scene=scene)

    # Init
    # if hand_mode == 'right_hand':
    #     env_init_ori = transforms3d.euler.euler2quat(0, np.pi / 2, 0)
    # else:
    #     env_init_ori = transforms3d.euler.euler2quat(0, np.pi / 2, np.pi)
    # env_init_pos = np.zeros(3)
    scene.step()
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=0.03, pose=sapien.Pose())
    end_effector = builder.build_static("end_effector")
    set_entity_color([end_effector], [0, 1, 0, 0.6])
    is_initialized = False

    duration = 1 / 30.0
    try:
        while not gui.closed:
            tic = time.time()
            for _ in range(frame_skip):
                scene.step()
            gui.render()

            if not is_initialized:
                is_initialized = True
                client.wait_for_server_start()
            else:
                ee_pose = client.get_ee_pose()
                sapien_ee_pose = sapien.Pose(ee_pose[:3], ee_pose[3:])
                end_effector.set_pose(robot.get_pose() * sapien_ee_pose)
                robot_qpos = client.get_teleop_cmd()
                robot.set_drive_target(robot_qpos)

                tac = time.time()
                if tac - tic < duration:
                    sleep_time = duration - (tac - tic)
                    time.sleep(sleep_time)
                    # # Data recording
                record_data = {"robot_qpos": robot_qpos}
                recorder.step(record_data)

    except KeyboardInterrupt:
        print("User interrupt")
        print(len(recorder.data_list))
        meta_data = dict(
            env_class=env.__class__.__name__,
            env_kwargs=env_dict,
            operator=operator,
            robot_name=robot_name,
            hand_mode=hand_mode,
        )
        if "free" in robot_name:
            meta_data["finger_control_params"] = weight * finger_control_params
            meta_data["root_rotation_control_params"] = (
                weight * root_rotation_control_params
            )
            meta_data["root_translation_control_params"] = (
                weight * root_translation_control_params
            )
        elif robot_name != "mano":
            meta_data["finger_control_params"] = weight * finger_control_params
            meta_data["robot_arm_control_params"] = weight * robot_arm_control_params
        recorder.dump(meta_data)


if __name__ == "__main__":
    main()
