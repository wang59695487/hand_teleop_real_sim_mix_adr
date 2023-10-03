from pathlib import Path
from typing import Tuple

import numpy as np

from hand_teleop.retargeting.kinematics.optimizer import VectorOptimizer, DexPilotAllegroIOptimizer
from hand_teleop.retargeting.kinematics.optimizer_utils import SAPIENKinematicsModelStandalone
from hand_teleop.retargeting.kinematics.seq_retarget import SeqRetargeting

_GLOBAL_SAPIEN_PTR = []


def build_default_retargeting(asset_path: str, robot_name: str, retargeting_type="vector") -> Tuple[SeqRetargeting, np.ndarray, Path]:
    robot_name = robot_name.lower()
    retargeting_type = retargeting_type.lower()
    supported_retargeting_types = ["vector", "dexpilot"]
    if retargeting_type not in supported_retargeting_types:
        raise ValueError(f"Retargeting type {retargeting_type} is not supported")

    asset_path = asset_path / "robot"

    # Build retargeting Optimizer
    if robot_name == "allegro_hand_free":
        urdf_path = asset_path / "allegro_hand_description" / "allegro_hand_free.urdf"
    elif robot_name == "allegro_hand_free_left":
        urdf_path = asset_path / "allegro_hand_description" / "allegro_hand_free_left.urdf"
    elif robot_name == "allegro_hand_xarm6":
        urdf_path = asset_path / "xarm6_description" / "xarm6_allegro.urdf"
    elif robot_name == "allegro_hand_xarm6_wrist_mounted_face_down" or robot_name == "allegro_hand_xarm6_wrist_mounted_face_front":
        urdf_path = asset_path / "xarm6_description" / "xarm6_allegro_wrist_mounted_rotate.urdf"
    elif robot_name == "allegro_hand_xarm6_wrist_mounted_face_down_left" or robot_name == "allegro_hand_xarm6_wrist_mounted_face_front_left":
        urdf_path = asset_path / "xarm6_description" / "xarm6_allegro_wrist_mounted_rotate_left.urdf"
    elif robot_name == "allegro_biotac_hand":
        urdf_path = asset_path / "allegro" / ""
    elif robot_name == "adroit_hand":
        urdf_path = asset_path / "robot/adroit_hand_retargeting.urdf"
    elif robot_name == "svh_hand":
        urdf_path = asset_path / "robot/svh_hand_retargeting.urdf"
    elif robot_name == "ar10_hand":
        urdf_path = asset_path / "robot/ar10_hand_retargeting.urdf"
    elif robot_name == "dlr_hand":
        urdf_path = asset_path / "robot/dlr_hand_retargeting.urdf"
    elif robot_name == "panda_gripper_hand":
        urdf_path = asset_path / "robot/panda_hand_retargeting.urdf"
    else:
        raise NotImplementedError

    sapien_model = SAPIENKinematicsModelStandalone(str(urdf_path), add_dummy_rotation=False)
    robot = sapien_model.robot
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]

    # if robot_name == "allegro_hand" and retargeting_type == "dexpilot":
    #     optimizer = DexPilotAllegroIOptimizer(robot, joint_names)
    #     link_hand_indices = DexPilotAllegroIOptimizer.MANO_INDEX
    if (robot_name == "allegro_hand_free" or robot_name=="allegro_hand_free_left") and retargeting_type == "vector":
        origin_link_names = ["palm"] * 4
        # task_link_names = ["thumb_link_3_tip", "index_link_3_tip", "middle_link_3_tip", "ring_link_3_tip"]
        task_link_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.6)
        link_hand_indices = np.array([[0, 0, 0, 0], [4, 8, 12, 16]])
    elif ("allegro_hand_xarm" in robot_name) and retargeting_type == "vector":
        origin_link_names = ["palm"] * 4
        # task_link_names = ["thumb_link_3_tip", "index_link_3_tip", "middle_link_3_tip", "ring_link_3_tip"]
        task_link_names = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.6)
        link_hand_indices = np.array([[0, 0, 0, 0], [4, 8, 12, 16]])
    elif robot_name == "adroit_hand" and retargeting_type == "vector":
        origin_link_names = ["palm"] * 5
        task_link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.2)
        link_hand_indices = np.array([[0, 0, 0, 0, 0], [4, 8, 12, 16, 20]])
    elif robot_name == "svh_hand" and retargeting_type == "vector":
        origin_link_names = ["right_hand_base_link"] * 5
        task_link_names = ["right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.2)
        link_hand_indices = np.array([[0, 0, 0, 0, 0], [4, 8, 12, 16, 20]])
    elif robot_name == "ar10_hand" and retargeting_type == "vector":
        origin_link_names = ["palm"] * 5
        task_link_names = ["thumbtip", "fingertip4", "fingertip3", "fingertip2", "fingertip1"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.2)
        link_hand_indices = np.array([[0, 0, 0, 0, 0], [4, 8, 12, 16, 20]])
    elif robot_name == "dlr_hand" and retargeting_type == "vector":
        origin_link_names = ["right_palm_link"] * 5
        task_link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1.3)
        link_hand_indices = np.array([[0, 0, 0, 0, 0], [4, 8, 12, 16, 20]])
    elif robot_name == "panda_gripper" and retargeting_type == "vector":
        origin_link_names = ["left_grasp_site"]
        task_link_names = ["right_grasp_site"]
        optimizer = VectorOptimizer(robot, joint_names, origin_link_names=origin_link_names,
                                    task_link_names=task_link_names, scaling=1)
        link_hand_indices = np.array([[4], [8]])
    else:
        raise NotImplementedError

    retargeting = SeqRetargeting(optimizer, has_joint_limits=True)
    _GLOBAL_SAPIEN_PTR.append(sapien_model)

    return retargeting, link_hand_indices
