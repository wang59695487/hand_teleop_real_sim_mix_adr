from hand_teleop.env.rl_env.pc_processing import process_relocate_pc, add_gaussian_noise
from hand_teleop.real_world import lab
import numpy as np

# Camera config
CAMERA_CONFIG = {
    "relocate": {
        "relocate": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(64, 64),
                         ), },
    "viz_only": {  # only for visualization (human), not for visual observation
        "relocate_viz": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT, fov=np.deg2rad(69.4), resolution=(640, 480),),
        "door_viz": dict(position=np.array([-0.6, -0.3, 0.8]), look_at_dir=np.array([0.6, 0.3, -0.8]),
                         right_dir=np.array([1, -2, 0]), fov=np.deg2rad(69.4), resolution=(640, 480))},
}

# Observation config type
OBS_CONFIG = {
    "relocate": {
        "relocate": {"point_cloud": {"process_fn": process_relocate_pc, "num_points": 256}},
    },
    "relocate_noise": {
        "relocate": {"point_cloud": {"process_fn": process_relocate_pc, "num_points": 256,
                                     "additional_process_fn": [add_gaussian_noise]}, "pose_perturb_level": 1.0},
    }
}

# Imagination config type
IMG_CONFIG = {
    "relocate_goal_only": {
        "goal": {"target_object": 64},
    },
    "relocate_robot_only": {
        "robot": {
            "link_15.0_tip": 8, "link_3.0_tip": 8, "link_7.0_tip": 8, "link_11.0_tip": 8,
            "link_15.0": 8, "link_3.0": 8, "link_7.0": 8, "link_11.0": 8,
            "base_link": 32,
        },
    },
    "relocate_goal_robot": {
        "goal": {"target_object": 64},
        "robot": {
            "link_15.0_tip": 8, "link_3.0_tip": 8, "link_7.0_tip": 8, "link_11.0_tip": 8,
            "link_15.0": 8, "link_3.0": 8, "link_7.0": 8, "link_11.0": 8,
            "base_link": 32,
        },
    },
}
