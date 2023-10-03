from typing import Dict

import numpy as np
import pyrender
import transforms3d
import trimesh.creation
from urdfpy import URDF


def setup_robot_viewer(scene: pyrender.Scene, distance_scale=1):
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    trackball = viewer._trackball
    default_view_pose = trackball.pose
    default_view_pose[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    default_view_pose[:3, 3] = [distance_scale, 0, distance_scale * 0.05]
    trackball._n_pose = default_view_pose
    trackball._pose = trackball._n_pose
    return viewer


class RobotVisualizer:
    def __init__(
            self,
            urdf: str,
            viz_ee_target=False,
            ee_name="",
    ):
        self.viz_ee_target = viz_ee_target

        # Load robot
        self.robot = URDF.load(urdf)
        dof = len(self.robot.actuated_joints)
        joint_names = self.robot.actuated_joint_names
        cfg = {k: v for k, v in zip(joint_names, [0] * dof)}
        fk = self.robot.visual_trimesh_fk(cfg=cfg)
        self.joint_names = joint_names

        # Construct scene
        self.scene = pyrender.Scene()
        self.nodes = []
        for tm in fk:
            pose = fk[tm]
            mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            self.nodes.append(self.scene.add(mesh, pose=pose))
        viewer_distance = 2 if viz_ee_target else 0.5
        self.viewer = setup_robot_viewer(self.scene, viewer_distance)

        if not self.viz_ee_target:
            pass
        else:
            if len(ee_name) == 0:
                raise ValueError(f"Full robot visualization must specify a ee_name.")
            self.ee_name = ee_name

            # Target EE pose
            axis_mesh = trimesh.creation.axis(origin_size=0.03, axis_length=0.1)
            mesh = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)
            self.nodes.append(self.scene.add(mesh, pose=np.eye(4)))

    def update_robot_geometry(self, qpos_dict: Dict[str, np.ndarray]):
        fk = self.robot.visual_trimesh_fk(cfg=qpos_dict)

        self.viewer.render_lock.acquire()
        for index, tm in enumerate(fk):
            pose = fk[tm]
            self.scene.set_pose(self.nodes[index], pose)
        self.viewer.render_lock.release()

    def update_ee_target(self, ee_pose: np.ndarray):
        if not self.viz_ee_target:
            return
        # Compute end effector target pose SE(3) matrix
        ee_rot = transforms3d.quaternions.quat2mat(ee_pose[3:7])
        ee_pose_mat = np.eye(4)
        ee_pose_mat[:3, :3] = ee_rot
        ee_pose_mat[:3, 3] = ee_pose[:3]

        self.viewer.render_lock.acquire()
        self.scene.set_pose(self.nodes[-1], ee_pose_mat)
        self.viewer.render_lock.release()
