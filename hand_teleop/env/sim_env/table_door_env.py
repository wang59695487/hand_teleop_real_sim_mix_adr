import numpy as np
import sapien.core as sapien
import transforms3d
from typing import Dict, Any, Optional, List

from hand_teleop.env.sim_env.base import BaseSimulationEnv


class TableDoorEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, friction=1, init_obj_pos: Optional[sapien.Pose] = None, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        # Dynamics info
        if init_obj_pos is None:
            print('Randomizing Object Location')
            self.init_pose = self.generate_random_object_pose()
        else:
            print('Using Given Object Location')
            self.init_pose = init_obj_pos

        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.friction = friction

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])
        self.table_door = self.load_door()
        self.table_door.set_pose(self.init_pose)
        self.table_door.set_qpos(np.zeros(self.table_door.dof))

    def generate_random_object_pose(self):
        # if randomness_scale == 1:
        #     random_pose = sapien.Pose([-0.05, 0.2, 0.1])
        # elif randomness_scale == 2:
        #     random_pose = sapien.Pose([0.05, 0.1, 0.1])
        # elif randomness_scale == 3:
        #     random_pose = sapien.Pose([-0.15, -0.2, 0.1])
        # elif randomness_scale == 4:
        #     random_pose = sapien.Pose([-0.2, -0.1, 0.1])
        # elif randomness_scale == 5:
        #     random_pose = sapien.Pose([0.1, 0.1, 0.1])
        # elif randomness_scale == 6:
        #     random_pose = sapien.Pose([0.15, 0.15, 0.1])
        # Broad Random
        # pos_x = self.np_random.uniform(low=-0.2, high=0.2) * randomness_scale
        # pos_y = self.np_random.uniform(low=0.05, high=0.2) * randomness_scale
        # Small Random
        pos_x = self.np_random.uniform(low=0.05, high=0.15)
        pos_y = self.np_random.uniform(low=-0.05, high=0.05)
        random_pose = sapien.Pose([pos_x, pos_y, 0.01])
        return random_pose

    def load_door(self):
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()

        root.set_name("frame")
        table_physics_mat = self.scene.create_physical_material(1.0 * self.friction, 0.5 * self.friction, 0.01)
        root.add_capsule_collision(pose=sapien.Pose([0, 0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                   radius=0.025, half_length=0.175)
        root.add_capsule_collision(pose=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                   radius=0.025, half_length=0.175)

        door = builder.create_link_builder(root)
        door.set_name("door")
        door.add_box_collision(pose=sapien.Pose([0, 0, 0.2]), half_size=[0.025, 0.15, 0.2], density=100,
                               material=table_physics_mat)
        door.set_joint_properties(
            'revolute',
            limits=[[0, np.pi / 2]],
            pose_in_parent=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
            pose_in_child=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
            friction=0.5,
            damping=0.2,
        )

        hinge = builder.create_link_builder(door)
        hinge.add_capsule_collision(pose=sapien.Pose([-0.02, 0.10, 0.2]), radius=0.015, half_length=0.07,
                                    material=table_physics_mat)
        hinge.add_capsule_collision(pose=sapien.Pose([-0.09, 0.05, 0.2], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.05, material=table_physics_mat)
        hinge.add_capsule_collision(pose=sapien.Pose([0.05, 0.14, 0.2], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.04, material=table_physics_mat)
        hinge.set_joint_properties(
            'revolute',
            limits=[[0, np.pi / 2]],
            pose_in_parent=sapien.Pose([0, 0.10, 0.2]),
            pose_in_child=sapien.Pose([0, 0.10, 0.2]),
            friction=1,
        )
        hinge.set_name("handle")

        # Visual
        # if self.use_gui:
        frame_viz_mat = self.renderer.create_material()
        frame_viz_mat.set_base_color([0.25, 0.0, 0.0, 1])
        door_viz_mat = self.renderer.create_material()
        door_viz_mat.set_base_color([0.8, 0.3, 0.0, 1])
        door_viz_mat.set_roughness(0.2)
        root.add_capsule_visual(pose=sapien.Pose([0, 0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                radius=0.025, half_length=0.175, material=frame_viz_mat)
        root.add_capsule_visual(pose=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                radius=0.025, half_length=0.175, material=frame_viz_mat)
        door.add_box_visual(pose=sapien.Pose([0, 0, 0.2]), half_size=[0.025, 0.15, 0.2], material=door_viz_mat)
        hinge.add_capsule_visual(pose=sapien.Pose([-0.02, 0.10, 0.2]), radius=0.015, half_length=0.07)
        hinge.add_capsule_visual(pose=sapien.Pose([-0.09, 0.05, 0.2], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.05)
        hinge.add_capsule_visual(pose=sapien.Pose([0.05, 0.14, 0.2], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.04)

        door = builder.build(fix_root_link=True)
        # for joint in door.get_joints():
        #     joint.set_drive_property(200, 50, 10, mode="acceleration")
        return door

    def reset_env(self):
        self.table_door.set_qpos(np.zeros(self.table_door.dof))
        self.table_door.set_pose(self.init_pose)


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = TableDoorEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    # env.table_door.set_qpos([0, np.pi / 2])

    while not viewer.closed:
        # env.table_door.get_links()[1].add_force_torque(np.zeros(3), np.array([0, 0, 1]))
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
