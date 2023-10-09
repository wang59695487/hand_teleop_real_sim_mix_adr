import random
from pathlib import Path
from typing import Optional

import numpy as np
import sapien.core as sapien

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.real_world import lab


class DClawEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_name="dclaw_3x", object_seed=0,
                 object_scale=1.0, randomness_scale=1, friction=1, use_visual_obs=False,
                 init_obj_pos: Optional[sapien.Pose] = None, init_target_pos: Optional[sapien.Pose] = None,
                 **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        # Object info
        self.object_name = object_name
        self.object_scale = object_scale
        self.object_height = 0
        self.object_seed = object_seed

        # Dynamics info
        if init_obj_pos is None:
            self.init_pose = self.generate_random_object_pose(randomness_scale)
            print('Randomizing Object Location')
            # print('Object Seed', self.object_seed)
            # print('Object Scale', self.object_scale)
            # print('Object Height', self.object_height)
            print('Object Location', self.init_pose)
        else:
            print('Using Given Object Location')
            self.init_pose = init_obj_pos
        self.friction = friction

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.002)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        # self.tables = self.create_lab_tables(table_height=0.73)
        self.tables = self.create_lab_tables(table_height=0.91)

        asset_path = Path(__file__).parent.parent.parent.parent / "assets"
        if "3x" in self.object_name:
            urdf_path = asset_path / "robel" / "dclaw_3x.urdf"
        else:
            raise NotImplementedError

        loader = self.scene.create_urdf_loader()
        loader.scale = self.object_scale / 2
        loader.load_multiple_collisions_from_file = True
        builder = loader.load_file_as_articulation_builder(str(urdf_path))
        self.manipulated_object = builder.build(fix_root_link=True)
        rotating_joint = self.manipulated_object.get_active_joints()[0]
        rotating_joint.set_drive_property(0, 0.1)        
        self.generate_random_object_texture(2)

    def generate_random_object_pose(self, randomness_scale):
        random.seed(self.object_seed)
        pos_x = random.uniform(-0.05, 0.05) * randomness_scale
        pos_y = random.uniform(-0.05, 0.05) * randomness_scale
        position = np.array([pos_x, pos_y, 0.0])
        position = np.array([0, 0, 0.0])
        random_pose = sapien.Pose(position, [0.707, 0, 0, 0.707] )
        return random_pose
    
    def generate_random_object_texture(self, randomness_scale):
        var = 0.1 * randomness_scale
        for link in self.manipulated_object.get_links():
            if link.get_name() == "dclaw_up":
                default_color = np.array([1, 1, 1, 1])
            elif link.get_name() == "dclaw_down":
                default_color = np.array([1, 0, 0, 1])

            for visual in link.get_visual_bodies():
                if link.get_name() == "dclaw_up":
                    continue
                for geom in visual.get_render_shapes():
                    mat = geom.material
                    mat.set_base_color(default_color)
                    mat.set_specular(random.uniform(0, var))
                    mat.set_roughness(random.uniform(0.7-var, 0.7+var))
                    mat.set_metallic(random.uniform(0, var))
                    geom.set_material(mat)
                    
        for table in self.tables:
            for visual in table.get_visual_bodies():
                for geom in visual.get_render_shapes():
                    mat = geom.material
                    mat.set_specular(random.uniform(0, var))
                    mat.set_roughness(random.uniform(0.7-var, 0.7+var))
                    mat.set_metallic(random.uniform(0, var))
                    geom.set_material(mat)

    def reset_env(self):
        self.manipulated_object.set_pose(self.init_pose)

    def create_lab_tables(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.03

        # Top
        top_pose = sapien.Pose(np.array([lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        table_half_size = np.concatenate([lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)

            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1

            builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg0")
            builder.add_box_visual(pose=sapien.Pose([x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg1")
            builder.add_box_visual(pose=sapien.Pose([-x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg2")
            builder.add_box_visual(pose=sapien.Pose([-x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg3")
        object_table = builder.build_static("object_table")

        # Build robot table
        robot_table_thickness = 0.025
        table_half_size = np.concatenate([lab.ROBOT_TABLE_XY_SIZE / 2, [robot_table_thickness / 2]])
        robot_table_offset = -lab.DESK2ROBOT_Z_AXIS
        table_height += robot_table_offset
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose(
            np.array([lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
                      lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
                      -robot_table_thickness / 2 + robot_table_offset]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.2)
            table_visual_material.set_base_color(np.array([0, 0, 0, 255]) / 255)
            table_visual_material.set_roughness(0.1)
            builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
        robot_table = builder.build_static("robot_table")
        return object_table, robot_table


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    randomness_scale = 1
    env = DClawEnv(object_name="dclaw_3x", randomness_scale=randomness_scale, object_scale=0.8)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    env.reset_env()
    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
