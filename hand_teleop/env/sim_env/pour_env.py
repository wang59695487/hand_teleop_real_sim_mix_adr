from pathlib import Path
from typing import Optional
import random

import numpy as np
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.real_world import lab
from hand_teleop.utils.ycb_object_utils import load_ycb_object, YCB_SIZE


class PourBoxEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui=True,
        frame_skip=5,
        object_seed=0,
        object_name="chip_can",
        randomness_scale=1,
        friction=1,
        use_visual_obs=False,
        init_obj_pos: Optional[sapien.Pose] = None,
        init_target_pos: Optional[sapien.Pose] = None,
        **renderer_kwargs,
    ):
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            use_visual_obs=use_visual_obs,
            **renderer_kwargs,
        )

        # Object info
        self.bottle_name = object_name
        self.bottle_scale = 0.0066
        self.bottle_height = 17.7909 * self.bottle_scale
        self.bowl_name = "bowl"
        self.bowl_height = YCB_SIZE[self.bowl_name][2] / 2
        self.object_seed = object_seed

        # Dynamics info
        if init_obj_pos is None:
            self.init_pose = self.generate_random_object_pose(randomness_scale)
            print("Randomizing Object Location")
            print("Object Location", self.init_pose)
        else:
            print("Using Given Object Location")
            self.init_pose = init_obj_pos
        self.friction = friction
        self.randomness_scale = randomness_scale

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera(
                "init_not_used", width=10, height=10, fovy=1, near=0.1, far=1
            )
            self.scene.remove_camera(cam)

        # Load table
        self.tables = self.create_lab_tables(table_height=0.91)

        # Load bowl
        if init_target_pos is None:
            self.target_pose = sapien.Pose([0.0, 0.2, self.bowl_height])
        else:
            print("Using Given Target Location")
            self.target_pose = init_target_pos
        self.target_object = load_ycb_object(
            self.scene, self.bowl_name, static=True)
        self.target_object.set_pose(self.target_pose)

        # Load box
        self.boxes = self.load_box()
        for box in self.boxes:
            box.set_pose(self.init_pose)

        # Load bottle
        bottle_material = self.scene.create_physical_material(0.5, 0.3, 0.01)
        bottle_dir = Path(__file__).parent.parent.parent.parent / \
            "assets/misc/chip_can"
        visual_mesh = bottle_dir / "pringles.obj"
        collision_mesh = bottle_dir / "pringles_collision.obj"
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            str(visual_mesh),
            scale=[self.bottle_scale] * 3,
            pose=sapien.Pose(q=[0.707, 0.707, 0, 0]),
        )
        builder.add_multiple_collisions_from_file(
            str(collision_mesh),
            scale=[self.bottle_scale] * 3,
            pose=sapien.Pose(q=[0.707, 0.707, 0, 0]),
            material=bottle_material,
        )
        self.manipulated_object = builder.build(self.bottle_name)

        print('################################Randomizing Object Texture##########################')
        self.generate_random_object_texture(randomness_scale)

    def generate_random_object_pose(self, randomness_scale):
        random.seed(self.object_seed)
        pos_x = random.uniform(-0.1, 0.1) * randomness_scale
        pos_y = random.uniform(-0.18, -0.08) * randomness_scale
        position = np.array([pos_x, pos_y, 0])
        euler = random.uniform(np.deg2rad(0), np.deg2rad(300))
        orientation = transforms3d.euler.euler2quat(0, 0, euler)
        return sapien.Pose(position, orientation)

    def generate_random_object_texture(self, randomness_scale):
        var = 0.1 * randomness_scale
        default_color = np.array([1, 0, 0, 1])
        for visual in self.target_object.get_visual_bodies():
            for geom in visual.get_render_shapes():
                mat = geom.material
                mat.set_base_color(default_color)
                mat.set_specular(random.uniform(0, var))
                mat.set_roughness(random.uniform(0.7 - var, 0.7 + var))
                mat.set_metallic(random.uniform(0, var))
                geom.set_material(mat)

        for visual in self.manipulated_object.get_visual_bodies():
            for geom in visual.get_render_shapes():
                mat = geom.material
                mat.set_specular(random.uniform(0, var))
                mat.set_roughness(random.uniform(0.7 - var, 0.7 + var))
                mat.set_metallic(random.uniform(0, var))
                geom.set_material(mat)

        for table in self.tables:
            for visual in table.get_visual_bodies():
                for geom in visual.get_render_shapes():
                    mat = geom.material
                    mat.set_specular(random.uniform(0, var))
                    mat.set_roughness(random.uniform(0.7 - var, 0.7 + var))
                    mat.set_metallic(random.uniform(0, var))
                    geom.set_material(mat)

    def reset_env(self):
        init_pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(init_pose)
        self.target_object.set_pose(self.target_pose)

        print(init_pose)
        for box in self.boxes:
            box.set_pose(init_pose)

    def create_lab_tables(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.03

        # Top
        top_pose = sapien.Pose(
            np.array(
                [lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2])
        )
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        table_half_size = np.concatenate(
            [lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])
        builder.add_box_collision(
            pose=top_pose, half_size=table_half_size, material=top_material
        )
        # Leg
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)

            leg_size = np.array(
                [0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1

            builder.add_box_visual(
                pose=top_pose, half_size=table_half_size, material=table_visual_material
            )
            builder.add_box_visual(
                pose=sapien.Pose([x, y + lab.TABLE_ORIGIN[1], leg_height]),
                half_size=leg_size,
                material=table_visual_material,
                name="leg0",
            )
            builder.add_box_visual(
                pose=sapien.Pose([x, -y + lab.TABLE_ORIGIN[1], leg_height]),
                half_size=leg_size,
                material=table_visual_material,
                name="leg1",
            )
            builder.add_box_visual(
                pose=sapien.Pose([-x, y + lab.TABLE_ORIGIN[1], leg_height]),
                half_size=leg_size,
                material=table_visual_material,
                name="leg2",
            )
            builder.add_box_visual(
                pose=sapien.Pose([-x, -y + lab.TABLE_ORIGIN[1], leg_height]),
                half_size=leg_size,
                material=table_visual_material,
                name="leg3",
            )
        object_table = builder.build_static("object_table")

        # Build robot table
        robot_table_thickness = 0.025
        table_half_size = np.concatenate(
            [lab.ROBOT_TABLE_XY_SIZE / 2, [robot_table_thickness / 2]]
        )
        robot_table_offset = -lab.DESK2ROBOT_Z_AXIS
        table_height += robot_table_offset
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose(
            np.array(
                [
                    lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
                    lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
                    -robot_table_thickness / 2 + robot_table_offset,
                ]
            )
        )
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(
            pose=top_pose, half_size=table_half_size, material=top_material
        )
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.2)
            table_visual_material.set_base_color(
                np.array([0, 0, 0, 255]) / 255)
            table_visual_material.set_roughness(0.1)
            builder.add_box_visual(
                pose=top_pose, half_size=table_half_size, material=table_visual_material
            )
        robot_table = builder.build_static("robot_table")
        return object_table, robot_table

    def load_box(self):
        box_material = self.scene.create_physical_material(0.1, 0.1, 0.01)
        colors = [
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
            np.array([1, 1, 0, 1]),
        ]
        box_half_len = 0.0125
        box_len = box_half_len * 2
        box_size = np.ones(3) * box_half_len
        boxes = []
        for i, color in enumerate(colors):
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                half_size=box_size,
                color=color,
                pose=sapien.Pose([0, 0, box_len * i + 0.1]),
            )
            builder.add_box_collision(
                half_size=box_size,
                material=box_material,
                pose=sapien.Pose([0, 0, box_len * i + 0.1]),
            )
            boxes.append(builder.build(f"box_{i}"))
        return boxes


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light

    randomness_scale = 1
    env = PourBoxEnv(randomness_scale=randomness_scale)
    env.reset_env()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    viewer.toggle_pause(True)
    viewer.render()
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    env_test()
