from random import random
from typing import Dict, Any, Optional, List
import numpy as np
import random
import sapien.core as sapien
import transforms3d

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.real_world import lab
from hand_teleop.utils.render_scene_utils import set_entity_color
from hand_teleop.utils.ycb_object_utils import (
    load_ycb_object,
    YCB_SIZE,
    YCB_ORIENTATION,
)
from hand_teleop.utils.egad_object_utils import load_egad_object, EGAD_NAME
from hand_teleop.utils.shapenet_object_utils import load_shapenet_object, COLOR_LIST


class PickPlaceEnv(BaseSimulationEnv):
    def __init__(
        self,
        use_gui=True,
        frame_skip=5,
        object_category="YCB",
        object_name="tomato_soup_can",
        object_seed=0,
        object_scale=1.0,
        randomness_scale=1,
        friction=1,
        use_visual_obs=False,
        init_obj_pos: Optional[sapien.Pose] = None,
        init_target_pos: Optional[sapien.Pose] = None,
        **renderer_kwargs
    ):
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            use_visual_obs=use_visual_obs,
            **renderer_kwargs
        )

        # Object info
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = object_scale
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
        # self.tables = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])

        # Load box/plate
        # self.box = self.load_partnet_obj(100426, scale = 0.3, material_spec=[1,1,0], density=100, fix_root=False)
        # self.box.set_pose(sapien.Pose([0, 0, 0.1]))
        self.target_object = load_ycb_object(self.scene, "plate", static=True)
        if init_target_pos is None:
            print("Randomizing Target Location")
            self.target_pose = self.generate_random_target_pose(
                randomness_scale)
        else:
            print("Using Given Target Location")
            self.target_pose = init_target_pos
        self.target_object.set_pose(self.target_pose)

        # Load object
        if self.object_category.lower() == "ycb":
            self.object_height = object_scale * \
                YCB_SIZE[self.object_name][2] / 2
            self.manipulated_object = load_ycb_object(self.scene, object_name)

        elif self.object_category.lower() == "shape_net":
            # Load bottle
            object_class = self.object_name.split("_")[0]
            object_id = self.object_name.split("_")[1]
            if object_class == "bottle":
                cat_id = "02876657"
            self.manipulated_object, self.object_height = load_shapenet_object(
                self.scene, cat_id, object_id)

        else:
            raise NotImplementedError
        self.manipulated_object.set_pose(self.init_pose)

        self.generate_random_object_texture(randomness_scale)

    def generate_random_object_pose(self, randomness_scale):
        random.seed(self.object_seed)
        pos_x = random.uniform(-0.1, 0.1) * randomness_scale
        pos_y = random.uniform(0.2, 0.3) * randomness_scale
        position = np.array([pos_x, pos_y, 0.1])
        # euler = self.np_random.uniform(low=np.deg2rad(15), high=np.deg2rad(25))
        if self.object_name != "sugar_box":
            euler = random.uniform(np.deg2rad(15), np.deg2rad(25))
        else:
            euler = random.uniform(np.deg2rad(80), np.deg2rad(90))
        orientation = transforms3d.euler.euler2quat(0, 0, euler)
        random_pose = sapien.Pose(position, orientation)
        return random_pose

    def generate_random_target_pose(self, randomness_scale):
        pos_x = self.np_random.uniform(low=-0.15, high=0.15) * randomness_scale
        pos_y = self.np_random.uniform(low=-0.12, high=-0.2) * randomness_scale
        random_pose = sapien.Pose([pos_x, pos_y, 0.1])
        random_pose = sapien.Pose([-0.005, -0.12, 0])

        return random_pose

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

        for table in self.tables:
            for visual in table.get_visual_bodies():
                for geom in visual.get_render_shapes():
                    mat = geom.material
                    mat.set_specular(random.uniform(0, var))
                    mat.set_roughness(random.uniform(0.7 - var, 0.7 + var))
                    mat.set_metallic(random.uniform(0, var))
                    geom.set_material(mat)

        for visual in self.manipulated_object.get_visual_bodies():
            for geom in visual.get_render_shapes():
                mat = geom.material
                if self.object_category.lower() == "shape_net":
                    color = COLOR_LIST[np.random.randint(
                        0, len(COLOR_LIST))] * random.uniform(var, 1)
                    color[-1] = 1
                    mat.set_base_color(color)
                mat.set_specular(random.uniform(0, var))
                mat.set_roughness(random.uniform(0.7 - var, 0.7 + var))
                mat.set_metallic(random.uniform(0, var))
                geom.set_material(mat)

    def reset_env(self):
        self.manipulated_object.set_pose(self.init_pose)
        self.target_object.set_pose(self.target_pose)

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


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light

    randomness_scale = 1
    env = PickPlaceEnv(
        object_name="tomato_soup_can",
        object_category="YCB",
        randomness_scale=randomness_scale,
        object_scale=0.8,
    )
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    env.reset_env()
    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == "__main__":
    env_test()
