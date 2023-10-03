from pathlib import Path

import numpy as np
import sapien.core as sapien

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from env.constructor import download_maniskill
from utils.ycb_object_utils import load_ycb_object


class DrawerFetchObjectEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, object_name="tomato_soup_can", drawer_id=1000, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        self.object_name = object_name
        self.partnet_mobility_id = drawer_id

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, -0.4]))
        self.drawer = self.load_drawer()
        self.drawer.set_pose(sapien.Pose([0, 1, -0.6]))

        # Load object
        self.target_object = load_ycb_object(self.scene, object_name)
        self.target_object.set_pose(sapien.Pose([0, 0, -0.35]))

    def load_drawer(self):
        urdf = download_maniskill(
            self.partnet_mobility_id,
            directory=None
        )
        vhacd_urdf = Path(urdf).parent.joinpath('mobility_fixed.urdf')
        if vhacd_urdf.exists():
            urdf = str(vhacd_urdf)

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.6
        loader.fix_root_link = True
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        drawer = loader.load(urdf, config=config)
        return drawer

    def create_table(self, table_height=1.0, table_half_size=(0.8, 0.8, 0.025)):
        builder = self.scene.create_actor_builder()

        # Top
        top_pose = sapien.Pose([0, 0, -table_half_size[2]])
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        if self.renderer:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_diffuse_texture_from_file(
                "assets/misc/table_map.jpg")
            table_visual_material.set_roughness(0.3)
            builder.add_visual_from_file("assets/misc/cube.obj",
                                         pose=top_pose, material=table_visual_material, scale=table_half_size)
            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1
            builder.add_box_visual(pose=sapien.Pose([x, y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([x, -y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([-x, y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([-x, -y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
        return builder.build_static("table")


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = DrawerFetchObjectEnv(drawer_id="1005")
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
