import numpy as np
import sapien.core as sapien
from typing import Dict, Any, Optional, List

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.model_utils import create_visual_material


class InsertObjectEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, init_obj_pos: Optional[sapien.Pose] = None, init_target_pos: Optional[sapien.Pose] = None, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, 0]))

        self.box, self.manipulated_object = self.load_block_and_box()
        if init_obj_pos is None:
            print('Randomizing Object Location')
            self.init_pose = sapien.Pose([0, -0.1, 0.0])
        else:
            print('Using Given Object Location')
            self.init_pose = init_obj_pos
        if init_target_pos is None:
            print('Randomizing Target Location')
            self.target_pose = self.generate_random_target_pose()
        else:
            print('Using Given Target Location')
            self.target_pose = init_target_pos        
        self.box.set_pose(self.target_pose)
        self.manipulated_object.set_pose(self.init_pose)

    def generate_random_target_pose(self):
        pos_x = self.np_random.uniform(low=-0.05, high=0.05)
        pos_y = self.np_random.uniform(low=0.15, high=0.25)
        random_pose = sapien.Pose([pos_x, pos_y, 0.0])
        return random_pose

    def load_block_and_box(self):
        density = 400
        # build box
        viz_mat = create_visual_material(self.renderer, 0.2, 0.8, 0.4,
                                         base_color=np.array([140, 138, 132, 255]) / 255)
        physical_mat = self.scene.create_physical_material(1.5, 1, 0)
        builder = self.scene.create_actor_builder()
        top_half_size = np.array([0.042, 0.042, 0.07])
        half_width = top_half_size[0] + 0.005
        half_thickness = 0.003
        half_driver_height = top_half_size[2] - 0.005
        extend = half_thickness + half_width
        pos_list = [(extend, 0, half_driver_height), (-extend, 0, half_driver_height),
                    (0, extend, half_driver_height), (0, -extend, half_driver_height)]
        half_size_list = [
            (half_thickness, extend + half_thickness, half_driver_height),
            (half_thickness, extend + half_thickness, half_driver_height),
            (half_width, half_thickness, half_driver_height),
            (half_width, half_thickness, half_driver_height)
        ]
        for i in range(4):
            builder.add_box_visual(sapien.Pose(np.array(pos_list[i])), half_size_list[i], material=viz_mat)
            builder.add_box_collision(sapien.Pose(np.array(pos_list[i])), half_size_list[i], material=physical_mat,
                                      density=density * 1000)
        
        box = builder.build("box")

        # build block
        top_half_size = np.array([0.03, 0.03, 0.08])
        viz_mat_block = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                              base_color=np.array([80, 10, 74, 255]) / 255)
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=top_half_size, material=viz_mat_block)
        builder.add_box_collision(half_size=top_half_size, material=physical_mat, density=density*2)
        
        block = builder.build("block")

        return box, block

    def reset_env(self):
        # pose = self.generate_random_object_pose(self.randomness_scale)
        # self.manipulated_object.set_pose(pose)
        self.manipulated_object.set_pose(self.init_pose)

        # pose = self.generate_random_target_pose(self.randomness_scale)
        self.box.set_pose(self.target_pose)
        # self.target_pose = pose


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = InsertObjectEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()