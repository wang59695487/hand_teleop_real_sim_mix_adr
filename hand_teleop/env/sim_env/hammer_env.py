import numpy as np
import sapien.core as sapien
import transforms3d
from typing import Dict, Any, Optional, List

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.model_utils import create_visual_material

class HammerEnv(BaseSimulationEnv):
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
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, 0]))
        
        # Load hammer and nail
        self.nail, self.hammer = self.load_hammer_and_nail()
        if init_obj_pos is None:
            print('Randomizing Object Location')
            self.init_pose = self.generate_random_init_pose()
        else:
            print('Using Given Object Location')
            self.init_pose = init_obj_pos        
        if init_target_pos is None:
            print('Randomizing Target Location')
            self.target_pose = sapien.Pose([0, 0.2, 0.0])
        else:
            print('Using Given Target Location')
            self.target_pose = init_target_pos  
        self.nail.set_pose(self.target_pose)
        self.hammer.set_pose(self.init_pose)

    def generate_random_init_pose(self):
        position = np.array([0, -0.1, 0.0])
        random_z_rotate = self.np_random.uniform(np.pi, 5*np.pi/4)
        orientation = transforms3d.euler.euler2quat(0, 0, random_z_rotate)
        pose = sapien.Pose(position, orientation)
        return pose        

    def load_hammer_and_nail(self):
        builder = self.scene.create_articulation_builder()
        root_link = builder.create_link_builder()
        bottom_half_size = np.array([0.05, 0.05, 0.03])
        middle_half_size = np.array([0.003, 0.003, 0.03])
        top_half_size = np.array([0.015, 0.015, 0.002])
        viz_mat_nail = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                              base_color=np.array([80, 72, 74, 255]) / 255)
        viz_mat_wood = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                              base_color=np.array([250, 200, 74, 255]) / 255)
        physical_mat = self.scene.create_physical_material(1.5, 1, 0)

        # Build nail
        root_link.add_box_visual(pose=sapien.Pose([0, 0, bottom_half_size[2]]), half_size=bottom_half_size,
                                 material=viz_mat_wood)
        root_link.add_box_collision(pose=sapien.Pose([0, 0, bottom_half_size[2]]), half_size=bottom_half_size,
                                    material=physical_mat)
        root_link.set_name("nail_root")

        child_link = builder.create_link_builder(root_link)
        child_link.set_joint_name("nail_joint")
        rotation_quat = np.array([0.707, 0, 0.707, 0])
        middle_height = 2*middle_half_size[2] + bottom_half_size[2]
        top_height = middle_half_size[2] + top_half_size[2]
        child_link.set_joint_properties("prismatic", limits=np.array([[0, 0.06]]),
                                        pose_in_parent=sapien.Pose([0, 0, middle_height], rotation_quat),
                                        pose_in_child=sapien.Pose(q=rotation_quat),
                                        friction=5, damping=30)
        # child_link.add_capsule_visual(sapien.Pose([0, 0, middle_height],transforms3d.euler.euler2quat(0, np.pi/2, 0)), radius=0.0025, half_length=0.03, material=viz_mat_nail)
        # child_link.add_capsule_collision(sapien.Pose([0, 0, middle_height],transforms3d.euler.euler2quat(0, np.pi/2, 0)), radius=0.0025, half_length=0.03, material=physical_mat)        
        child_link.add_box_visual(half_size=middle_half_size, material=viz_mat_nail)
        child_link.add_box_collision(half_size=middle_half_size, material=physical_mat)
        # child_link.add_capsule_visual(sapien.Pose([0, 0, top_height],transforms3d.euler.euler2quat(0, 0, np.pi/2)), radius=0.01, half_length=0.0002, material=viz_mat_nail)
        # child_link.add_capsule_collision(sapien.Pose([0, 0, top_height],transforms3d.euler.euler2quat(0, 0, np.pi/2)), radius=0.01, half_length=0.0002, material=physical_mat)        

        child_link.add_box_visual(sapien.Pose([0, 0, top_height]), half_size=top_half_size, material=viz_mat_nail)
        child_link.add_box_collision(sapien.Pose([0, 0, top_height]), half_size=top_half_size, material=physical_mat)
        child_link.set_name("nail")
        nail = builder.build(fix_root_link=True)

        # Build hammer head
        hammer_half_size = np.array([0.02, 0.02, 0.04])
        density = 400
        viz_mat = create_visual_material(self.renderer, 0.2, 0.8, 0.4,
                                         base_color=np.array([140, 138, 132, 255]) / 255)
        builder = self.scene.create_actor_builder()
        half_width = hammer_half_size[0] + 0.003
        half_thickness = 0.003
        half_driver_height = hammer_half_size[2] - 0.005
        handle_height = half_driver_height
        extend = half_thickness + half_width
        head_half_size = np.array([hammer_half_size[0], hammer_half_size[1], half_driver_height])
        head_pos = [extend, 0, half_driver_height]
        builder.add_box_visual(sapien.Pose(np.array(head_pos)),half_size=head_half_size, material=viz_mat)
        builder.add_box_collision(sapien.Pose(np.array(head_pos)), half_size=hammer_half_size, material=physical_mat, density=density*2)

        # Add hammer handle
        handle_length = 0.2
        radius = half_thickness * 4
        builder.add_capsule_visual(sapien.Pose([half_width + radius + handle_length / 2, 0, handle_height]),
                                   radius=radius, half_length=handle_length / 2, material=viz_mat_wood)
        builder.add_capsule_collision(sapien.Pose([half_width + radius + handle_length / 2, 0, handle_height]),
                                      radius=radius, half_length=handle_length / 2, material=physical_mat,
                                      density=density)
        hammer = builder.build("hammer")

        return nail, hammer

    def reset_env(self):
        self.hammer.set_pose(self.init_pose)
        self.nail.set_qpos(np.zeros(self.nail.dof))
        self.nail.set_pose(self.target_pose)

def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = HammerEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()