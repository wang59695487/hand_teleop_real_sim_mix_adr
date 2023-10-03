import numpy as np
import sapien.core as sapien

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.model_utils import create_visual_material


class WrenchUsingEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, 0]))

        self.bolt, self.driver = self.load_hex_bolt_and_driver()
        self.bolt.set_pose(sapien.Pose([0, 0.2, 0.0]))
        self.driver.set_pose(sapien.Pose([0, -0.1, 0.0], [0.707, 0, 0, -0.707]))

    def reset_env(self):
        pass

    def load_hex_bolt_and_driver(self):
        builder = self.scene.create_articulation_builder()
        root_link = builder.create_link_builder()
        bottom_half_size = np.array([0.15, 0.15, 0.015])
        middle_half_size = np.array([0.03, 0.03, 0.1])
        top_half_size = np.array([0.015, 0.015, 0.03])
        viz_mat_bolt = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                              base_color=np.array([80, 72, 74, 255]) / 255)
        physical_mat = self.scene.create_physical_material(1.5, 1, 0)

        # Build hex bolt
        root_link.add_box_visual(pose=sapien.Pose([0, 0, bottom_half_size[2]]), half_size=bottom_half_size,
                                 material=viz_mat_bolt)
        root_link.add_box_collision(pose=sapien.Pose([0, 0, bottom_half_size[2]]), half_size=bottom_half_size,
                                    material=physical_mat)
        root_link.set_name("hex_root")

        child_link = builder.create_link_builder(root_link)
        child_link.set_joint_name("hinge_joint")
        rotation_quat = np.array([0.707, 0, 0.707, 0])
        middle_height = middle_half_size[2] + bottom_half_size[2]
        top_height = middle_half_size[2] + top_half_size[2]
        child_link.set_joint_properties("revolute", limits=np.array([[0, np.pi / 2]]),
                                        pose_in_parent=sapien.Pose([0, 0, middle_height], rotation_quat),
                                        pose_in_child=sapien.Pose(q=rotation_quat),
                                        friction=5, damping=20)
        child_link.add_box_visual(half_size=middle_half_size, material=viz_mat_bolt)
        child_link.add_box_collision(half_size=middle_half_size, material=physical_mat)
        child_link.add_box_visual(sapien.Pose([0, 0, top_height]), half_size=top_half_size, material=viz_mat_bolt)
        child_link.add_box_collision(sapien.Pose([0, 0, top_height]), half_size=top_half_size, material=physical_mat)
        child_link.set_name("hex_head")
        bolt = builder.build(fix_root_link=True)

        # Build hex driver
        density = 400
        viz_mat = create_visual_material(self.renderer, 0.2, 0.8, 0.4,
                                         base_color=np.array([140, 138, 132, 255]) / 255)
        builder = self.scene.create_actor_builder()
        half_width = top_half_size[0] + 0.003
        half_thickness = 0.003
        half_driver_height = top_half_size[2] - 0.005
        handle_height = half_driver_height
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
                                      density=density * 2)

        # Add hex handle
        handle_length = 0.2
        radius = half_thickness * 4
        builder.add_capsule_visual(sapien.Pose([half_width + radius + handle_length / 2, 0, handle_height]),
                                   radius=radius, half_length=handle_length / 2, material=viz_mat)
        builder.add_capsule_collision(sapien.Pose([half_width + radius + handle_length / 2, 0, handle_height]),
                                      radius=radius, half_length=handle_length / 2, material=physical_mat,
                                      density=density)
        driver = builder.build("hex_driver")

        return bolt, driver


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = WrenchUsingEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
