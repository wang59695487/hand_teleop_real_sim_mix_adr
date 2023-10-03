from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d

from env.base import BaseSimulationEnv
from utils.common_robot_utils import load_robot
from utils.model_utils import create_visual_material

VALID_PARTNET_ID = ["100182"]


class PlierEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=100182, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        if str(partnet_mobility_id) not in VALID_PARTNET_ID:
            raise ValueError(f"Partnet mobility ID {partnet_mobility_id} not valid")
        self.partnet_mobility_id = partnet_mobility_id

        # Construct scene
        scene_config = sapien.SceneConfig()
        # scene_config.gravity = np.zeros(3)
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.25, 0.25, 0.025])
        self.plier = self.load_plier()
        self.plier.set_pose(sapien.Pose([-0.1, 0, 0.05], transforms3d.euler.euler2quat(0, np.pi, 0)))
        self.plier.set_qpos(np.zeros(self.plier.dof))
        for joint in self.plier.get_active_joints():
            joint.set_friction(10)

        # Load robot
        # self.robot = load_robot(self.renderer, self.scene, "adroit_free")
        # for joint in self.robot.get_active_joints():
        #     joint.set_drive_property(10000, 2000)
        # qpos = np.zeros(self.robot.dof)
        # qpos[:6] = [-0.2, 0, 0.08, 0, np.pi / 2, 0]
        # self.robot.set_qpos(qpos)
        # self.robot.set_drive_target(qpos)

        # Load tower and block
        tower, block, support = self.load_tower_and_block()
        block.set_pose(sapien.Pose([0, -0.1, 0.03]))
        support.set_pose(sapien.Pose([0, -0.1, 0]))
        tower.set_pose(sapien.Pose([0, 0.1, 0]))

    def load_plier(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.1
        loader.fix_root_link = False
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        builder = loader.load_file_as_articulation_builder(str(vhacd_urdf), config=config)
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)
        plier = builder.build(False)

        return plier

    def load_tower_and_block(self):
        tower_inner_half_size = 0.01
        block_outer_half_size = 0.025
        block_thickness = 0.03
        shell_size, shell_thickness, shell_offset = 0.04, 0.005, 0.03
        base_thickness, pin_height = 0.01, 0.15

        tower = create_tower(self.scene, self.renderer, inner_size=tower_inner_half_size - 0.002, shell_size=shell_size,
                             shell_thickness=shell_thickness / 2, shell_offset=shell_offset,
                             base_thickness=base_thickness / 2, pin_height=pin_height)

        block, support = create_hole_block_and_support(self.scene, self.renderer, block_outer_half_size,
                                                       tower_inner_half_size, block_thickness / 2)
        return tower, block, support


def create_tower(scene: sapien.Scene, renderer, inner_size, shell_size, shell_thickness, shell_offset,
                 base_thickness, pin_height):
    builder = scene.create_actor_builder()
    viz_mat = create_visual_material(renderer, 0.1, 0.5, 0.4,
                                     base_color=np.array([80, 72, 74, 255]) / 255)
    pin_viz_mat = create_visual_material(renderer, 0.1, 0.1, 0.4,
                                         base_color=np.array([180, 172, 174, 255]) / 255)
    # Build base
    base_size = np.array([shell_size, shell_size, base_thickness])
    builder.add_box_visual(sapien.Pose([0, 0, base_thickness]), base_size, material=viz_mat)
    builder.add_box_collision(sapien.Pose([0, 0, base_thickness]), base_size)

    # Build pin
    pin_size = np.array([inner_size, inner_size, pin_height / 2])
    builder.add_box_visual(sapien.Pose([0, 0, pin_height / 2]), pin_size, material=pin_viz_mat)

    # Build shell
    shell_half_height = (pin_height + shell_offset) / 2
    shell_sizes = [
        (shell_size + shell_thickness, shell_thickness, shell_half_height),
        (shell_thickness, shell_size, shell_half_height),
        (shell_size + shell_thickness, shell_thickness, shell_half_height),
        (shell_size + shell_thickness, shell_size + shell_thickness * 2, shell_thickness)
    ]
    shell_pos = [
        (shell_thickness, shell_size + shell_thickness, shell_half_height),
        (shell_thickness + shell_size, 0, shell_half_height),
        (shell_thickness, -shell_size - shell_thickness, shell_half_height),
        (shell_thickness, 0, shell_half_height * 2 + shell_thickness)
    ]

    for i in range(4):
        builder.add_box_visual(sapien.Pose(np.array(shell_pos[i])), np.array(shell_sizes[i]), material=viz_mat)
        builder.add_box_collision(sapien.Pose(np.array(shell_pos[i])), np.array(shell_sizes[i]))

    return builder.build("tower")


def create_hole_block_and_support(scene: sapien.Scene, renderer, outer_size, inner_size, thickness):
    # Build block
    builder = scene.create_actor_builder()
    R = outer_size + inner_size
    r = outer_size - inner_size
    block_sizes = [(R / 2, r / 2), (r / 2, R / 2)] * 2
    block_xy = [(-r / 2, R / 2), (-R / 2, -r / 2), (r / 2, -R / 2), (R / 2, r / 2)]
    viz_mat = create_visual_material(renderer, 0.1, 0.5, 0.4,
                                     base_color=np.array([40, 36, 37, 255]) / 255)
    for i in range(4):
        builder.add_box_visual(sapien.Pose(np.array([block_xy[i][0], block_xy[i][1], thickness / 2])),
                               half_size=np.array([block_sizes[i][0], block_sizes[i][1], thickness / 2]),
                               material=viz_mat)
        builder.add_box_collision(sapien.Pose(np.array([block_xy[i][0], block_xy[i][1], thickness / 2])),
                                  half_size=np.array([block_sizes[i][0], block_sizes[i][1], thickness / 2]))
    block = builder.build("block")

    # Build support
    builder = scene.create_actor_builder()
    builder.add_box_visual(sapien.Pose(np.array([0, 0, thickness])), np.array([R / 2, R / 2, thickness]),
                           color=np.array([0, 0, 0.5, 1]))
    builder.add_box_collision(sapien.Pose(np.array([0, 0, thickness])), np.array([R / 2, R / 2, thickness]))
    support = builder.build("support")

    return block, support


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = PlierEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()
        # robot_qpos = env.robot.get_drive_target()
        # robot_qpos[0] -=0.001
        # env.robot.set_drive_target(robot_qpos)


if __name__ == '__main__':
    env_test()
