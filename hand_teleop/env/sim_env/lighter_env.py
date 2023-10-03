from pathlib import Path

import numpy as np
import sapien.core as sapien

from env.base import BaseSimulationEnv
from utils.common_robot_utils import load_robot

VALID_PARTNET_ID = ["100348"]


class LighterEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=100348, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        if str(partnet_mobility_id) not in VALID_PARTNET_ID:
            raise ValueError(f"Partnet mobility ID {partnet_mobility_id} not valid")
        self.partnet_mobility_id = partnet_mobility_id

        # Construct scene
        scene_config = sapien.SceneConfig()
        scene_config.gravity = np.zeros(3)
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.lighter = self.load_lighter()
        self.lighter.set_pose(sapien.Pose([0, 0, 0.25], [0, 0, 0, 1]))
        self.lighter.set_qpos(np.array([0, 0]))

        # Load robot
        init_qpos = [-0.08, -0.023, 0.239, 1.57, 1.57, 0] + [0, 1.05, 1.57, 1.2] + [0, 1.23, 1.57, 1.12]
        init_qpos += [0, 0.92, 1.57, 1.27] + [0, -0.43, 0.87, 1.57, 1.38] + [-1, 0.9, 0, 0, 0]
        self.robot = load_robot(self.renderer, self.scene, "adroit_free")
        for joint in self.robot.get_active_joints():
            joint.set_drive_property(10000, 2000)
        self.robot.set_qpos(np.array(init_qpos))
        self.robot.set_drive_target(np.array(init_qpos))

    def load_lighter(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.06
        loader.fix_root_link = False
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        builder = loader.load_file_as_articulation_builder(str(vhacd_urdf), config=config)
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)
        lighter = builder.build(False)

        return lighter


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = LighterEnv()
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
