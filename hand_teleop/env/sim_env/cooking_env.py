from pathlib import Path

import sapien.core as sapien

from env.base import BaseSimulationEnv


class CookingEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, oven_id=102055, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        self.partnet_mobility_id = oven_id

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.3, 0.3, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, -0.4]))
        self.pot, self.lid, self.oven = self.load_pot_and_oven()
        self.oven.set_pose(sapien.Pose([0, 0.9, -0.6]))
        self.pot.set_pose(sapien.Pose([-0.1, 0, -0.36], [0.707, 0.707, 0, 0]))
        self.lid.set_pose(sapien.Pose([-0.1, 0, -0.34], [0.707, 0.707, 0, 0]))

    def load_pot_and_oven(self):
        mobility_root = Path("/home/sim/sapien_resources/mobility_dataset/mobility_v1_alpha5")
        urdf = str(mobility_root / str(self.partnet_mobility_id) / "mobility.urdf")

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.6
        loader.fix_root_link = True
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        oven = loader.load(urdf, config=config)

        # Pot bottom
        mobility_convex_root = Path("/home/sim/sapien_resources/mobility_dataset/mobility_convex_alpha5")
        urdf = str(mobility_convex_root / "100693" / "mobility.urdf")
        loader.fix_root_link = False
        loader.scale = 0.2
        pot_builder = loader.load_file_as_articulation_builder(urdf)
        pot_link = pot_builder.get_link_builders()[1]
        pot = pot_link.build("pot")

        # Pot lid
        lid_builder = loader.load_file_as_articulation_builder(urdf)
        lid_link = lid_builder.get_link_builders()[-1]
        lid = lid_link.build("lid")

        return pot, lid, oven


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    from utils.common_robot_utils import load_robot
    env = CookingEnv(oven_id=101921)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    robot = load_robot(env.scene, "allegro_free")

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
