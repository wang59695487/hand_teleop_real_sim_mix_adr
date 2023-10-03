from pathlib import Path

import numpy as np
import sapien.core as sapien

from env.base import BaseSimulationEnv

VALID_PARTNET_ID = ["103299", "103111", "103271"]


class StaplerEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=103299, **renderer_kwargs):
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
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.stapler = self.load_stapler()
        self.stapler.set_pose(sapien.Pose([0, 0, 0.05], [1, 0, 0, 0]))
        self.stapler.set_qpos(np.array([1, 0]))

        # Load paper
        self.paper = self.load_paper()
        self.paper.set_pose(sapien.Pose([0, 0.2, 0.1]))

    def load_stapler(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.15
        loader.fix_root_link = True
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        builder = loader.load_file_as_articulation_builder(str(vhacd_urdf), config=config)
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)
        stapler = builder.build(True)

        stapler.get_active_joints()[-1].set_limits(np.array([[0, 0.1]]))
        return stapler

    def load_paper(self):
        builder = self.scene.create_actor_builder()
        half_size = np.array([0.06, 0.003, 0.1])
        builder.add_box_visual(half_size=half_size, color=np.array([0.8, 0.8, 0.8, 1]))
        builder.add_box_collision(half_size=half_size)

        paper = builder.build("paper")
        return paper


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = StaplerEnv(partnet_mobility_id=103299)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
