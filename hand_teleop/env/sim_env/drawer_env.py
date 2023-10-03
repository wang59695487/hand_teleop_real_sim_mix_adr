from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.ycb_object_utils import load_ycb_object

from icecream import ic


class DrawerEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, friction=1, object_name="drawer_1", use_visual_obs=False, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        # Construct scene
        # scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)
        self.friction = friction

        # Dummy camera creation to initial geometry object
        ic(use_visual_obs)
        if use_visual_obs: # TODO: [2022-05-05 12:38:36.273] [SAPIEN] [error] Failed to add camera: renderer is not added to simulation.
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)
        ic('loading table')
        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])
        self.drawer = self.load_drawer(object_name.split('_')[-1]) # index = object_name.split('_')[-1]
        self.drawer.set_qpos([0.15])
        ic('loaded table')

    def load_drawer(self, index):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()

        loader.fix_root_link = 0

        current_dir = Path(__file__).absolute().parent
        dir = current_dir.parent.parent.parent / "assets" / "akb48" / "drawer" / ("drawer" + str(index)) / ("drawer" + str(index) + "_vhacd.urdf")
        # ic(dir)
        loader.load_multiple_collisions_from_file = True
        drawer: sapien.Articulation = loader.load(dir.__str__())
        drawer.set_root_pose(sapien.Pose([-0.05, -0.05, 0.1], [1, 0, 0, -np.pi / 2]))
        joints = drawer.get_joints()
        joints[1].set_friction(100)
        joints[1].set_drive_property(0, 5)
        return drawer

    def reset_env(self):
        self.drawer.set_qpos([0.15])
        random_xy = (np.random.rand(2) * 2 - 1) * 0.05
        random_pos = np.concatenate([random_xy, [0.1]])
        self.drawer.set_pose(sapien.Pose(random_pos, [1, 0, 0, -np.pi / 2]))


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = DrawerEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        print(env.drawer.get_qpos())
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()


