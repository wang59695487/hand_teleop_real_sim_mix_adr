from pathlib import Path

import numpy as np
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
        self.mug, self.bottle = self.load_pot_and_oven()
        self.mug.set_pose(sapien.Pose([0, -0.1, -0.362]))
        self.bottle.set_pose(sapien.Pose([0, 0.1, -0.1]))

    def load_pot_and_oven(self):
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file("/home/sim/project/xiaolong_hand/hand_teleop/assets/misc/objects/mug_visual.obj")
        builder.add_multiple_collisions_from_file(
            "/home/sim/project/xiaolong_hand/hand_teleop/assets/misc/objects/mug_collision.obj")
        mug = builder.build(name="mug")

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            "/home/sim/project/xiaolong_hand/hand_teleop/assets/misc/objects/bottle_blue_google_norm.obj")
        builder.add_multiple_collisions_from_file(
            "/home/sim/project/xiaolong_hand/hand_teleop/assets/misc/objects/bottle_blue_google_norm.obj")
        bottle = builder.build(name="mug")

        return mug, bottle


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    from utils.common_robot_utils import load_robot
    env = CookingEnv(oven_id=101921)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    robot = load_robot(env.renderer, env.scene, "allegro_free")
    robot.set_pose(sapien.Pose([0, 0, 0], [0, 0, 0, -1]))


    i = 0
    while not viewer.closed:
        env.render()
        object_pose = env.bottle.get_pose()
        robot_pose = robot.get_pose()
        qpos = robot.get_qpos()
        data = [object_pose.p, object_pose.q, robot_pose.p, robot_pose.q, qpos]
        data = np.concatenate(data)
        np.save(f"{i}-pr.npy", data)
        print(f"Save {i}")
        i += 1
        viewer.toggle_pause(True)


def env_evaluate():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    from utils.common_robot_utils import load_robot
    env = CookingEnv(oven_id=101921)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    robot = load_robot(env.renderer, env.scene, "allegro_free")
    robot.set_pose(sapien.Pose([0, 0, 0], [0, 0, 0, -1]))
    viewer.toggle_pause(True)

    from utils.common_robot_utils import modify_robot_visual

    modify_robot_visual(robot)

    # for link in robot.get_links():
    #     for geom in link.get_visual_bodies():
    #         for mesh in geom.get_render_shapes():
    #             material = mesh.material
    #             material.set_base_color([0.4, 0.4, 0.4, 1])
    #             material.set_metallic(0.2)
    #             material.set_specular(0.5)
    viewer.toggle_pause(True)

    i = 0
    while not viewer.closed:
        data = np.load(f"{i}-pr.npy")
        print(f"Load {i}")
        env.render()
        object_pose = data[:7]
        robot_pose = data[7:14]
        qpos = data[14:]
        robot.set_qpos(qpos)
        robot.set_pose(sapien.Pose(robot_pose[:3], robot_pose[3:7]))
        env.bottle.set_pose(sapien.Pose(object_pose[:3], object_pose[3:7]))
        i += 1
        viewer.toggle_pause(True)


if __name__ == '__main__':
    env_evaluate()
