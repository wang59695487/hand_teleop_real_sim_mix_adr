import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
from typing import Dict
from pathlib import Path
import json


class ArticulationScaleAnnotator:
    def __init__(self, scene: sapien.Scene, renderer: sapien.VulkanRenderer):
        self.scene = scene
        self.renderer = renderer
        self.viewer = Viewer(renderer)
        self.viewer.set_scene(scene)

    def annotate_scales(self, path_dict: Dict[str, str], result_path: str):
        """
        Annotate articulation size interactively in the viewer
        Args:
            path_dict: dict with key as articulation name (used for saving) and value as urdf path
            result_path: path to save annotation results
        """

        scale = 1
        scale_dict = {}
        for name, path in path_dict.items():
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.scale = scale
            art = loader.load(path)
            self.scene.step()
            while not self.viewer.closed:
                self.viewer.render()
                if self.viewer.window.key_down("left"):
                    scale -= 0.02
                    loader.scale = scale
                    self.scene.remove_articulation(art)
                    art = loader.load(path)
                    self.scene.step()
                elif self.viewer.window.key_down("right"):
                    scale += 0.02
                    loader.scale = scale
                    self.scene.remove_articulation(art)
                    art = loader.load(path)
                    self.scene.step()
                elif self.viewer.window.key_down("enter"):
                    scale_dict[name] = scale
                    self.scene.remove_articulation(art)
                    loader = None
                    self.scene.step()
                    break

        with open(result_path, "w") as f:
            json.dump(scale_dict, f, indent=2)


def main():
    from hand_teleop.env.sim_env.constructor import get_engine_and_renderer, add_default_scene_light
    from hand_teleop.utils.common_robot_utils import load_robot, modify_robot_visual
    engine, renderer = get_engine_and_renderer(use_gui=True, need_offscreen_render=False)
    scene = engine.create_scene()
    add_default_scene_light(scene, renderer)

    robot = load_robot(scene, "allegro_hand_xarm6_wrist_mounted_face_front")
    modify_robot_visual(robot)
    robot.set_pose(sapien.Pose([-0.55, 0, 0.14]))

    partnet_mobility_path = "/home/sim/sapien_resources/mobility_dataset/mobility_convex_alpha5"
    path_dict = {"148": f"{partnet_mobility_path}/148/mobility.urdf",
                 "149": f"{partnet_mobility_path}/149/mobility.urdf"}
    output_path = "faucet_scale.json"

    annotator = ArticulationScaleAnnotator(scene, renderer)
    annotator.annotate_scales(path_dict, output_path)


if __name__ == '__main__':
    main()
