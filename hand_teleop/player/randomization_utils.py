from pathlib import Path
from typing import NamedTuple, List, Dict

import numpy as np
import sapien.core as sapien

from hand_teleop.env.rl_env.laptop_env import LaptopRLEnv
from hand_teleop.env.rl_env.pick_place_env import PickPlaceRLEnv
from hand_teleop.env.rl_env.table_door_env import TableDoorRLEnv
from hand_teleop.env.rl_env.insert_object_env import InsertObjectRLEnv
from hand_teleop.env.rl_env.hammer_env import HammerRLEnv
from hand_teleop.env.rl_env.mug_flip_env import MugFlipRLEnv

## THIS SCRIPT WILL INCLUDE DOMAIN RANDOMIZATION FUNCTIONS

def randomize_robot_colors(robot: sapien.Articulation):
    robot_name = robot.get_name()
    if "mano" in robot_name:
        return robot
    arm_link_names = [f"link{i}" for i in range(1, 8)] + ["link_base"]
    rand_color_hand = np.random.rand(3)
    rand_color_arm = np.random.rand(3)
    for link in robot.get_links():
        if link.get_name() in arm_link_names:
            rand_color = rand_color_arm
            for geom in link.get_visual_bodies():
                for shape in geom.get_render_shapes():
                    mat_viz = shape.material
                    mat_viz.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
        else:
            rand_color = rand_color_hand
            for geom in link.get_visual_bodies():
                for shape in geom.get_render_shapes():
                    mat_viz = shape.material
                    mat_viz.set_specular(0.07)
                    mat_viz.set_metallic(0.3)
                    mat_viz.set_roughness(0.2)
                    if 'adroit' in robot_name:
                        mat_viz.set_specular(0.02)
                        mat_viz.set_metallic(0.1)
                        mat_viz.set_base_color(np.power(np.array([rand_color[0], rand_color[1], rand_color[2], 1]), 1.5))
                    elif 'allegro' in robot_name:
                        if "tip" not in link.get_name():
                            mat_viz.set_specular(0.5)
                            mat_viz.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                        else:
                            mat_viz.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    elif 'svh' in robot_name:
                        link_names = ["right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q",
                                      "right_hand_e1"]
                        if link.get_name() not in link_names:
                            mat_viz.set_specular(0.02)
                            mat_viz.set_metallic(0.1)
                    elif 'ar10' in robot_name:
                        rand_color = rand_color*255
                        if "tip" in link.get_name():
                            mat_viz.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 255]) / 255)
                            mat_viz.set_metallic(0)
                            mat_viz.set_specular(0)
                            mat_viz.set_roughness(1)
                        else:
                            color = np.array([rand_color[0], rand_color[1], rand_color[2], 255]) / 255
                            mat_viz.set_base_color(np.power(color, 2.2))
                    else:
                        pass
    return robot

def randomize_env_colors(task_name, env):
    # Randomize table colors in all environments
    tables = []
    for table in env.tables:
        rand_color = np.random.rand(3)
        for geom in table.get_visual_bodies():
            for shape in geom.get_render_shapes():
                    table_visual_material = shape.material
                    table_visual_material.set_metallic(0.0)
                    table_visual_material.set_specular(0.3)
                    table_visual_material.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    table_visual_material.set_roughness(0.3)
        tables.append(table)
    env.tables = tables

    if task_name == 'pick_place':
        mug = env.manipulated_object
        for geom in mug.get_visual_bodies():
            for shape in geom.get_render_shapes():
                    rand_color = np.random.rand(3)
                    mug_visual_material = shape.material
                    mug_visual_material.set_metallic(0.0)
                    mug_visual_material.set_specular(0.3)
                    mug_visual_material.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    mug_visual_material.set_roughness(0.3)
        env.manipulated_object = mug
        plate = env.plate
        for geom in plate.get_visual_bodies():
            for shape in geom.get_render_shapes():
                    rand_color = np.random.rand(3)
                    plate_visual_material = shape.material
                    plate_visual_material.set_metallic(0.0)
                    plate_visual_material.set_specular(0.3)
                    plate_visual_material.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    plate_visual_material.set_roughness(0.3)
        env.plate = plate    
    elif task_name == 'hammer':
        raise NotImplementedError
    elif task_name == 'table_door':
        door = env.table_door
        for geom in door.get_visual_bodies():
            for shape in geom.get_render_shapes():
                    rand_color = np.random.rand(3)
                    door_visual_material = shape.material
                    door_visual_material.set_metallic(0.0)
                    door_visual_material.set_specular(0.3)
                    door_visual_material.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    door_visual_material.set_roughness(0.3)
        env.table_door = door
    elif task_name == 'insert_object':
        raise NotImplementedError
    elif task_name == 'mug_flip':
        mug = env.manipulated_object
        for geom in mug.get_visual_bodies():
            for shape in geom.get_render_shapes():
                    rand_color = np.random.rand(3)
                    mug_visual_material = shape.material
                    mug_visual_material.set_metallic(0.0)
                    mug_visual_material.set_specular(0.3)
                    mug_visual_material.set_base_color(np.array([rand_color[0], rand_color[1], rand_color[2], 1]))
                    mug_visual_material.set_roughness(0.3)
        env.manipulated_object = mug
    else:
        raise NotImplementedError

    env.robot = randomize_robot_colors(env.robot)

    return env
