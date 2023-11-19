import io
import os
import zipfile
from pathlib import Path

import numpy as np
import numpy.random as random
import time
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

import requests
import sapien.core as sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

_engine = None
_renderer = None
_init = False
_use_gui = None
_use_ray_tracing = None


def get_engine_and_renderer(use_gui=True, use_ray_tracing=False, device="", mipmap_levels=1,
                            need_offscreen_render=False, no_rgb=False):
    global _engine, _renderer
    no_rgb = no_rgb and (not use_gui)
    if _init:
        if use_gui is not _use_gui:
            raise RuntimeError(
                f"Use GUI setting has already been initialized.\n"
                f"Conflict: current renderer:{_use_gui}, but required: {use_gui}")
        if _use_ray_tracing is not use_ray_tracing:
            raise RuntimeError(
                f"Use GUI setting has already been initialized.\n"
                f"Conflict: current renderer:{_use_gui}, but required: {use_gui}")
        return _engine, _renderer

    _engine = sapien.Engine()
    need_renderer = need_offscreen_render or use_gui
    if use_ray_tracing:
        raise NotImplementedError
    else:
        if need_renderer:
            # _renderer = sapien.VulkanRenderer(default_mipmap_levels=mipmap_levels, offscreen_only=not use_gui,
            #                                   device=device, do_not_load_texture=no_rgb)
            _renderer = sapien.VulkanRenderer(default_mipmap_levels=mipmap_levels, offscreen_only=not use_gui,
                                              device=device)
            _engine.set_renderer(_renderer)
            if no_rgb:
                print(f"Use trivial renderer without color.")
                sapien.VulkanRenderer.set_camera_shader_dir("trivial")
            else:
                sapien.VulkanRenderer.set_camera_shader_dir("ibl")
        if use_gui:
            sapien.VulkanRenderer.set_viewer_shader_dir("ibl")
            viewer = Viewer(_renderer)
            viewer.close()
    _engine.set_log_level("error")
    return _engine, _renderer


def download_maniskill(model_id, directory=None):
    url = 'https://storage1.ucsd.edu/datasets/PartNetMobilityScrambled/{}.zip'.format(
        model_id
    )
    if not directory:
        directory = os.environ.get('PARTNET_MOBILITY_DATASET')
        if not directory:
            directory = 'partnet-mobility-dataset'
    urdf_file = os.path.join(directory, str(model_id), 'mobility.urdf')

    # return if file exists
    if os.path.exists(urdf_file):
        return urdf_file

    # download file
    r = requests.get(url, stream=True)
    if not r.ok:
        raise Exception(
            'Download PartNet-Mobility failed. '
            'Please check your token and IP address.'
            'Also make sure sure the model id is valid'
        )

    z = zipfile.ZipFile(io.BytesIO(r.content))

    os.makedirs(directory, exist_ok=True)
    z.extractall(directory)
    return urdf_file


def add_default_scene_light(scene: sapien.Scene, renderer: sapien.VulkanRenderer, add_ground=True, cast_shadow=True):
    # If the light is already set, then we just skip the function.
    # if len(scene.get_all_lights()) >= 3:
    #     return
    # asset_dir = Path(__file__).parent.parent.parent.parent / "assets"
    # ktx_path = asset_dir / "misc" / "ktx" / "day.ktx"
    # scene.set_environment_map(str(ktx_path))
    scene.add_directional_light(
        np.array([-1, -1, -1]), np.array([0.5, 0.5, 0.5]), shadow=cast_shadow)
    scene.add_directional_light([0, 0, -1], [0.9, 0.8, 0.8], shadow=False)
    scene.add_spot_light(np.array([0, 0, 1.5]), direction=np.array([0, 0, -1]), inner_fov=0.3, outer_fov=1.0,
                         color=np.array([0.5, 0.5, 0.5]), shadow=False)

    visual_material = renderer.create_material()
    visual_material.set_base_color(np.array([0, 0, 0, 0.5]))
    visual_material.set_roughness(0.7)
    visual_material.set_metallic(1)
    visual_material.set_specular(0.04)
    if add_ground:
        scene.add_ground(-1, render_material=visual_material,
                         render_half_size=np.array([50, 50]))


def generate_direction_in_scale(randomness_scale=1, num=1):
    r_limited = 0.5 + randomness_scale * 0.1
    directions = []
    # Limited the direction in a certain scale
    for i in range(num):
        while True:
            direction = R.random().as_euler('xyz', degrees=False)
            # normalize direction
            direction = direction/norm(direction, 2)
            r = np.sqrt(1-direction[2]**2)
            if r < r_limited:
                if direction[2] > 0:
                    direction[2] = -direction[2]
                directions.append(direction)
                break
    return directions


def random_environment_map(scene: sapien.Scene, randomness_scale=1):
    var = randomness_scale * 0.1
    scene.set_environment_map(create_dome_envmap(filename='/tmp/sapien_dome.ktx', sky_color=random.uniform(
        0.5-var, 0.5+var, size=3), ground_color=random.uniform(0.2-var, 0.2+var, size=3), blend=0.3, resolution=256))


def random_scene_light(scene: sapien.Scene, renderer: sapien.VulkanRenderer, randomness_scale=1, add_ground=True, cast_shadow=True):
    # If the light is already set, then we just skip the function.
    var = randomness_scale * 0.1

    for light in scene.get_all_lights():
        light.set_direction(generate_direction_in_scale(
            randomness_scale=randomness_scale)[0])
        # print(light.color)
        colors = []
        for color in light.color:
            color = random.uniform(color-var, color+var)
            colors.append(color)
        light.set_color(colors)
