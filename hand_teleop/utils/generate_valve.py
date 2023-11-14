import shutil
from pathlib import Path
from typing import Optional, List

import coacd
import numpy as np
import transforms3d
import trimesh
from trimesh import creation


def generate_valve(valve_angles: List[float], dir_name: str, radius_scale: float = 1.0,
                   capsule_radius_scale: float = 1.0, valve_colors: Optional[List[List[float]]] = None):
    if valve_colors is None:
        valve_colors = [np.array([255, 255, 255])] * len(valve_angles)
    assert len(valve_angles) == len(valve_colors)
    num_finger = len(valve_angles)

    robel_auto_dir = Path(__file__).parent.parent.parent / "assets/robel_auto"
    output_folder = robel_auto_dir / dir_name
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    base_mount_path = robel_auto_dir / "mount_base.obj"
    base_mount_convex_path = robel_auto_dir / "mount_base_convex_hull.obj"
    scene = trimesh.Scene()

    # Constant
    capsule_height = 39.854
    capsule_length = 61.706 * radius_scale  # Distance between sphere center
    capsule_radius = 19.5 * capsule_radius_scale

    # Capsule
    merged = None
    visual_scene = trimesh.Scene()
    for angle, color in zip(valve_angles, valve_colors):
        capsule = creation.capsule(
            height=capsule_length, radius=capsule_radius)
        rotation = transforms3d.euler.euler2mat(
            0, np.pi / 2, np.deg2rad(angle), "sxyz")
        mat44 = np.eye(4)
        mat44[:3, :3] = rotation
        mat44[:3, 3] = [0, 0, capsule_height]
        trans = np.eye(4)
        trans[2, 3] = capsule_length / 2
        capsule.apply_transform(mat44 @ trans)

        if merged is None:
            merged = capsule
        else:
            merged = merged.union(capsule)

        visual_sphere = creation.uv_sphere(radius=capsule_radius + 1e-3)
        visual_cylinder = creation.cylinder(
            radius=capsule_radius + 1e-3, height=capsule_length)
        rotation = transforms3d.euler.euler2mat(
            0, np.pi / 2, np.deg2rad(angle), "sxyz")
        mat44 = np.eye(4)
        mat44[:3, :3] = rotation
        mat44[:3, 3] = [0, 0, capsule_height]
        trans = np.eye(4)
        trans[2, 3] = capsule_length / 2
        visual_cylinder.apply_transform(mat44 @ trans)
        trans[2, 3] = capsule_length
        visual_sphere.apply_transform(mat44 @ trans)
        visual = visual_sphere.union(visual_cylinder)
        visual.visual.vertex_colors[:, :3] = np.array(
            color).astype(np.uint8)[None, :]
        visual.visual = visual.visual.to_texture()
        visual_scene.add_geometry(visual)

    coordinate = creation.axis(
        origin_color=[1, 0, 0], origin_size=1, transform=np.eye(4))
    scene.add_geometry(coordinate)
    visual_scene.add_geometry(coordinate)
    # visual_scene.show()

    # Mount
    base = trimesh.load(str(base_mount_path), force="mesh")
    convex = trimesh.load(str(base_mount_convex_path), force="mesh")
    mat44 = np.eye(4)
    mat44[2, 3] = -0.1
    convex.apply_transform(mat44)
    drilled = merged.difference(convex, engine="blender")
    # final = drilled.union(base, engine="blender")
    final = drilled

    # Color
    # vertices = final.vertices[:, :2]
    # dis = np.linalg.norm(vertices, axis=1)
    # dis_bool = dis > (capsule_radius + 1)
    #
    # x, y = vertices[:, 0], vertices[:, 1]
    # t_angle = np.arctan2(y, x)
    # t_angle[t_angle < 0] += np.pi * 2
    # more_angles = valve_angles * 3
    # final.visual.vertex_colors = np.array([255, 255, 255, 255]).astype(np.uint8)
    # for i, (angle, color) in enumerate(zip(valve_angles, valve_colors)):
    #     # TODO: unwrap
    #     next_angle = more_angles[i + num_finger + 1]
    #     if next_angle < angle:
    #         next_angle += 360
    #     angle_upper = (next_angle + angle) / 2
    #     previous_angle = more_angles[i + num_finger - 1]
    #     if previous_angle > angle:
    #         previous_angle -= 360
    #     angle_lower = (previous_angle + angle) / 2
    #
    #     angle_bool1 = np.logical_and(t_angle > np.deg2rad(angle_lower + 1), t_angle < np.deg2rad(angle_upper - 1))
    #     angle_bool2 = np.logical_and(t_angle + np.pi * 2 > np.deg2rad(angle_lower + 1),
    #                                  t_angle + np.pi * 2 < np.deg2rad(angle_upper - 1))
    #     angle_bool3 = np.logical_and(t_angle - np.pi * 2 > np.deg2rad(angle_lower + 1),
    #                                  t_angle - np.pi * 2 < np.deg2rad(angle_upper - 1))
    #     vertex_bool = np.logical_or.reduce([angle_bool1, angle_bool2, angle_bool3])
    #     vertex_bool = np.logical_and(vertex_bool, dis_bool)
    #     final.visual.vertex_colors[vertex_bool, :3] = np.array(color).astype(np.uint8)[None, :]
    #     face_bool = np.sum(vertex_bool[final.faces], axis=1) > 1
    #     final.visual.face_colors[face_bool, :3] = np.array(color).astype(np.uint8)[None, :]

    scene.add_geometry(final)
    visual_scene.export(
        str(output_folder / "up_visual.obj"), include_color=True)
    drilled.export(str(output_folder / "3d_print.obj"))

    # COACD
    mesh = coacd.Mesh(final.vertices, final.faces)
    result = coacd.run_coacd(mesh, preprocess_mode="off")
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    coacd_scene = trimesh.Scene()
    for p in mesh_parts:
        coacd_scene.add_geometry(p)
    coacd_scene.export(str(output_folder / "up_collision.obj"))

    # Export URDF
    shutil.copy(str(robel_auto_dir / "urdf_template.urdf"),
                str(output_folder / "dclaw_valve.urdf"))
    print(f"Generate {robel_auto_dir}")


if __name__ == '__main__':
    name = f"valve_5-cross_72_1.2"
    generate_valve([0, 72, 144, 216, 288], name, valve_colors=[[255, 0, 0], [255, 255, 255], [
        0, 0, 0], [255, 255, 255], [255, 255, 255]], radius_scale=1.2)

    # sets = [
    #     [0, 135, 270],
    #     [0, 90, 180, 270],
    #     [0, 60, 180, 240],
    #     [0, 75, 180, 255],
    #     [0, 75, 150, 225, 300],
    #     [0, 72, 144, 216, 288],
    #     [0, 60, 120, 180, 240],
    # ]

    # for angles in sets:
    #     if len(angles) == 3:
    #         color = [[255, 0, 0], [0, 0, 0], [255, 255, 255]]
    #     elif len(angles) == 2:
    #         color = [[255, 0, 0], [0, 0, 0]]
    #     elif len(angles) == 4:
    #         color = [[255, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]]
    #     elif len(angles) == 5:
    #         color = [[255, 0, 0], [255, 255, 255], [
    #             0, 0, 0], [255, 255, 255], [255, 255, 255]]
    #     else:
    #         raise NotImplementedError

    #     name = f"valve_{len(angles)}-cross_{angles[1]}"
    #     generate_valve(angles, name, valve_colors=color)
