from typing import Optional

import numpy as np
import sapien.core as sapien
from sapien.core.pysapien import renderer as R
from transforms3d.euler import quat2euler
from transforms3d.quaternions import axangle2quat as aa, qmult, mat2quat, rotate_vector


class FPSCameraController:
    def __init__(self, window: sapien.VulkanWindow):
        self.window = window
        self.forward = np.array([1, 0, 0])
        self.up = np.array([0, 0, 1])
        self.left = np.cross(self.up, self.forward)
        self.initial_rotation = mat2quat(
            np.array([-self.left, self.up, -self.forward]).T
        )
        self.xyz = np.zeros(3)
        self.rpy = np.zeros(3)

    def setRPY(self, roll, pitch, yaw):
        self.rpy = np.array([roll, pitch, yaw])
        self.update()

    def setXYZ(self, x, y, z):
        self.xyz = np.array([x, y, z])
        self.update()

    def move(self, forward, left, up):
        q = qmult(
            qmult(aa(self.up, -self.rpy[2]), aa(self.left, -self.rpy[1])),
            aa(self.forward, self.rpy[0]),
        )
        self.xyz = self.xyz + (
                rotate_vector(self.forward, q) * forward
                + rotate_vector(self.left, q) * left
                + rotate_vector(self.up, q) * up
        )
        self.update()

    def rotate(self, roll, pitch, yaw):
        self.rpy = self.rpy + np.array([roll, pitch, yaw])
        self.update()

    def update(self):
        self.rpy[1] = np.clip(self.rpy[1], -1.57, 1.57)
        if self.rpy[2] >= 3.15:
            self.rpy[2] = self.rpy[2] - 2 * np.pi
        elif self.rpy[2] <= -3.15:
            self.rpy[2] = self.rpy[2] + 2 * np.pi

        rot = qmult(
            qmult(
                qmult(aa(self.up, -self.rpy[2]), aa(self.left, -self.rpy[1])),
                aa(self.forward, self.rpy[0]),
            ),
            self.initial_rotation,
        )
        self.window.set_camera_rotation(rot)
        self.window.set_camera_position(self.xyz)


class MinimalCameraViewer:
    def __init__(
            self,
            renderer: sapien.VulkanRenderer,
            shader_dir="",
            resolution=(960, 1080),
            camera_name="undefined",
    ):
        self.shader_dir = shader_dir
        self.renderer = renderer
        self.renderer_context: R.Context = renderer._internal_context
        self.scene = Optional[sapien.Scene]

        self.window = None
        self.set_window_resolution(resolution)
        self.fps_camera_controller = Optional[FPSCameraController]
        self.resolution = None

        self.fovy = np.pi / 2
        self.camera_ui = None

        self.basic_info_window = None
        self.camera_name = camera_name

    def set_window_resolution(self, resolution):
        assert len(resolution)
        assert len(resolution) == 2

        self.window = self.renderer.create_window(
            resolution[0], resolution[1], self.shader_dir
        )
        self.resolution = resolution

    @property
    def closed(self):
        return self.window is None

    def close(self):
        self.scene = None
        self.fps_camera_controller = None
        self.window = None
        self.camera_ui = None
        self.basic_info_window = None

    @staticmethod
    def get_camera_pose(camera: sapien.VulkanCamera):
        """Get the camera pose in the Sapien world."""
        opengl_pose = camera.get_model_matrix()  # opengl camera-> sapien world
        # sapien camera -> opengl camera
        sapien2opengl = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cam_pose = sapien.Pose.from_transformation_matrix(opengl_pose @ sapien2opengl)
        return cam_pose

    def set_scene(self, scene: sapien.Scene):
        self.scene = scene
        self.window.set_scene(scene)
        self.fps_camera_controller = FPSCameraController(self.window)
        self.build_basic_info_window()
        self.set_fovy(np.pi / 2)

    def set_camera_xyz(self, x, y, z):
        self.fps_camera_controller.setXYZ(x, y, z)
        self.build_basic_info_window()

    def set_camera_rpy(self, r, p, y):
        self.fps_camera_controller.setRPY(r, p, y)
        self.build_basic_info_window()

    def set_camera_pose(self, cam_pose: sapien.Pose):
        rpy = quat2euler(cam_pose.q)
        self.set_camera_xyz(*cam_pose.p)
        self.set_camera_rpy(rpy[0], -rpy[1], -rpy[2])

    def set_fovy(self, fovy):
        self.fovy = fovy
        self.window.set_camera_parameters(0.1, 100, fovy)

    def build_basic_info_window(self):
        self.basic_info_window = R.UIWindow().Label("Info").append(R.UIDisplayText())
        self.basic_info_window.get_children()[0].Text(f"Camera Name: {self.camera_name}")
        self.basic_info_window.append(
            R.UIDisplayText().Text("XYZ:"),
            R.UIInputFloat()
                .Label("x##cameraposx")
                .Value(self.fps_camera_controller.xyz[0])
                .ReadOnly(True),
            R.UIInputFloat()
                .Label("y##cameraposy")
                .Value(self.fps_camera_controller.xyz[1])
                .ReadOnly(True),
            R.UIInputFloat()
                .Label("z##cameraposz")
                .Value(self.fps_camera_controller.xyz[2])
                .ReadOnly(True),
            R.UIDisplayText().Text("RPY:"),
            R.UIInputFloat()
                .Label("r##camerarotr")
                .Value(self.fps_camera_controller.rpy[0])
                .ReadOnly(True),
            R.UIInputFloat()
                .Label("p##camerarotp")
                .Value(self.fps_camera_controller.rpy[1])
                .ReadOnly(True),
            R.UIInputFloat()
                .Label("y##cameraroty")
                .Value(self.fps_camera_controller.rpy[2])
                .ReadOnly(True),
        )

    def render(self):
        if self.closed:
            return

        self.scene.update_render()
        self.window.renderer("Color", [self.basic_info_window])

        if self.window.key_down("q") or self.window.should_close:
            self.close()
            return
