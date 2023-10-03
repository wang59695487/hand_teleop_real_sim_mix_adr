from pathlib import Path

import numpy as np
import sapien.core as sapien

from hand_teleop.env.sim_env.base import BaseSimulationEnv
# from utils.common_robot_utils import load_robot

VALID_PARTNET_ID = ["9410", "9281", "9164", "8936", "8897"]


class DoorEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=1000, **renderer_kwargs):
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
        self.door, self.handle_joint, self.board_joint = self.load_door()
        self.door.set_pose(sapien.Pose([0, 0, -0.1], [0, 0, 0, 1]))
        self.door_original_limit = self.board_joint.get_limits()
        self.board_joint.set_limits(np.zeros([1, 2]))
        self.handle_joint.set_drive_property(0.2, 0.01)
        self.handle_active = False

    def load_door(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 1
        loader.fix_root_link = True
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        door = loader.load(str(vhacd_urdf), config=config)
        handle_joint = door.get_active_joints()[-1]
        board_joint = door.get_active_joints()[-2]
        return door, handle_joint, board_joint

    def pre_step(self):
        if self.door.get_qpos()[-1] > 1.3:
            self.handle_active = True
            self.board_joint.set_limits(self.door_original_limit)
        else:
            if not self.handle_active:
                pass
            else:
                self.handle_active = False
                board_angle = self.door.get_qpos()[0]
                if abs(board_angle) < 0.03:
                    self.board_joint.set_limits(np.ones([1, 2]) * board_angle)


# class DoorEnvHeuristics:
#     def __init__(self, env: DoorEnv):
#         self.env = env
#         self.robot = load_robot(self.env.renderer, self.env.scene, "adroit_free")
#         self.robot.set_pose(sapien.Pose([-0.4, 0, 0]))
#         for joint in self.robot.get_active_joints():
#             joint.set_drive_property(10000, 500)

#         self.palm_link = [link for link in self.robot.get_links() if link.get_name() == "palm"][0]
#         self.hand_link = self.env.door.get_links()[-1]
#         self.stage = 0
#         self.current_qpos_target = None
#         self.previous_qpos_target = None

#     def get_handle_mesh_center(self):
#         vertices = []
#         for visual_body in self.hand_link.get_visual_bodies():
#             for render_shape in visual_body.get_render_shapes():
#                 v = render_shape.mesh.vertices * visual_body.scale
#                 pose = self.hand_link.get_pose() * visual_body.local_pose
#                 pose_mat = pose.to_transformation_matrix()
#                 v_global = v @ pose_mat[:3, :3].T + pose_mat[:3, 3][None, :]
#                 vertices.append(v_global)
#         all_v = np.concatenate(vertices, axis=0)
#         center = all_v.mean(0)
#         return center

#     def solve(self):
#         center = self.get_handle_mesh_center()
#         distance_sign = np.sign(self.palm_link.get_pose().p[0] - center[0])
#         finger_joint_indices = np.array([7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22])
#         joint_limit = self.robot.get_qlimits()
#         if self.stage == 0:
#             pre_grasp_pos = center + distance_sign * np.array([0.20, 0.02, 0]) + np.array(
#                 [0, 0, -0.01]) - self.robot.get_pose().p
#             pre_grasp_euler = np.array([0, np.pi / 3, 0])
#             qpos = self.robot.get_qpos()
#             qpos[:6] = np.concatenate([pre_grasp_pos, pre_grasp_euler])
#             qpos[finger_joint_indices] = 0.4
#             self.current_qpos_target = np.clip(qpos, joint_limit[:, 0], joint_limit[:, 1])
#         elif self.stage == 1:
#             qpos = self.robot.get_qpos()
#             qpos[4] = np.pi / 12
#             qpos[2] -= 0.07
#             qpos[0] += distance_sign * (-0.06)
#             qpos[finger_joint_indices] = 1.2
#             self.current_qpos_target = np.clip(qpos, joint_limit[:, 0], joint_limit[:, 1])
#         elif self.stage == 2:
#             qpos = self.robot.get_qpos()
#             qpos[3] += np.pi / 2
#             qpos[2] += 0.03
#             qpos[1] += 0.15
#             self.current_qpos_target = np.clip(qpos, joint_limit[:, 0], joint_limit[:, 1])
#         elif self.stage == 3:
#             qpos = self.robot.get_qpos()
#             # qpos[4] = np.pi / 2
#             # qpos[0] -= 0.6
#             # qpos[1] -= 0.6
#             # self.current_qpos_target = np.clip(qpos, joint_limit[:, 0], joint_limit[:, 1])

#     def move_to_target_qpos(self, step_size=0.04):
#         current_qpos = self.robot.get_qpos()
#         diff_qpos = self.current_qpos_target - current_qpos
#         diff_qpos_dir = diff_qpos / np.linalg.norm(diff_qpos)
#         target_qpos = diff_qpos_dir * step_size + current_qpos
#         self.robot.set_drive_target(target_qpos)

#         if np.linalg.norm(current_qpos - self.current_qpos_target) < 0.02:
#             return True
#         else:
#             return False

#     def step(self):
#         # Pre grasp
#         print(self.stage)
#         if self.stage == 0:
#             finish = self.move_to_target_qpos()
#             if finish:
#                 self.stage += 1
#                 self.solve()
#         elif self.stage == 1:
#             self.previous_qpos_target = self.robot.get_drive_target()
#             finish = self.move_to_target_qpos()
#             finish = finish or np.linalg.norm(self.previous_qpos_target - self.robot.get_drive_target()) < 1e-3
#             if finish:
#                 self.stage += 1
#                 self.solve()
#         elif self.stage == 2:
#             self.move_to_target_qpos()
#             finish = self.env.handle_active
#             if finish:
#                 self.stage += 1
#                 self.solve()
#         elif self.stage == 3:
#             door_length = 0.65
#             palm_length = 0.1
#             theta = self.env.door.get_qpos()[0]
#             board_angle = theta + 0.01
#             qpos = self.robot.get_qpos()
#             qpos[4] = board_angle
#             qpos[0] = self.current_qpos_target[0] - door_length * np.sin(board_angle) - np.sin(
#                 board_angle) * palm_length
#             qpos[1] = self.current_qpos_target[1] + door_length * (np.cos(board_angle) - 1) - np.sin(
#                 board_angle) * palm_length
#             self.robot.set_drive_target(qpos)
#             finish = board_angle > 1
#             if finish:
#                 self.stage += 1

#         pass


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = DoorEnv(partnet_mobility_id=9410)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


# def heuristic_test():
#     from sapien.utils import Viewer
#     from constructor import add_default_scene_light
#     env = DoorEnv(partnet_mobility_id=9410)
#     solution = DoorEnvHeuristics(env)

#     viewer = Viewer(env.renderer)
#     viewer.set_scene(env.scene)
#     add_default_scene_light(env.scene, env.renderer)
#     env.viewer = viewer

#     solution.solve()
#     while not viewer.closed:
#         env.simple_step()
#         env.render()
#         solution.step()


if __name__ == '__main__':
    env_test()
