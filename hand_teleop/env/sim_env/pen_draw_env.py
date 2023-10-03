import numpy as np
import transforms3d
import sapien.core as sapien
import sys 
sys.path.append("/teleop")

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.ycb_object_utils import YCB_SIZE, load_ycb_object

PEN_ID = 101796
BUCKET_ID = 102254

def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names =  [joint.name for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}


class white_board:
    def __init__(
        self,
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size,
        color=None,
        density=1000.0,
        physical_material: sapien.PhysicalMaterial = None,
        name='',
        saturate_distance = 0.1,
        point_radius = 0.01,
        num_points = 10000
    ):
        """Create a sphere."""
        self.half_size = np.array(half_size)
        self.scene = scene
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=self.half_size, material=physical_material, density=density)
        builder.add_box_visual(half_size=self.half_size, color=color)
        self.board = builder.build_kinematic(name=name)
        self.board.set_pose(pose)
        # NOTE: 10000 magic number. Maximum number of points on white board
        self.pixels = np.zeros((num_points,3))
        self.pixel_ptr = 0
        self.saturate_distance = saturate_distance
        self.point_radius = point_radius
        self.points = []

    def create_point(self, pose: np.array, color=[255,255,0]):
        if self.pixel_ptr >= self.pixels.shape[0]:
            temp = self.pixels 
            self.pixels = np.zeros((self.pixels.shpae[0] + 10000, 3))
            self.pixels[:self.pixel_ptr,:] = temp 
        if self.pixel_ptr == 0 or not np.any(np.linalg.norm(self.pixels[:self.pixel_ptr,:]-pose,axis=1) <= self.saturate_distance): 
            self.pixels[self.pixel_ptr] = pose
            self.pixel_ptr += 1
            draw_builder = self.scene.create_actor_builder()
            draw_builder.add_sphere_visual(radius=self.point_radius, color=np.array(color))
            pixel = draw_builder.build_kinematic()
            pixel.set_pose(sapien.Pose(pose))
            self.points.append(pixel) 

    def reset_board(self):
        for point in self.points:
            self.scene.remove_actor(point)
        self.points = []
        self.pixels *= 0.0
        self.pixel_ptr = 0
    
    #convert from whiteboard frame to world frame 
    def world_to_board(pose:sapien.Pose):
        raise NotImplementedError

class PenDrawEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, use_visual_obs = False, no_rgb=False, need_offscreen_render=False, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, no_rgb = no_rgb, need_offscreen_render = need_offscreen_render, **renderer_kwargs)
        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)
        # Initial positions of pen
        #NOTE: currently may not be a good initialization. Pen drops on table
        self.pen_pose = sapien.Pose(np.array([-0.1, 0, -0.3]), transforms3d.euler.euler2quat(np.pi, np.pi / 2, 0))
        # Load table and keyboard
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, -0.4]))
        self.pen = self.load_partnet_obj(PEN_ID,scale = 0.1, material_spec=[1,1,0], density=100)
        self.white_board = white_board(self.scene,sapien.Pose([0.2,0.0,0.0]),[0.01,0.4,0.4],[1,1,1],name="white_board")
        self.pen.set_pose(self.pen_pose)
        self.pen_trajectory = []
        self.manipulated_object = self.pen 

    def post_step(self):
        contacts = self.scene.get_contacts()
        for contact in contacts:
            for point in contact.points:
                pose = point.position
                if contact.actor0.name == 'white_board' and contact.actor1.name=="pen_tip":
                    self.white_board.create_point(pose)
        return super().post_step()

    def reset_env(self):
        self.pen.set_pose(self.pen_pose)
        self.white_board.reset_board()

    def get_oracle_state(self):
        robot_qpos_vec = self.robot.get_qpos()
        pen_center_pose = self.manipulated_object.get_pose()
        object_pose_vec = np.concatenate([pen_center_pose.p, pen_center_pose.q])
        v = self.manipulated_object.get_velocity()
        w = self.manipulated_object.get_angular_velocity()
        #NOTE: currently returning robot q pose, pen pose and rot, pen linear v, pen angular w
        #May be of dimension [q pose dim] + 7 + 3 + 3 dimensional observation
        return np.concatenate([robot_qpos_vec, object_pose_vec, v, w])

def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = PenDrawEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()

# Player for pen draw environment. Derived from MugFilp Player.

# class PenDrawEnvPlayer(DataPlayer):
#     def __init__(self, meta_data: Dict[str, Any], data: Dict[str, Any], env: penDrawEnv,
#                  zero_joint_pos: Optional[np.ndarray] = None):
#         super().__init__(meta_data, data, env, zero_joint_pos)

#     def bake_demonstration(self, retargeting: Optional[PositionRetargeting] = None, method="tip_middle", indices=None):
#         use_human_hand = self.human_robot_hand is not None and retargeting is not None
#         baked_data = dict(obs=[], action=[])
#         manipulated_object = self.env.manipulated_object

#         for i in range(self.meta_data["data_len"]):
#             self.scene.step()
#             self.scene.unpack(self.get_sim_data(i))
#             contact_finger_index = self.human_robot_hand.check_contact_finger([manipulated_object])

#             # Robot qpos
#             qpos = self.env.robot.get_qpos()
#             baked_data["robot_qpos"].append(qpos)
#             self.env.robot.set_qpos(qpos)
#             if i >= 1:
#                 baked_data["action"].append(self.compute_action_from_states(baked_data["robot_qpos"][i - 1], qpos,
#                                                                             np.sum(contact_finger_index) > 0))
#             if i >= 2:
#                 duration = self.env.frame_skip * self.scene.get_timestep()
#                 finger_qvel = (baked_data["action"][-1][6:] - baked_data["action"][-2][6:]) / duration
#                 root_qvel = baked_data["action"][-1][:6]
#                 self.env.robot.set_qvel(np.concatenate([root_qvel, finger_qvel]))

#             # Environment observation
#             baked_data["obs"].append(self.env.get_observation())

#             # Environment state
#             baked_data["state"].append(self.collect_env_state([manipulated_object]))

#         baked_data["action"].append(baked_data["action"][-1])
#         return baked_data