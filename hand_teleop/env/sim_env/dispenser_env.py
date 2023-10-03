from pathlib import Path

import numpy as np
import sapien.core as sapien

from env.base import BaseSimulationEnv
from utils.common_robot_utils import load_robot
from utils.model_utils import create_visual_material
from utils.scene_utils import get_unique_contact

VALID_PARTNET_ID = ["103619"]


class DispenserEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=103619, **renderer_kwargs):
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
        self.table = self.create_table(table_height=0.6, table_half_size=[0.3, 0.3, 0.025])
        self.dispenser = self.load_dispenser()
        self.dispenser.set_pose(sapien.Pose([0, 0.1, 0.085], [0, 0, 0, 1]))
        self.dispenser.set_qpos(np.zeros(self.dispenser.dof))
        self.functional_region = self.load_functional_region()
        self.functional_region.set_pose(sapien.Pose([0, -0.1, 0.2]))

        # Load robot
        # init_qpos = [-0.08, -0.023, 0.239, 1.57, 1.57, 0] + [0, 1.05, 1.57, 1.2] + [0, 1.23, 1.57, 1.12]
        # init_qpos += [0, 0.92, 1.57, 1.27] + [0, -0.43, 0.87, 1.57, 1.38] + [-1, 0.9, 0, 0, 0]
        # self.robot = load_robot(self.renderer, self.scene, "adroit_free")
        # for joint in self.robot.get_active_joints():
        #     joint.set_drive_property(10000, 2000)
        # self.robot.set_qpos(np.array(init_qpos))
        # self.robot.set_drive_target(np.array(init_qpos))

    def load_dispenser(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.1
        loader.fix_root_link = False
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        builder = loader.load_file_as_articulation_builder(str(vhacd_urdf), config=config)
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)
        dispenser = builder.build(False)

        # for i in range(dispenser.dof - 1):
        #     joint = dispenser.get_active_joints()[i]
        #     joint.set_limits(np.zeros([1, 2]))

        return dispenser

    def load_functional_region(self):
        viz_mat = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                         base_color=np.array([1, 0, 0, 0.6]))
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=np.array([0.04, 0.04, 0.06]), material=viz_mat)
        return builder.build_static("functional_region")

    def pre_step(self):
        region_pos = self.functional_region.get_pose().p
        dispenser_pos = self.dispenser.get_pose().p
        locate_at_region = np.linalg.norm(dispenser_pos - region_pos) < 0.1
        joint_trigger = self.dispenser.get_qpos()[0] > 0.3
        mat = self.functional_region.get_visual_bodies()[0].get_render_shapes()[0].material
        if locate_at_region:
            if joint_trigger:
                print("green")
                mat.set_base_color(np.array([0, 1, 0, 0.6]))
            else:
                print("blue")
                mat.set_base_color(np.array([0, 0, 1, 0.6]))
        else:
            mat.set_base_color(np.array([1, 0, 0, 0.6]))
        self.functional_region.get_visual_bodies()[0].get_render_shapes()[0].set_material(mat)


class KinematicsDispenserEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, partnet_mobility_id=103619, **renderer_kwargs):
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
        self.table = self.create_table(table_height=0.6, table_half_size=[0.3, 0.3, 0.025])
        self.dispenser = self.load_dispenser()
        self.dispenser.set_pose(sapien.Pose([0, 0.1, 0.08], [0, 0, 0, 1]))
        self.dispenser.set_qpos(np.zeros(self.dispenser.dof))
        self.functional_region = self.load_functional_region()
        self.functional_region.set_pose(sapien.Pose([0, -0.1, 0.2]))

        # Load robot
        init_qpos = [-0.08, -0.023, 0.239, 1.57, 1.57, 0] + [0, 1.05, 1.57, 1.2] + [0, 1.23, 1.57, 1.12]
        init_qpos += [0, 0.92, 1.57, 1.27] + [0, -0.43, 0.87, 1.57, 1.38] + [-1, 0.9, 0, 0, 0]
        self.robot = load_robot(self.renderer, self.scene, "adroit_free")
        # for joint in self.robot.get_active_joints():
        #     joint.set_drive_property(10000, 2000)
        self.robot.set_qpos(np.array(init_qpos))
        self.palm_link = [link for link in self.robot.get_links() if link.get_name() == "palm"][0]
        # self.robot.set_drive_target(np.array(init_qpos))

        self.init_height = self.dispenser.get_pose().p[2]
        limit = [0.03] * 3 + [0.1] * 3 + [0.2] * 22
        self.lower_limit = -np.array(limit)
        self.upper_limit = np.array(limit)

        finger_link_names = ["ffdistal", "mfdistal", "rfdistal"]
        self.finger_tip_links = [link for link in self.robot.get_links() if link.get_name() in finger_link_names][:3]
        self.base_contact = True

    def load_dispenser(self):
        data_root = Path(__file__).parent.parent / "partnet-mobility-dataset" / str(self.partnet_mobility_id)
        vhacd_urdf = data_root.joinpath('mobility.urdf')

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.1
        loader.fix_root_link = False
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        builder = loader.load_file_as_articulation_builder(str(vhacd_urdf), config=config)
        for link in builder.get_link_builders():
            link.set_collision_groups(1, 1, 4, 4)
        dispenser = builder.build(fix_root_link=True)

        for i in range(dispenser.dof - 1):
            joint = dispenser.get_active_joints()[i]
            joint.set_limits(np.zeros([1, 2]))

        return dispenser

    def load_functional_region(self):
        viz_mat = create_visual_material(self.renderer, 0.1, 0.5, 0.4,
                                         base_color=np.array([1, 0, 0, 0.6]))
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=np.array([0.04, 0.04, 0.06]), material=viz_mat)
        return builder.build_static("functional_region")

    def pre_step(self):
        region_pos = self.functional_region.get_pose().p
        dispenser_pos = self.dispenser.get_pose().p
        locate_at_region = np.linalg.norm(dispenser_pos - region_pos) < 0.1
        joint_trigger = self.dispenser.get_qpos()[2] > 0.3
        mat = self.functional_region.get_visual_bodies()[0].get_render_shapes()[0].material
        if locate_at_region:
            if joint_trigger:
                print("green")
                mat.set_base_color(np.array([0, 1, 0, 0.6]))
            else:
                print("blue")
                mat.set_base_color(np.array([0, 0, 1, 0.6]))
        else:
            mat.set_base_color(np.array([1, 0, 0, 0.6]))
        self.functional_region.get_visual_bodies()[0].get_render_shapes()[0].set_material(mat)

    def step(self, action):
        robot_qpos = self.robot.get_qpos()
        action = np.clip(action, self.lower_limit, self.upper_limit)

        new_robot_qpos = robot_qpos + action
        self.robot.set_qpos(new_robot_qpos)
        if self.base_contact:
            object_pose = self.dispenser.get_pose()
            object_pos = object_pose.p
            object_pos += action[:3]
            self.dispenser.set_pose(sapien.Pose(object_pos, object_pose.q))

        handle_pos = self.dispenser.get_links()[-1].pose.p
        base_pos = self.dispenser.get_links()[1].pose.p
        for finger_link in self.finger_tip_links:
            pos = finger_link.pose.p
            if np.linalg.norm(pos - handle_pos) < 0.01:
                self.base_contact = True
            # else:
            #     self.base_contact = False

        reward, done = self.compute_reward()
        return None, reward, done, dict()

    def compute_reward(self):
        object_pose = self.dispenser.get_pose()
        palm_pose = self.palm_link.get_pose()

        reward = -0.1 * min(np.linalg.norm(palm_pose.p - object_pose.p), 0.5)
        done = False

        if True:
            reward += 0.1
            lift = min(object_pose.p[2], self.functional_region.pose.p[2]) - self.init_height
            lift = max(lift, 0)
            reward += 5 * lift
            if lift > 0.015:
                reward += 2
                obj_target_distance = min(np.linalg.norm(object_pose.p - self.functional_region.pose.p), 0.5)
                reward += -1 * min(np.linalg.norm(palm_pose.p - self.functional_region.pose.p), 0.5)
                reward += -3 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.1:
                    reward += (0.1 - obj_target_distance) * 20
                    if obj_target_distance < 0.005:
                        done = True

        return reward, done

    @property
    def action_dim(self):
        return self.robot.dof

    def set_state(self, state):
        self.scene.unpack(state)

    def get_state(self):
        return self.scene.pack()


class MPC:
    def __init__(self, env: KinematicsDispenserEnv, plan_horizon=8, popsize=200, num_elites=20, max_iters=4,
                 use_mpc=True):
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon = plan_horizon
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.action_dim = 28
        self.n = self.action_dim * self.plan_horizon
        self.reset()

        self.mean = None
        self.std = None

    def reset(self):
        self.mean = np.zeros((self.plan_horizon * self.action_dim))
        self.std = 0.5 * np.ones((self.plan_horizon * self.action_dim))

    def cem_optimize(self, state):
        mean = self.mean.copy()
        std = self.std.copy()
        initial_state = state.copy()
        env = self.env
        for k in range(self.max_iters):
            rs = []
            actions = []
            for _ in range(self.popsize):
                r = 0
                action = np.random.normal(mean, std, self.n)
                env.set_state(initial_state)
                for i in range(self.plan_horizon):
                    action_start = i * self.action_dim
                    obs, reward, done, info = env.step(action[action_start: action_start + self.action_dim])
                    if done:
                        break
                    r += reward
                rs.append(r)
                actions.append(action)
            actions = np.array(actions)
            rs = np.array(rs)
            top_k = rs.argsort()[-self.num_elites:]
            top_k_actions = actions[top_k, :]
            mean = top_k_actions.mean(0)
            std = np.std(top_k_actions, axis=0)

        return mean, std

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        if not self.use_mpc:
            if t % self.plan_horizon == 0:
                self.reset()
                mean, std = self.cem_optimize(state)
                self.mean = mean
                return mean[:self.action_dim]
            else:
                num = (t % self.plan_horizon) * self.action_dim
                return self.mean[num:num + self.action_dim]
        else:
            self.reset()
            mean, std = self.cem_optimize(state)
            return mean[: self.action_dim]


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = DispenserEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not env.viewer.closed:
        env.simple_step()
        env.render()

    # while not viewer.closed:
    #     env.simple_step()
    #     env.render()
    # robot_qpos = env.robot.get_drive_target()
    # robot_qpos[0] -=0.001
    # env.robot.set_drive_target(robot_qpos)


if __name__ == '__main__':
    env_test()
    # env = KinematicsDispenserEnv()
    # env.simple_step()
    #
    # init_state = env.get_state()
    # env.set_state(init_state)
    #
    # cem_mpc = MPC(env, use_mpc=True)
    # states = []
    # actions = []
    # next_states = []
    # infos = []
    #
    # on_screen = True
    # if on_screen:
    #     from sapien.utils import Viewer
    #     from constructor import add_default_scene_light
    #
    #     viewer = Viewer(env.renderer)
    #     viewer.set_scene(env.scene)
    #     add_default_scene_light(env.scene, env.renderer)
    #     env.viewer = viewer
    #
    # for t in range(200):
    #     state = env.get_state()
    #     action = cem_mpc.act(state, t)
    #     states.append(state)
    #     actions.append(action)
    #     obs, reward, done, info = env.step(action)
    #     next_states.append(obs)
    #     infos.append(infos)
    #     print("---step #%d----" % t)
    #     print("reward: %f" % reward)
    #     if on_screen:
    #         env.render()
    #     if done:
    #         print("Success")
    #         break

