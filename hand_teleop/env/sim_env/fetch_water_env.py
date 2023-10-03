from pathlib import Path

import numpy as np
import sapien.core as sapien

from env.base import BaseSimulationEnv
from env.constructor import download_maniskill
from utils.ycb_object_utils import load_ycb_object
from utils.model_utils import create_visual_material


class FetchWaterEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, coffee_machine_id=1000,
                 **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        self.partnet_mobility_id = coffee_machine_id

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Load table and drawer
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, -0.4]))
        self.coffee_machine = self.load_coffee_machine()
        self.coffee_machine.set_pose(sapien.Pose([0, 0.2, -0.23]))

        # Load mug
        self.target_object = load_ycb_object(self.scene, "mug")
        self.target_object.set_pose(sapien.Pose([0, -0.2, -0.4 + 0.1], [0.707, 0, 0, 0.707]))

        # Init coffee maker
        self.inv_flow_rate = 5
        self.flow_step = 0
        self.max_particle_num = 20
        self.trigger_this_step = False
        self.joint_trigger_info = dict(joint_name="joint_5", trigger_value=1.4, larger_than=True)
        joint_names = [joint.get_name() for joint in self.coffee_machine.get_active_joints()]
        self.trigger_joint_index = joint_names.index(self.joint_trigger_info["joint_name"])

        # Init coffee property
        self.coffee_visual_material = create_visual_material(self.renderer, 0.5, 0, 0.4,
                                                             np.array([212, 241, 249, 100]) / 255)
        coffee_output_link_name = "link_5"
        self.coffee_output_link = [link for link in self.coffee_machine.get_links() if
                                   link.get_name() == coffee_output_link_name][0]
        self.water_particles = []

    def load_coffee_machine(self):
        mobility_root = Path("/home/sim/sapien_resources/mobility_dataset/mobility_convex_alpha5")
        urdf = str(mobility_root / str(self.partnet_mobility_id) / "mobility.urdf")

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.25
        loader.fix_root_link = True
        material = self.scene.create_physical_material(1, 1, 0)

        config = {'material': material, 'density': 1000}
        drawer = loader.load(urdf, config=config)
        return drawer

    def make_coffee(self):
        self.flow_step += 1
        if self.flow_step % self.inv_flow_rate != 0:
            return

        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.01, material=self.coffee_visual_material)
        builder.add_sphere_collision(radius=0.01)
        water_particle = builder.build(f"water_particle_{self.flow_step // self.inv_flow_rate}")
        coffee_output_link_pose = self.coffee_output_link.get_pose()
        coffee_output_link_pose.set_p(coffee_output_link_pose.p - np.array([0, 0, 0.03]))
        water_particle.set_pose(coffee_output_link_pose)
        self.water_particles.append(water_particle)

    def pre_step(self):
        joint_value = self.coffee_machine.get_qpos()[self.trigger_joint_index]
        if joint_value > self.joint_trigger_info["trigger_value"] and self.joint_trigger_info["larger_than"]:
            trigger = True
        elif joint_value < self.joint_trigger_info["trigger_value"] and not self.joint_trigger_info["larger_than"]:
            trigger = True
        else:
            trigger = False
        self.trigger_this_step = trigger
        if trigger and len(self.water_particles) < self.max_particle_num:
            self.make_coffee()


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = FetchWaterEnv(coffee_machine_id=103046)
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
