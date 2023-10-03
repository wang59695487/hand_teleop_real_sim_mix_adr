import numpy as np
import sapien.core as sapien

from hand_teleop.env.sim_env.base import BaseSimulationEnv
from hand_teleop.utils.ycb_object_utils import load_ycb_object

def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names =  [joint.name for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}

class LaptopEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, test_can_fall=True, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.004)

        # Load table and keyboard
        self.table = self.create_table(table_height=0.6, table_half_size=[0.4, 0.4, 0.025])
        self.table.set_pose(sapien.Pose([0, 0, -0.4]))
        self.laptop = self.load_partnet_obj(10238, scale = 0.3, material_spec=[1,1,0], density=100, fix_root=True)
        self.manipulated_object = self.laptop
        self.laptop.set_pose(sapien.Pose([0, 0, -0.3]))
        self.laptop_joints = self.laptop.get_joints()
        self.laptop.set_qpos(-1*np.ones(1))
        # qf = self.laptop.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        # self.laptop.set_qf(qf)
        for lap_joint in self.laptop_joints:
            lap_joint.set_drive_property(stiffness=0.2, damping=2)

    def post_step(self):
        # self.compensate_gravity()
        return super().post_step()

    def compensate_gravity(self):
        #@NOTE: Hack to make the keyboard have spring
        qf = self.laptop.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        self.laptop.set_qf(qf)

    def create_table(self, table_height=1.0, table_half_size=(0.8, 0.8, 0.025)):
        builder = self.scene.create_actor_builder()

        # Top
        top_pose = sapien.Pose([0, 0, -table_half_size[2]])
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        if self.renderer:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_diffuse_texture_from_file(
                "assets/misc/table_map.jpg")
            table_visual_material.set_roughness(0.3)
            builder.add_visual_from_file("assets/misc/cube.obj",
                                         pose=top_pose, material=table_visual_material, scale=table_half_size)
            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1
            builder.add_box_visual(pose=sapien.Pose([x, y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([x, -y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([-x, y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([-x, -y, leg_height]), half_size=leg_size,
                                   material=table_visual_material)
        return builder.build_static("table")
        
    def reset_env(self):
        # self.laptop.set_qpos(-np.ones(1))
        # random_xy = (np.random.rand(2) * 2 - 1) * 0.05
        # random_pos = np.concatenate([random_xy, [0.01]])
        # self.laptop.set_pose(sapien.Pose(random_pos))
        self.laptop.set_pose(sapien.Pose([0, 0, -0.3]))
       

def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = LaptopEnv()
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
